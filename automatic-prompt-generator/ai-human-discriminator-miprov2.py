import dspy
import numpy as np
from typing import Literal
from modelscope.msdatasets import MsDataset

# 加载数据，划分train/val/test
ds = MsDataset.load('simpleai/HC3-Chinese', subset_name='psychology', split='train', cache_dir='ms_data/')
ds = ds.shuffle(seed=42)

train = [dspy.Example(question=ds[i]['question'], answer=ds[i]['human_answers'][0], label='human').with_inputs('question', 'answer') for i in range(5)]
train += [dspy.Example(question=ds[i]['question'], answer=ds[i]['chatgpt_answers'][0], label='ai').with_inputs('question', 'answer') for i in range(5)]
np.random.shuffle(train)

val = [dspy.Example(question=ds[i]['question'], answer=ds[i]['human_answers'][0], label='human').with_inputs('question', 'answer') for i in range(5, 55)]
val += [dspy.Example(question=ds[i]['question'], answer=ds[i]['chatgpt_answers'][0], label='ai').with_inputs('question', 'answer') for i in range(5, 55)]
np.random.shuffle(val)

test = [dspy.Example(question=ds[i]['question'], answer=ds[i]['human_answers'][0], label='human').with_inputs('question', 'answer') for i in range(55, 105)]
test += [dspy.Example(question=ds[i]['question'], answer=ds[i]['chatgpt_answers'][0], label='ai').with_inputs('question', 'answer') for i in range(55, 105)]



# 定义metric、evals
def TaskMetric(example: dspy.Example, prediction: dspy.Prediction, trace=None):
    return prediction.label == example.label

evaluate_correctness = dspy.Evaluate(
    devset=test,
    metric=TaskMetric,
    num_threads=2,
    display_progress=True,
    display_table=False,
)

# 配置task lm
model = 'Qwen/Qwen2.5-Coder-7B-Instruct'
api_key = ''
api_base = ''

lm = dspy.LM(f'openai/{model}',
             api_key=api_key,
             api_base=api_base,
             temperature=0,
             cache=False)
dspy.configure(lm=lm)

# 配置分类器
class Task(dspy.Signature):
    question: str = dspy.InputField()
    answer: str = dspy.InputField()
    label: Literal['human', 'ai'] = dspy.OutputField()

classifier = dspy.ChainOfThought(Task)

# 测试test
metric = evaluate_correctness(classifier, devset=test)
print('before optimize', metric)
dspy.inspect_history(n=1)

# 优化program
# prompt/teacher LLM
big_lm = dspy.LM('openai/Qwen/Qwen2.5-Coder-32B-Instruct',
                 api_key=api_key,
                 api_base=api_base,
                 temperature=0.8,
                 cache=False)

optimizer = dspy.MIPROv2(
    metric=TaskMetric,
    auto='light', #优化力度
    num_threads=2,
    prompt_model=big_lm, #写提示词的LLM
    init_temperature=0.8, #prompt LLM的temp
    teacher_settings=dict(lm=big_lm), #生成bootstrap examples的LLM
    seed=42,
    verbose=False, #是否显示优化过程
)

optimized_classifier = optimizer.compile(
    classifier,
    trainset=train,
    valset=val,
    max_bootstrapped_demos=0,
    max_labeled_demos=0,
    requires_permission_to_run=False,
    minibatch=False,
    minibatch_size=len(val),
)

# 存储optimized program
optimized_classifier.save('optimized.json')

# 解决中文乱码问题
import json
with open('optimized.json', 'r', encoding='utf-8') as f:
    tmp = json.loads(f.read())
    with open('optimized_zh.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(tmp, ensure_ascii=False, indent=2))

# 批量测试test
metric = evaluate_correctness(optimized_classifier, devset=test)
print('after optimize', metric)

dspy.inspect_history(n=1)