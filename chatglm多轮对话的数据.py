#==========定义知识样本.######先处理我们的数据.
from argparse import Namespace
cfg = Namespace()
cfg.batch_size=1
cfg.prompt_column = 'prompt'                  #========首先预料里面的3个列名在这里定义好.
cfg.response_column = 'response'
cfg.history_column = 'history'
cfg.source_prefix = '' #添加到每个prompt开头的前缀引导语
cfg.model_name_or_path = 'THUDM/chatglm2-6b'  #远程'THUDM/chatglm-6b' 
cfg.quantization_bit = None #仅仅预测时可以选 4 or 8 


from torch.utils.data import Dataset,DataLoader 
import transformers
from transformers import  AutoModel,AutoTokenizer,AutoConfig,DataCollatorForSeq2Seq


import pandas as pd 
# keyword = '梦中情炉'

# description = '''梦中情炉一般指的是炼丹工具torchkeras。
# 这是一个通用的pytorch模型训练模版工具。
# torchkeras是一个三好炼丹炉：好看，好用，好改。
# 她有torch的灵动，也有keras的优雅，并且她的美丽，无与伦比。
# 所以她的作者一个有毅力的吃货给她取了一个别名叫做梦中情炉。'''


#==========如果数据没有histroy那么就写空列表即可. 否则写一个二级列表. 每一个元素是一个长为2的列表.
dataall=[{'prompt': '梦中情炉'
,
'response' : '''梦中情炉一般指的是炼丹工具torchkeras。
这是一个通用的pytorch模型训练模版工具。
torchkeras是一个三好炼丹炉：好看，好用，好改。
她有torch的灵动，也有keras的优雅，并且她的美丽，无与伦比。
所以她的作者一个有毅力的吃货给她取了一个别名叫做梦中情炉。''', 'history':[]

}


,{
    "prompt": "What can be done at a regulatory level to ensure, in a monopsony, that the power over the employees isn't abused? List a few options and focus on the organs that should act.",
    "input": "",
    "response": "Here are some potential regulatory options to address monopsony power over employees:\n\n- Antitrust enforcement agencies like the FTC or DOJ could investigate and take action against dominant employers that are abusing their market power to suppress wages or worsen working conditions. They could force divestitures or other structural remedies.\n- Labor standards agencies could set industry-specific minimum wage floors or other regulations around compensation and working conditions. This would reduce the ability of dominant employers to push wages/conditions below certain thresholds.\n- Sector-specific regulators could impose requirements around how companies hire, compensate, and treat workers. For example, banking regulators could prohibit abusive practices by dominant banks with regard to their employees.\n- Legislatures could pass laws",
    "history": [
      [
        "Can you write a short introduction about the relevance of the term \"monopsony\" in economics? Please use examples related to potential monopsonies in the labour market and cite relevant research.",
        "Monopsony refers to a market structure where there is only one buyer of a good or service. In the context of labor markets, a monopsony occurs when there is only one employer in a particular geographic area or industry, giving them significant power to set wages and working conditions. Research has shown that the existence of monopsonies can lead to lower wages and reduced job opportunities for workers. For example, a study by the National Bureau of Economic Research found that in industries with high levels of concentration, workers earn lower wages and are less likely to receive benefits such as health insurance."
      ]
    ]
  }

]




tokenizer = AutoTokenizer.from_pretrained(
    cfg.model_name_or_path, trust_remote_code=True)
#对prompt使用一些简单的数据增强的方法，以便更好地收敛。
def get_prompt_list(keyword):
    return [f'{keyword}', 
            f'你知道{keyword}吗?',
            f'{keyword}是什么？',
            f'介绍一下{keyword}',
            f'你听过{keyword}吗?',
            f'啥是{keyword}？',
            f'{keyword}是何物？',
            f'何为{keyword}？',
           ]





import datasets 
#训练集和验证集一样





prefix=cfg.source_prefix
def format_example(examples): # support question with a single answer or multiple answers
        for i in range(len(examples)):
            if examples[i][cfg.prompt_column] and examples[i][cfg.response_column]:
                query, answer = examples[i][cfg.prompt_column], examples[i][cfg.response_column]
                query = query + examples[i]["query"] if "query"in examples[i] else query
                history = examples[i][cfg.history_column] if examples[i][cfg.history_column] else []
                prompt = ""
                for j, (old_query, response) in enumerate(history):
                    prompt += "[Round {}]\n\n问：{}\n\n答：{}\n\n".format(j+1, old_query, response)
                prompt += "[Round {}]\n\n问：{}\n\n答：".format(len(history)+1, query)
                prompt = prefix + prompt
                yield prompt, answer
import json
# dataall=format_example(dataall)
# dataall=json.load( open('ChatGLM-Efficient-Tuning-main/data/oaast_rm_zh.json') )
dataall=format_example(dataall)

cfg.max_source_length = 128 
cfg.max_target_length = 128
data =[{'prompt':x[0],'response':x[1]} for x in dataall ]
dfdata = pd.DataFrame(data)
ds_train_raw = ds_val_raw = datasets.Dataset.from_pandas(dfdata)
#这是支持 history列处理，并且按照batch预处理数据的方法。

def preprocess(examples):
    max_seq_length = cfg.max_source_length + cfg.max_target_length
    model_inputs = {
        "input_ids": [],
        "labels": [],
    }
    for i in range(len(examples[cfg.prompt_column])):
        if examples[cfg.prompt_column][i] and examples[cfg.response_column][i]:
            query, answer = examples[cfg.prompt_column][i], examples[cfg.response_column][i]

            history = examples[cfg.history_column][i] if cfg.history_column is not None and cfg.history_column in examples else None
            prompt = tokenizer.build_prompt(query, history)

            prompt = cfg.source_prefix + prompt
            a_ids = tokenizer.encode(text=prompt, add_special_tokens=True, truncation=True,
                                     max_length=cfg.max_source_length)
            b_ids = tokenizer.encode(text=answer, add_special_tokens=False, truncation=True,
                                     max_length=cfg.max_target_length)

            context_length = len(a_ids)
            input_ids = a_ids + b_ids + [tokenizer.eos_token_id]
            labels = [tokenizer.pad_token_id] * context_length + b_ids + [tokenizer.eos_token_id]

            pad_len = max_seq_length - len(input_ids)
            input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
            labels = labels + [tokenizer.pad_token_id] * pad_len
            labels = [(l if l != tokenizer.pad_token_id else -100) for l in labels]
            model_inputs["input_ids"].append(input_ids)
            model_inputs["labels"].append(labels)
    return model_inputs


ds_train = ds_train_raw.map(
    preprocess,
    batched=True,
    num_proc=4,
    remove_columns=ds_train_raw.column_names
)

ds_val = ds_val_raw.map(
    preprocess,
    batched=True,
    num_proc=4,
    remove_columns=ds_val_raw.column_names
)
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=None,
    label_pad_token_id=-100,
    pad_to_multiple_of=None,
    padding=False
)

dl_train = DataLoader(ds_train,batch_size = cfg.batch_size,
                      num_workers = 2, shuffle = True, collate_fn = data_collator 
                     )
dl_val = DataLoader(ds_val,batch_size = cfg.batch_size,
                      num_workers = 2, shuffle = False, collate_fn = data_collator 
                     )
print('我们打印第一条训练数据',ds_train[0])
print('我们一共有训练数据',len(dl_train))