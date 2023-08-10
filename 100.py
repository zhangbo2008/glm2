# datasets的使用.最方便方法是从json加载, 
from datasets import load_dataset
import datasets
import datasets.arrow_dataset
if 1:
    from datasets import load_dataset

    dataset = load_dataset("rotten_tomatoes", split="train")
    print(1)

    dataset= dataset.select( range(10))
    dataset.to_json('debug.json')
# 修改的方法也是直接改json, 改完再加载回来, 低代码还是好用. 这样就加载回来了.修改自己去json文件修改即可.

dataset = load_dataset("json", data_files='debug.json')

print(1)
dataset.add_item({"text":"1111111big big the rock is destined to be the 21st century's new \" conan \" and that he's going to make a splash even greater than arnold schwarzenegger , jean-claud van damme or steven segal .","label":1})
print(1)

dataset.to_json('debug.json')









from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True).half().cuda()
model = model.eval()
response, history = model.chat(tokenizer, "你好", history=[])
print(response)
response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history)
print(response)
from torchkeras.chat import ChatGLM 
chatglm = ChatGLM(model,tokenizer)
ChatGLM(model,tokenizer,max_chat_rounds=20) 