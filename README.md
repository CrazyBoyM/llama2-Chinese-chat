# llama2-Chinese-chat   
更新记录:
- 2023-07-19 全网首个中文对话llama2 13b版本发布，实现可流畅中文对话 
- 2023-07-23 完成第2个epoch训练放出，有更好的对话体验
- 2023-08-03 分支版本：bimoGPT放出，拥有自我身份认知、不错的代码问答能力，权重下载地址：https://huggingface.co/shareAI/bimoGPT-llama2-13b
- 2023-08-21 更新世界模型排名榜，超越某号称“中文Llama2官方”社区的收费模型十多个名次。

![image](https://github.com/CrazyBoyM/llama2-Chinese-chat/assets/35400185/1aefbbf4-cd54-4ea2-a07a-c15f73a829ef)


llama2 Chinese chat - 本项目是一个教程记录整理的repo，旨在提供给新手的参照价值和开箱即用的中文LLaMa2对话体验。包含训练过程记录，各种主要量化方式，部署后端api的推荐方案，以及在一个具体的前端网页上实现开箱即用的流畅对话体验。  
也是当前第一个实际可用的中文13b llama2对话（已实现且放出实际文件）。
- 中文llama2对话模型下载（仅差分部分）：https://huggingface.co/shareAI/llama2-13b-Chinese-chat  
- 完整权重下载（最新的练到了第2个epoch）：https://www.codewithgpu.com/m/file/llama2-13b-Chinese-chat  
- 训练用数据集：https://huggingface.co/datasets/shareAI/ShareGPT-Chinese-English-90k
- llama2训练交流QQ群：443064756

更多实测问答记录：https://gist.github.com/goog/a3276016bd17eb401d6cc74d1f96d2e0  感谢@goog 提供
<img width="874" alt="image" src="https://github.com/CrazyBoyM/llama2-Chinese-chat/assets/35400185/634d8bba-d013-44bb-91c9-1623e3d4040a">
<img width="808" alt="image" src="https://github.com/CrazyBoyM/llama2-Chinese-chat/assets/35400185/7bf4d236-9da6-4033-b040-531abb09c101">


---
library_name: peft
datasets:
- shareAI/shareGPT_cn
- FreedomIntelligence/ShareGPT-CN
language:
- zh
pipeline_tag: question-answering
tags:
- chat
- llm
- llama2
- chatgpt
---
tip: 优秀对话llm的训练离不开高质量的多轮对话数据集，如果你也想成为志愿者  
欢迎加入QQ群：130920969，共同进行优质数据集的交流、收集和建设工作  

项目在中文sharegpt数据集上训练得到的llama2 Chinese chat 13b，为减轻文件大小负担这里只放出了adapter的权重  
请拉取https://huggingface.co/TheBloke/Llama-2-13B-fp16 作为基础权重，使用如下脚步执行合并得到可工作的总权重：  

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name_or_path = '/data/TheBloke/Llama-2-13B-fp16'
adapter_name_or_path = '/data/llama2-13b-Chinese-chat'
save_path = '/data/llama2-13b-Chinese-chat_v1'

tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path,
    trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    device_map='auto'
)
print("load model success")
model = PeftModel.from_pretrained(model, adapter_name_or_path)
print("load adapter success")
model = model.merge_and_unload()
print("merge success")

tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)
print("save done.")
```
合并后，体验对话：
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def main():
    model_name = '/data/llama2-13b-Chinese-chat_v1'

    device = 'cuda'
    max_new_tokens = 500    # 每轮对话最多生成多少个token
    history_max_len = 2000  # 模型记忆的最大token长度
    top_p = 0.9
    temperature = 0.35 # 越大模型越浪
    repetition_penalty = 1.2 # 如果模型出现重复说话可以调节该系数

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map='auto'
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        # llama不支持fast
        use_fast=False if model.config.model_type == 'llama' else True
    )
    # 记录所有历史记录
    history_token_ids = tokenizer('<s>', return_tensors="pt").input_ids

    # 开始对话
    user_input = input('User：')
    while True:
        user_input = '{}</s>'.format(user_input)
        user_input_ids = tokenizer(user_input, return_tensors="pt", add_special_tokens=False).input_ids
        history_token_ids = torch.concat((history_token_ids, user_input_ids), dim=1)
        model_input_ids = history_token_ids[:, -history_max_len:].to(device)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=model_input_ids, max_new_tokens=max_new_tokens, do_sample=True, top_p=top_p,
                temperature=temperature, repetition_penalty=repetition_penalty, eos_token_id=tokenizer.eos_token_id
            )
        model_input_ids_len = model_input_ids.size(1)
        response_ids = outputs[:, model_input_ids_len:]
        history_token_ids = torch.concat((history_token_ids, response_ids.cpu()), dim=1)
        response = tokenizer.batch_decode(response_ids)
        print("Bot：" + response[0].strip().replace('</s>', ""))
        user_input = input('User：')


if __name__ == '__main__':
    main()

```
推荐继续二次训练以针对性调优对话效果~ 
## Training procedure


The following `bitsandbytes` quantization config was used during training:
- load_in_8bit: False
- load_in_4bit: True
- llm_int8_threshold: 6.0
- llm_int8_skip_modules: None
- llm_int8_enable_fp32_cpu_offload: False
- llm_int8_has_fp16_weight: False
- bnb_4bit_quant_type: nf4
- bnb_4bit_use_double_quant: True
- bnb_4bit_compute_dtype: float16
### Framework versions


- PEFT 0.4.0.dev0
训练1个epoch，loss 0.9，实测用中文对话体验优于baichuan13b(仅主观感受)。还有很大潜力，建议作为底座把文件拉回去继续调优。

感谢：  
- LLaMA2  
- Firefly项目  
- shareGPT中文数据集的建设者们 
