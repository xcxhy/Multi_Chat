'''
Descripttion: 推理代码
Author: xcxhy
version: V1
Date: 2023-08-29 09:56:24
LastEditors: xcxhy
LastEditTime: 2023-08-29 19:41:50
'''
import os
import torch
from transformers import GenerationConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

class Chat(object):
    def __init__(self, model_name, use_lora):
        self.model_name = model_name
        self.use_lora = use_lora
        self.tokenzier = AutoTokenizer.from_pretrained(self.model_name,trust_remote_code=True)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True,
                )
        self.streamer = TextStreamer(self.model)
        
    def generate(self, input, max_length=256, temperature=0.1, top_p=1, **kwargs):
        instruction = input

        batch = self.tokenizer(instruction, return_tensors="pt", add_special_tokens=False)
        inputs = batch
        input_ids = inputs["input_ids"].to("cuda")
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            early_stopping=True,
            **kwargs,
        )

        generate_params = {
            "input_ids": input_ids,
            "generation_config": generation_config,
            "return_dict_in_generate": True,
            "output_scores": True,
            "max_new_tokens": max_length,
        }

        output = self.model.generate(streamer=self.streamer ,pad_token_id=self.tokenizer.eos_token_id,**generate_params).sequences[0]

        return  self.tokenizer.decode(output, skip_special_tokens=True)
    
if __name__=="__main__":
    
    model = Chat("",False)

    input = "你好"
    kwargs = {}
    generator = model.generate(instruction=input,max_length=50,**kwargs)
    print(generator)



