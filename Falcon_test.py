import json
import os
from pprint import pprint
import bitsandbytes as bnb
import torch
import torch.nn as nn
import transformers
from datasets import load_dataset
from huggingface_hub import notebook_login
from peft import (
    LoraConfig,
    PeftConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training
)
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


PEFT_MODEL = "C:/Rt/RA/pr/trained-model/new_fal2"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)


config = PeftConfig.from_pretrained(PEFT_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    return_dict=True,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

tokenizer=AutoTokenizer.from_pretrained(config.base_model_name_or_path)
tokenizer.pad_token = tokenizer.eos_token

model = PeftModel.from_pretrained(model, PEFT_MODEL)

#Running finetuned model

generation_config = model.generation_config
generation_config.max_new_tokens = 400   #200
generation_config.temperature = 0.7
generation_config.top_p = 0.7
generation_config.num_return_sequences = 1
generation_config.pad_token_id = tokenizer.eos_token_id
generation_config.eos_token_id = tokenizer.eos_token_id


prompt = """
<prompt>: Run metasploit commands for my machine & summarize report.
<response>:
""".strip()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoding = tokenizer(prompt, return_tensors="pt").to(device)
#encoding = tokenizer(prompt, return_tensors="pt")
with torch.inference_mode():
    start_time = time.time()
    outputs = model.generate(
      input_ids = encoding.input_ids,
      attention_mask = encoding.attention_mask,
      generation_config = generation_config
  )
    end_time = time.time()

print(tokenizer.decode(outputs[0], skip_special_tokens=True))

execution_time = end_time - start_time
print(f"Time to generate and print prompt: {execution_time} seconds")
