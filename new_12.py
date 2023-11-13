# Load model directly
import json
import os
import time
from pprint import pprint
#import bitsandbytes as bnb
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

tokenizer = AutoTokenizer.from_pretrained("C:/Users/rtas/vicuna_model")
model = AutoModelForCausalLM.from_pretrained("C:/Users/rtas/vicuna_model", local_files_only=True)
config = AutoConfig.from_pretrained("C:/Users/rtas/vicuna_model/config.json")

#tokenizer.save_pretrained("C:/Users/rtas/vicuna_model")
#model.save_pretrained("C:/Users/rtas/vicuna_model")

def print_trainable_parameters(model):
  """
  Prints the number of trainable parameters in the model.
  """
  trainable_params = 0
  all_param = 0
  for _, param in model.named_parameters():
    all_param += param.numel()
    if param.requires_grad:
      trainable_params += param.numel()
  print(
      f"trainable params: {trainable_params} || all params: {all_param} || trainables%: {100 * trainable_params / all_param}"
  )

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

config = LoraConfig(
    r=16,
    lora_alpha=32,
    #target_modules=["query_key_value"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)
print_trainable_parameters(model)

prompt = """
<human>: penetration testing prompt on kali linux using nmap tool in english
<assistant>:
""".strip()

generation_config = model.generation_config
generation_config.max_new_tokens = 200
generation_config.temperature = 0.7
generation_config.top_p = 0.7
generation_config.num_return_sequences = 1
generation_config.pad_token_id = tokenizer.eos_token_id
generation_config.eos_token_id = tokenizer.eos_token_id

#%%time
#device = "cuda:0"
#encoding = tokenizer(prompt, return_tensors="pt").to(device)
encoding = tokenizer(prompt, return_tensors="pt")
with torch.inference_mode():
    start_time = time.time()
    outputs = model.generate(
      input_ids = encoding.input_ids,
      attention_mask = encoding.attention_mask,
      generation_config = generation_config
  )
    end_time = time.time()

data = load_dataset("icantiemyshoe/cve-to-metasploit-module")
print(data)
print(data["train"][0])


print(tokenizer.decode(outputs[0], skip_special_tokens=True))



execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")

def generate_prompt(data_point):
  return f"""
<human>: {data_point["prompt"]}                 #instead of "User" used "prompt" from dataset
<assistant>: {data_point["response"]}          #instead of "Prompt" used "response" from dataset                  
""".strip()

def generate_and_tokenize_prompt(data_point):
  full_prompt = generate_prompt(data_point)
  tokenized_full_prompt = tokenizer(full_prompt, padding=True, truncation=True)
  return tokenized_full_prompt

data = data["train"].shuffle().map(generate_and_tokenize_prompt)

print(data)


#Training & FInetuning

training_args = transformers.TrainingArguments(
      per_device_train_batch_size=1,
      gradient_accumulation_steps=4,
      num_train_epochs=1,
      learning_rate=2e-4,
      fp16=True,
      save_total_limit=3,
      logging_steps=1,
      output_dir="experiments",
      optim="paged_adamw_8bit",
      lr_scheduler_type="cosine",
      warmup_ratio=0.05,
)

trainer = transformers.Trainer(
    model=model,
    train_dataset=data,
    args=training_args,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
model.config.use_cache = False
trainer.train()

model.save_pretrained("C:/Users/rtas/trained-model")

PEFT_MODEL = "C:/Users/rtas/trained-model"


config = PeftConfig.from_pretrained(PEFT_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    return_dict=True,
    #quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

tokenizer=AutoTokenizer.from_pretrained(config.base_model_name_or_path)
tokenizer.pad_token = tokenizer.eos_token

model = PeftModel.from_pretrained(model, PEFT_MODEL)

#Running finetuned model

generation_config = model.generation_config
generation_config.max_new_tokens = 200
generation_config.temperature = 0.7
generation_config.top_p = 0.7
generation_config.num_return_sequences = 1
generation_config.pad_token_id = tokenizer.eos_token_id
generation_config.eos_token_id = tokenizer.eos_token_id


prompt = """
<human>: penetration testing prompt on kali linux using nmap tool in english
<assistant>:
""".strip()

encoding = tokenizer(prompt, return_tensors="pt")
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