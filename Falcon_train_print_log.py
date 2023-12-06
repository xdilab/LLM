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
import pandas as pd
from transformers import Trainer, TrainingArguments

import csv
import logging
from transformers import TrainerCallback, IntervalStrategy
import matplotlib.pyplot as plt


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

MODEL_NAME = "vilsonrodrigues/falcon-7b-instruct-sharded"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    trust_remote_code=True,
    quantization_config=bnb_config
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

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

#model.gradient_checkpointing_enable()
#model = prepare_model_for_kbit_training(model)

config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["query_key_value"],
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


device = "cuda:0"

encoding = tokenizer(prompt, return_tensors="pt").to(device)
with torch.inference_mode():
  outputs = model.generate(
      input_ids = encoding.input_ids,
      attention_mask = encoding.attention_mask,
      generation_config = generation_config
  )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))

data = load_dataset("icantiemyshoe/cve-to-metasploit-module")

print(data)
print(data["train"][0])


def generate_prompt(data_point):
  return f"""
: {data_point["prompt"]}
: {data_point["response"]}
""".strip()

def generate_and_tokenize_prompt(data_point):
  full_prompt = generate_prompt(data_point)
  tokenized_full_prompt = tokenizer(full_prompt, padding=True, truncation=True)
  return tokenized_full_prompt


subset_fraction = 0.1  # Adjust the fraction as needed
num_train_examples = len(data["train"])
subset_size = int(num_train_examples * subset_fraction)

subset_train_data = data["train"].shuffle().select([i for i in range(subset_size)])
subset_data = subset_train_data.map(generate_and_tokenize_prompt)
#data = data["train"].shuffle().map(generate_and_tokenize_prompt)

print(subset_data)

training_args = transformers.TrainingArguments(
      per_device_train_batch_size=1,
      gradient_accumulation_steps=4,
      num_train_epochs=2,
      learning_rate=0.1,
      fp16=True,
      save_total_limit=3,
      logging_steps=1,
      output_dir="experiments",
      optim="paged_adamw_8bit",
      lr_scheduler_type="cosine",
      warmup_ratio=0.05,
    max_steps=1,
)

train_test_split_list = []
other_loss_values_list = []
prompt_list = []
response_list = []
loss_values = []
steps = []

class MyTrainerCallback(transformers.TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        # Extract relevant information
        current_step = state.global_step
        current_loss = logs.get("loss")
        current_train_test_split = train_test_split_list[-1] if train_test_split_list else None
        current_other_loss_values = other_loss_values_list[-1] if other_loss_values_list else None
        current_prompt = prompt_list[-1] if prompt_list else None
        current_response = response_list[-1] if response_list else None

        # Append information to lists
        steps.append(current_step)
        loss_values.append(current_loss)
        train_test_split_list.append(current_train_test_split)
        other_loss_values_list.append(current_other_loss_values)
        prompt_list.append(current_prompt)
        response_list.append(current_response)

trainer = transformers.Trainer(
    model=model,
    train_dataset=subset_data,
    args=training_args,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    callbacks=[MyTrainerCallback()],

)
model.config.use_cache = False
trainer.train()

json_file_path = "C:/Rt/RA/pr/experiments/checkpoint-500/trainer_state.json"  # Replace with your actual file path
with open(json_file_path, "r") as json_file:
    data = json.load(json_file)

# Extract loss values and steps from log_history
log_history = data["log_history"]
loss_values = [entry["loss"] for entry in log_history]
steps = [entry["step"] for entry in log_history]

# Plot the loss values
plt.plot(steps, loss_values, label="Training Loss")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.legend()
plt.show()

min_length = min(len(steps), len(loss_values), len(train_test_split_list), len(other_loss_values_list), len(prompt_list), len(response_list))

# Trim lists to the minimum length
steps = steps[:min_length]
loss_values = loss_values[:min_length]
train_test_split_list = train_test_split_list[:min_length]
other_loss_values_list = other_loss_values_list[:min_length]
prompt_list = prompt_list[:min_length]
response_list = response_list[:min_length]

# Create a DataFrame with the collected information
output_data = {
    "Step": steps,
    "Loss": loss_values,
    "Train_Test_Split": train_test_split_list,
    "Other_Loss_Values": other_loss_values_list,
    "Prompt": prompt_list,
    "Response": response_list,
}

output_df = pd.DataFrame(output_data)

output_csv_path = "experiments/training_information.csv"
output_df.to_csv(output_csv_path, index=False)

model.save_pretrained("trained-model/new_fal3")

