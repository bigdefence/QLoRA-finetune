import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, pipeline
from transformers import BitsAndBytesConfig
from trl import SFTTrainer
from peft import LoraConfig

import huggingface_hub
huggingface_hub.login()

base_model='MLP-KTLim/llama-3-Korean-Bllossom-8B'
new_model='llama-3-blossom-kakao-8B'
dataset_name='bigdefence/custom'
dataset=load_dataset(dataset_name,split='train')

def create_text_column(example):
    text = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
    example["text"] = text
    return example
dataset=dataset.map(create_text_column)

if torch.cuda.get_device_capability()[0] >= 8:
    !pip install -qqq flash-attn
    attn_implementation = "flash_attention_2"
    torch_dtype = torch.bfloat16
else:
    attn_implementation = "eager"
    torch_dtype = torch.float16

quant_config= BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=False,
    bnb_4bit_compute_dtype=torch_dtype,
 )

model= AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=quant_config,
    device_map="auto",
)
model.config.use_cache=False
model.config.pretraining_tp=1

tokenizer= AutoTokenizer.from_pretrained(base_model,trust_remote_code=True)
tokenizer.pad_token=tokenizer.eos_token
EOS_TOKEN=tokenizer.eos_token
def prompt_eos(sample):
    sample['text']=sample['text']+EOS_TOKEN
    return sample
dataset=dataset.map(prompt_eos)

dataset[0]

peft_params = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)
training_params = TrainingArguments(
    output_dir="./results",
    num_train_epochs=100,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=25,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to=None
)

trainer=SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_params,
    dataset_text_field="text",
    max_seq_length=256,
    tokenizer=tokenizer,
    args=training_params,
    packing=False
)
trainer.train()

def generate_response(prompt,model):
    encoded_input = tokenizer(prompt,  return_tensors="pt", add_special_tokens=True)
    model_inputs=encoded_input.to('cuda')
    generated_ids = model.generate(**model_inputs, max_new_tokens=256,do_sample=True,pad_token_id=tokenizer.eos_token_id)
    decoded_output=tokenizer.batch_decode(generated_ids)
    return decoded_output[0].replace(prompt,'')
key="카카오vx에 대해 설명해줘"
prompt=f"""you are a assistant please answer in korean lanauage

### Instruction:
{key}

### Response:"""
generate_response(prompt,model)

save_path='저장할 위치'
trainer.save_model(save_path)

