from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, logging, TrainingArguments
from transformers import AutoConfig, AutoModel
import torch
from peft import PeftModel
import huggingface_hub
huggingface_hub.login()

base_model='MLP-KTLim/llama-3-Korean-Bllossom-8B'
new_model='새로운 모델 저장 위치'

base_Model=AutoModelForCausalLM.from_pretrained(
    base_model,
    low_cpu_mem_usage=True,
    device_map='auto',
    return_dict=True,
    torch_dtype=torch.float16
)
tokenizer=AutoTokenizer.from_pretrained(base_model,trust_remote_code=True)
tokenizer.pad_token=tokenizer.eos_token
tokenizer.padding_side='right'

model=PeftModel.from_pretrained(base_Model,new_model)

def generate_response(prompt,model):
    encoded_input=tokenizer(prompt,return_tensors='pt',add_special_tokens=True)
    model_inputs=encoded_input.to('cuda')
    generated_ids=model.generate(
        **model_inputs,
        max_new_tokens=256,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    decoded_output=tokenizer.batch_decode(generated_ids)
    return decoded_output[0].replace(prompt,'')
key = "KakaoVX에 대해 설명해주세요."
prompt = f"""You are a helpful AI assistant. Please answer the user's questions kindly.

### Instruction:
{key}

### Response:"""
generate_response(prompt, model)

merged_model = model.merge_and_unload()
merged_model.save_pretrained("파인튜닝한 모델 저장 위치")
tokenizer.save_pretrained("토크나이저 저장 위치")
hfaddr='허깅페이스 저장 위치'
merged_model.push_to_hub(hfaddr)
tokenizer.push_to_hub(hfaddr)


