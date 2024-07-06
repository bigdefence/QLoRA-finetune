import torch
from transformers import AutoTokenizer, AutoModelForCausalLM,pipeline
base_model = '허깅페이스 모델 저장 위치'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float16, device_map="auto")
model.eval()  

def generate_response(prompt, model, tokenizer, max_new_tokens=256):
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    inputs = inputs.to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.replace(prompt, '').strip()

key = "카카오vx에 대해 설명해줘"
prompt = f"""당신은 한국어로 대답하는 어시스턴트입니다.

### 질문:
{key}

### 답변:"""

response = generate_response(prompt, model, tokenizer)
print(response)


