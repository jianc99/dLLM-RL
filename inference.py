from transformers import AutoModelForCausalLM, AutoTokenizer
from generate import block_diffusion_generate, block_diffusion_generate_ar_verify
import torch

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True

model_name_arbd = "/data02/home/zhijian/jian/modified/dLLM-RL/sft_sdar_arbd_16/ckpt/optimized"
model_name_bd = "/data02/home/zhijian/jian/models/JetLM/SDAR-4B-Chat-b16"

# model_name_arbd = "/data02/home/zhijian/jian/modified/dLLM-RL/sft_sdar_arbd_8/ckpt/optimized"
# model_name_bd = "/data02/home/zhijian/jian/models/JetLM/SDAR-8B-Chat-b8"

# model_name_arbd = "/data02/home/zhijian/jian/modified/dLLM-RL/sft_sdar/ckpt/optimized"
# model_name_bd = "/data02/home/zhijian/jian/models/JetLM/SDAR-8B-Chat-b16"

# model_name_arbd = "/data02/home/zhijian/jian/modified/dLLM-RL/sft_sdar_arbd_64/ckpt/optimized"
# model_name_bd = "/data02/home/zhijian/jian/models/JetLM/SDAR-8B-Chat-b64"


# model_name_arbd = "/data02/home/zhijian/jian/modified/dLLM-RL/sft_trado_arbd/ckpt/optimized"
# model_name_bd = "/data02/home/zhijian/jian/models/TraDo-8B-Instruct"

model_arbd = AutoModelForCausalLM.from_pretrained(
    model_name_arbd, trust_remote_code=True, torch_dtype="bfloat16", device_map="cuda"
)
model_bd = AutoModelForCausalLM.from_pretrained(
    model_name_bd, trust_remote_code=True, torch_dtype="bfloat16", device_map="cuda"
)

tokenizer = AutoTokenizer.from_pretrained(model_name_bd, trust_remote_code=True)

# prompt = "<|im_start|>user\nConvert the point $(0,3)$ in rectangular coordinates to polar coordinates.  Enter your answer in the form $(r,\\theta),$ where $r > 0$ and $0 \\le \\theta < 2 \\pi.$\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>assistant\n"
prompt = "<|im_start|>user\nThe product of three consecutive integers is 120. That product divided by the mean of the three integers is 24. What is the largest of the three consecutive integers?\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>assistant\n"
# messages = [{"role": "user", "content": prompt}]
# text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

tokens = tokenizer.batch_encode_plus([prompt], return_tensors='pt', padding=True, truncation=True, max_length=200)
tokens = {k: v.to(model_arbd.device) for k, v in tokens.items()}

output_ids_bd = block_diffusion_generate(
    model_bd,
    prompt=tokens,
    mask_id=151669,
    gen_length=1024,
    block_length=16, denoising_steps=16,
    temperature=1.0, top_k=0, top_p=1.0,
    remasking_strategy="low_confidence_dynamic",
    confidence_threshold=0.9,
    stopping_criteria_idx = [151645]
)

output_ids_arbd = block_diffusion_generate_ar_verify(
    model_arbd,
    prompt=tokens,
    mask_id=151669,
    gen_length=1024,
    block_length=16, denoising_steps=1,
    temperature=1.0, top_k=1, top_p=1.0,
    confidence_threshold=0.9,
    stopping_criteria_idx = [151645]
)

output_text = tokenizer.decode(output_ids_arbd[0], skip_special_tokens=False)
cleaned_text = output_text.replace('<|MASK|>', '').replace('<|endoftext|>', '')
print("=== AR-SDAR Output ===")
print(cleaned_text)

output_text_bd = tokenizer.decode(output_ids_bd[0], skip_special_tokens=False)
cleaned_text_bd = output_text_bd.replace('<|MASK|>', '').replace('<|endoftext|>', '')
print("=== Baseline SDAR Output ===")
print(cleaned_text_bd)

