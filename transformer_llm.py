from transformers import AutoTokenizer, AutoModelForCausalLM,BitsAndBytesConfig
import torch


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    llm_int8_skip_modules=["mamba"]  # Critical for Llama-3 architecture
)
def load_model():
    # Use a Llama-2 variant that's available on Hugging Face
    model_name = "NousResearch/Llama-2-7b-chat-hf"  # Requires HF access approval
    # Alternative: "NousResearch/Llama-2-7b-chat-hf" (no approval needed)

    # Load model with 4-bit quantization to save VRAM
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # config=bnb_config,
        # device_map="auto",
        # # load_in_4bit=True,
        # torch_dtype=torch.float16
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def create_text(tokenizer, model, prompt, max_length=500):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_length,
        temperature=0.7,
        do_sample=True
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Usage
model, tokenizer = load_model()
response = create_text(tokenizer, model, "Explain quantum computing in simple terms:")
print(response)