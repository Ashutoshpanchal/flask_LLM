from llama_cpp import Llama
from typing import Generator

def load_model():
    # Download GGUF model first (see notes below)
    model_path = "/Users/ashutoshpanchal/.lmstudio/models/Qwen/Qwen2-0.5B-Instruct-GGUF/qwen2-0_5b-instruct-q4_0.gguf" 
    
    # GPU offloading layers (adjust based on your VRAM)
    n_gpu_layers = -1  # -1 = all layers on GPU
    
    return Llama(
        model_path=model_path,
        n_ctx=4096,  # Context window size
        n_gpu_layers=n_gpu_layers,
        n_threads=8,  # CPU threads
        verbose=False
    )

def create_text(llm: Llama, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    
    response = llm.create_chat_completion(
        messages=messages,
        temperature=0.7,
        max_tokens=500,
        stop=["<|eot_id|>"]  # Llama-3 specific stop token
    )
    
    return response['choices'][0]['message']['content']

def create_text_stream(llm: Llama, prompt: str) -> Generator[str, None, None]:
    messages = [{"role": "user", "content": prompt}]
    
    stream = llm.create_chat_completion(
        messages=messages,
        temperature=0.7,
        max_tokens=500,
        stream=True,
        stop=["<|eot_id|>"]
    )
    
    for chunk in stream:
        if 'content' in chunk['choices'][0]['delta']:
            yield chunk['choices'][0]['delta']['content']

# Usage
if __name__ == "__main__":
    llm = load_model()
    
    # Single generation
    print(create_text(llm, "Explain quantum computing in simple terms"))
    
    # Streaming
    for chunk in create_text_stream(llm, "Write a poem about AI:"):
        print(chunk, end="", flush=True)