from huggingface_hub import hf_hub_download
import os
import os
os.environ['HF_TOKEN'] = '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'
def download_model():
    repo_id = "chiizaraa/Llama-3.2-1B-Instruct-Q4_K_M-GGUF"
    filename = "llama-3.2-1b-instruct-q4_k_m.gguf"
    
    
        # Download the model
    model_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        resume_download=True,
        token=True  # Requires Hugging Face login
    )
    
    print(f"Model downloaded to: {model_path}")
    return model_path
        


# Run download
model_path = download_model()

if model_path and os.path.exists(model_path):
    print("Download verification successful!")
    print(f"File size: {os.path.getsize(model_path)/1e6:.1f} MB")
else:
    print("Download failed. Check the error messages above.")