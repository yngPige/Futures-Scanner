import os
import sys
from huggingface_hub import hf_hub_download

# Create models directory
models_dir = os.path.join(os.path.expanduser("~"), ".cache", "futures_scanner", "models")
os.makedirs(models_dir, exist_ok=True)

print(f"Models will be downloaded to: {models_dir}")

# List of models to download
models = [
    {
        "repo_id": "TheBloke/finance-LLM-GGUF",
        "filename": "finance-llm.Q4_K_M.gguf",
        "description": "Finance LLM (4-bit quantized, 4.08 GB)"
    },
    {
        "repo_id": "QuantFactory/finance-Llama3-8B-GGUF",
        "filename": "finance-llama3-8b.Q4_K_M.gguf",
        "description": "Finance Llama3 8B (4-bit quantized, 4.92 GB)"
    },
    {
        "repo_id": "andrijdavid/finance-chat-GGUF",
        "filename": "finance-chat.Q4_K_M.gguf",
        "description": "Finance Chat (4-bit quantized, 4.08 GB)"
    }
]

# Download each model
for i, model in enumerate(models, 1):
    print(f"\n[{i}/{len(models)}] Downloading {model['description']}...")
    print(f"From: {model['repo_id']}")
    print(f"File: {model['filename']}")
    
    try:
        file_path = hf_hub_download(
            repo_id=model['repo_id'],
            filename=model['filename'],
            local_dir=models_dir,
            local_dir_use_symlinks=False
        )
        print(f"Successfully downloaded to: {file_path}")
    except Exception as e:
        print(f"Error downloading model: {e}")
        continue

print("\nDownload complete! The following models are now available:")
for file in os.listdir(models_dir):
    file_path = os.path.join(models_dir, file)
    file_size_gb = os.path.getsize(file_path) / (1024 * 1024 * 1024)
    print(f"- {file} ({file_size_gb:.2f} GB)")
