# Create models directory in the project root
$modelsDir = "models"
New-Item -ItemType Directory -Force -Path $modelsDir | Out-Null

Write-Host "Models will be downloaded to: $modelsDir"

# List of models to download
$models = @(
    @{
        Url = "https://huggingface.co/TheBloke/finance-LLM-GGUF/resolve/main/finance-llm.Q4_K_M.gguf"
        Filename = "finance-llm.Q4_K_M.gguf"
        Description = "Finance LLM (4-bit quantized, 4.08 GB)"
    },
    @{
        Url = "https://huggingface.co/QuantFactory/finance-Llama3-8B-GGUF/resolve/main/finance-llama3-8b.Q4_K_M.gguf"
        Filename = "finance-llama3-8b.Q4_K_M.gguf"
        Description = "Finance Llama3 8B (4-bit quantized, 4.92 GB)"
    },
    @{
        Url = "https://huggingface.co/andrijdavid/finance-chat-GGUF/resolve/main/finance-chat.Q4_K_M.gguf"
        Filename = "finance-chat.Q4_K_M.gguf"
        Description = "Finance Chat (4-bit quantized, 4.08 GB)"
    }
)

# Download each model
for ($i = 0; $i -lt $models.Count; $i++) {
    $model = $models[$i]
    Write-Host "`n[$(($i+1))/$($models.Count)] Downloading $($model.Description)..."
    Write-Host "From: $($model.Url)"
    Write-Host "To: $modelsDir\$($model.Filename)"
    
    try {
        $ProgressPreference = 'SilentlyContinue'  # Hide progress bar for faster downloads
        Invoke-WebRequest -Uri $model.Url -OutFile "$modelsDir\$($model.Filename)"
        Write-Host "Successfully downloaded: $($model.Filename)"
    } catch {
        Write-Host "Error downloading model: $_"
        continue
    }
}

Write-Host "`nDownload complete! The following models are now available:"
Get-ChildItem -Path $modelsDir | ForEach-Object {
    $fileSizeGB = [math]::Round(($_.Length / 1GB), 2)
    Write-Host "- $($_.Name) ($fileSizeGB GB)"
}
