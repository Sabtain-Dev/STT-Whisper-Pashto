import os
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# Default fallback pointing directly to your public Hugging Face model repository
DEFAULT_REPO = "Sabtain-Dev/STT-Whisper-Pashto"

def load_inference_components(model_id_or_path=None, hf_token=None):
    """
    Initializes and caches the frozen evaluation model and associated audio processors.
    Accepts an optional hf_token if needed in the future (though optional for public repos).
    
    Args:
        model_id_or_path (str): HF repo string or local directory path containing model weights.
        hf_token (str, optional): Hugging Face user access token string.
        
    Returns:
        tuple: (model, processor) ready for high-performance transcription.
    """
    model_path = model_id_or_path or os.environ.get("MERGED_MODEL_PATH", DEFAULT_REPO)
    active_token = hf_token or os.environ.get("HF_TOKEN", "")
    
    # Determine execution hardware architecture
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"[*] Initializing fully merged Pashto model architecture from: {model_path}")
    
    # Load processor assets (Tokenizer + Feature Extractor)
    processor = WhisperProcessor.from_pretrained(
        model_path, 
        token=active_token if active_token else None
    )
    
    # Load foundational model matrix
    model = WhisperForConditionalGeneration.from_pretrained(
        model_path, 
        token=active_token if active_token else None,
        low_cpu_mem_usage=True
    ).to(device)
    
    # Strip gradient layers and lock down to inference evaluation parameters
    model.eval()
    
    print(f"[+] Pashto transcription pipeline successfully mounted onto device target: {device.upper()}")
    return model, processor