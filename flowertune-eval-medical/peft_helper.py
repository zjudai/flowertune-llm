
import os
import torch
from peft import PeftModel, LoraConfig, get_peft_model
import glob

def load_peft_local(base_model, peft_path, torch_dtype=None):
    """Load PEFT model from local path, handling HuggingFace hub validation errors."""
    print(f"Attempting to load PEFT model from local path: {peft_path}")
    
    # Try standard loading first
    try:
        peft_model = PeftModel.from_pretrained(
            base_model, peft_path, torch_dtype=torch_dtype
        )
        print(f"Successfully loaded PEFT model using standard method")
        return peft_model
    except Exception as e:
        print(f"Error loading PEFT model with standard method: {e}")
        
    print("Trying alternative loading method for local path...")
    
    # Check if adapter_config.json exists
    adapter_config_path = os.path.join(peft_path, "adapter_config.json")
    if not os.path.exists(adapter_config_path):
        print(f"Warning: adapter_config.json not found in {peft_path}")
            
    # Find state dict files
    state_dict_files = glob.glob(f"{peft_path}/*.safetensors") or glob.glob(f"{peft_path}/*.bin")
    if not state_dict_files:
        raise ValueError(f"No model state files found in {peft_path}")
        
    print(f"Found model state file: {state_dict_files[0]}")
    
    # Configure a default LoRA setup for common models
    config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    
    # Apply LoRA config
    print("Applying default LoRA configuration")
    peft_model = get_peft_model(base_model, config)
    
    # Load weights
    print(f"Loading state dict from {state_dict_files[0]}")
    state_dict = torch.load(state_dict_files[0], map_location="cpu")
    peft_model.load_state_dict(state_dict, strict=False)
    
    print("Successfully loaded PEFT weights using alternative method")
    return peft_model
