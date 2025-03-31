import argparse
import os

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoConfig,
)

from benchmarks import MMLU_CATEGORY, infer_mmlu

# Fixed seed
torch.manual_seed(2024)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--base-model-name-path", type=str, default="mistralai/Mistral-7B-v0.3"
)
parser.add_argument("--run-name", type=str, default="fl")
parser.add_argument("--peft-path", type=str, default=None)
parser.add_argument(
    "--datasets",
    type=str,
    default="mmlu",
    help="The dataset to infer on",
)
parser.add_argument(
    "--category",
    type=str,
    default=None,
    help="The category for MMLU dataset, chosen from [stem, social_sciences, humanities, other]",
)
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--quantization", type=int, default=4)
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--trust-remote-code", action="store_true", help="Trust remote code when loading models")
args = parser.parse_args()

def get_model_config(model_name):
    """Get model configuration with proper error handling"""
    try:
        config = AutoConfig.from_pretrained(
            model_name,
            trust_remote_code=args.trust_remote_code,
            use_cache=True,
        )
        return config
    except Exception as e:
        print(f"Warning: Could not load config for {model_name}: {str(e)}")
        return None

def load_model_and_tokenizer(model_name, peft_path=None):
    """Load model and tokenizer with proper error handling"""
    # Get model config first
    config = get_model_config(model_name)
    
    # Setup quantization config
    if args.quantization == 4:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        torch_dtype = torch.float16
    elif args.quantization == 8:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )
        torch_dtype = torch.float16
    else:
        raise ValueError(f"Use 4-bit or 8-bit quantization. You passed: {args.quantization}/")

    print(f"Loading model: {model_name}")
    print(f"Trust remote code: {args.trust_remote_code}")
    print(f"Quantization: {args.quantization}-bit")
    print(f"PEFT path: {peft_path}")

    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=args.trust_remote_code,
            use_fast=False,
        )
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"Warning: Could not load tokenizer for {model_name}: {str(e)}")
        tokenizer = None

    # Load model
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            quantization_config=quantization_config,
            torch_dtype=torch_dtype,
            trust_remote_code=args.trust_remote_code,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        
        # Load PEFT weights if provided
        if peft_path and os.path.exists(peft_path):
            try:
                print(f"Loading PEFT weights from: {peft_path}")
                model = PeftModel.from_pretrained(
                    model,
                    peft_path,
                    torch_dtype=torch_dtype,
                    device_map="auto",
                    is_trainable=False,
                )
                print("PEFT weights loaded successfully")
            except Exception as e:
                print(f"Warning: Could not load PEFT weights from {peft_path}: {str(e)}")
        
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model {model_name}: {str(e)}")
        return None, None

# Load model and tokenizer
model, tokenizer = load_model_and_tokenizer(args.base_model_name_path, args.peft_path)

if model is None or tokenizer is None:
    raise RuntimeError("Failed to load model or tokenizer")

# Move model to specified device
model = model.to(args.device)

# Evaluate
for dataset in args.datasets.split(","):
    if dataset == "mmlu":
        for cate in args.category.split(","):
            if cate not in MMLU_CATEGORY.keys():
                raise ValueError("Undefined Category.")
            else:
                infer_mmlu(model, tokenizer, args.batch_size, cate, args.run_name)
    else:
        raise ValueError("Undefined Dataset.")
