import argparse

import torch
import os
# Added PEFT helper import
try:
    from peft_helper import load_peft_local
    print('PEFT helper loaded')
except ImportError:
    print('PEFT helper not found, using standard loading')

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from benchmarks import infer_fiqa, infer_fpb, infer_tfns

# Fixed seed
torch.manual_seed(2024)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--base-model-name-path", type=str, default="mistralai/Mistral-7B-v0.3"
)
parser.add_argument("--run-name", type=str, default="fl")
parser.add_argument("--peft-path", type=str, default=None)
parser.add_argument("--datasets", type=str, default="fpb")
parser.add_argument("--batch
parser.add_argument("--use-peft-helper", action="store_true", help="Use PEFT helper for local models")-size", type=int, default=32)
parser.add_argument("--quantization", type=int, default=4)
parser.add_argument("--trust-remote-code", action="store_true", help="Trust remote code when loading models")
args = parser.parse_args()


# Load model and tokenizer
if args.quantization == 4:
    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    torch_dtype = torch.float32
elif args.quantization == 8:
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    torch_dtype = torch.float16
else:
    raise ValueError(
        f"Use 4-bit or 8-bit quantization. You passed: {args.quantization}/"
    )

model = AutoModelForCausalLM.from_pretrained(
    args.base_model_name_path,
    quantization_config=quantization_config,
    torch_dtype=torch_dtype,
    trust_remote_code=args.trust_remote_code,
)
if args.peft_path is not None:
    try:
        if hasattr(args, 'use_peft_helper') and args.use_peft_helper:
            print(f"Using PEFT helper to load from local path: {args.peft_path}")
            # Use helper if available
            if 'load_peft_local' in globals():
                model = load_peft_local(model, args.peft_path, torch_dtype=torch_dtype).to("cuda")
            else:
                print("PEFT helper not available, falling back to standard loading")
                model = PeftModel.from_pretrained(model, args.peft_path, torch_dtype=torch_dtype).to("cuda")
        else:
            # Use standard loading
            model = PeftModel.from_pretrained(model, args.peft_path, torch_dtype=torch_dtype).to("cuda")
    except Exception as e:
        print(f"Error loading PEFT model: {e}")
        print("Attempting fallback loading method...")
        try:
            import glob
            if os.path.exists(args.peft_path):
                state_dict_files = glob.glob(f"{args.peft_path}/*.safetensors") or glob.glob(f"{args.peft_path}/*.bin")
                if state_dict_files:
                    print(f"Found state dict file: {state_dict_files[0]}")
                    from peft import LoraConfig, get_peft_model
                    config = LoraConfig(
                        r=32,
                        lora_alpha=64,
                        lora_dropout=0.1,
                        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                        task_type="CAUSAL_LM",
                    )
                    peft_model = get_peft_model(model, config)
                    state_dict = torch.load(state_dict_files[0], map_location="cpu")
                    peft_model.load_state_dict(state_dict, strict=False)
                    model = peft_model.to("cuda")
                    print("Successfully loaded PEFT weights with fallback method")
                else:
                    print(f"No state dict files found in {args.peft_path}")
                    raise
            else:
                print(f"PEFT path not found: {args.peft_path}")
                raise
        except Exception as fallback_error:
            print(f"Fallback loading also failed: {fallback_error}")
            print("Continuing with base model only").to("cuda")

tokenizer = AutoTokenizer.from_pretrained(
    args.base_model_name_path,
    trust_remote_code=args.trust_remote_code,
)

if not tokenizer.pad_token or tokenizer.pad_token_id == tokenizer.eos_token_id:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model.resize_token_embeddings(len(tokenizer))


# Evaluate
model = model.eval()
with torch.no_grad():
    for dataset in args.datasets.split(","):
        if dataset == "fpb":
            infer_fpb(model, tokenizer, args.batch_size, args.run_name)
        elif dataset == "fiqa":
            infer_fiqa(model, tokenizer, args.batch_size, args.run_name)
        elif dataset == "tfns":
            infer_tfns(model, tokenizer, args.batch_size, args.run_name)
        else:
            raise ValueError("Undefined Dataset.")
