#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from huggingface_hub import HfApi, create_repo, upload_folder
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

def parse_args():
    parser = argparse.ArgumentParser(description="Upload PEFT model to Hugging Face Hub")
    parser.add_argument(
        "--checkpoint_path", 
        type=str, 
        required=True,
        help="Path to the PEFT checkpoint directory (e.g., results/2025-03-31_14-22-50/peft_5)"
    )
    parser.add_argument(
        "--model_name", 
        type=str, 
        required=True,
        help="Name of the base model (e.g., mistralai/Mistral-7B-v0.3)"
    )
    parser.add_argument(
        "--repo_name", 
        type=str, 
        required=True,
        help="Name for the Hugging Face repository (e.g., username/model-name)"
    )
    parser.add_argument(
        "--private", 
        action="store_true",
        help="Make the repository private"
    )
    parser.add_argument(
        "--description", 
        type=str, 
        default="",
        help="Description for the Hugging Face repository"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Loading PEFT configuration from {args.checkpoint_path}")
    peft_config = PeftConfig.from_pretrained(args.checkpoint_path)
    
    print(f"Creating repository: {args.repo_name}")
    api = HfApi()
    
    # Create or get repository
    repo_url = create_repo(
        repo_id=args.repo_name,
        private=args.private,
        exist_ok=True,
        repo_type="model"
    )
    
    print(f"Repository URL: {repo_url}")
    
    # Create model card content
    model_card = f"""---
base_model: {args.model_name}
tags:
- peft
- lora
- federated-learning
- flower
---

# FlowerTune LoRA Model

This is a LoRA adapter for {args.model_name} fine-tuned with Flower federated learning framework on a general NLP dataset.

## Training Details

- Dataset: vicgalle/alpaca-gpt4
- Training method: Federated LoRA fine-tuning with FlowerTune
- Framework: Flower

{args.description}
"""
    
    # Write model card to a temporary file
    with open("README.md", "w") as f:
        f.write(model_card)
    
    # Upload the model to Hugging Face Hub
    print(f"Uploading model to {args.repo_name}...")
    upload_folder(
        folder_path=args.checkpoint_path,
        repo_id=args.repo_name,
        repo_type="model",
        commit_message="Upload LoRA adapter"
    )
    
    # Upload the README.md file
    api.upload_file(
        path_or_fileobj="README.md",
        path_in_repo="README.md",
        repo_id=args.repo_name,
        repo_type="model",
        commit_message="Add model card"
    )
    
    print(f"Successfully uploaded PEFT model to {args.repo_name}")
    print(f"Model available at: https://huggingface.co/{args.repo_name}")
    
    # Clean up
    os.remove("README.md")

if __name__ == "__main__":
    main() 