#!/usr/bin/env python3
import os
import subprocess
import tempfile
import argparse
from pathlib import Path

def create_readme_with_metadata(source_readme, output_file):
    """
    Add metadata to the README file
    """
    metadata = """---
tags:
- federated-learning
- flower
- lora
- peft
datasets:
- vicgalle/alpaca-gpt4
---

"""
    # Read the source README content
    with open(source_readme, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Write the metadata and content to the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(metadata + content)
    
    print(f"Created README with metadata at {output_file}")

def upload_to_huggingface(readme_path, repo_id):
    """
    Upload the README file to a Hugging Face repository
    Using existing authentication
    """
    try:
        # Install huggingface_hub if not already installed
        subprocess.check_call(["pip", "install", "huggingface_hub"])
        
        # Import the necessary functions from huggingface_hub
        from huggingface_hub import HfApi
        
        # Initialize the Hugging Face API (will use existing token)
        api = HfApi()
        
        # Upload the README to the repository
        api.upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=repo_id,
            commit_message="Update README with metadata"
        )
        
        print(f"Successfully uploaded README to {repo_id}")
        return True
    except Exception as e:
        print(f"Error uploading to Hugging Face: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Upload README with metadata to Hugging Face")
    parser.add_argument("--readme", default="README.md", help="Path to the source README file")
    parser.add_argument("--repo-id", default="zjudai/FlowerTune", help="Hugging Face repository ID")
    args = parser.parse_args()
    
    readme_path = Path(args.readme)
    if not readme_path.exists():
        print(f"Error: README file not found at {readme_path}")
        return
    
    # Create a temporary file for the README with metadata
    with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as tmp:
        temp_readme = tmp.name
    
    try:
        # Add metadata to the README
        create_readme_with_metadata(readme_path, temp_readme)
        
        # Upload to Hugging Face
        success = upload_to_huggingface(temp_readme, args.repo_id)
        
        if success:
            print(f"README uploaded successfully to {args.repo_id}")
        else:
            print("Failed to upload README")
    
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_readme):
            os.unlink(temp_readme)

if __name__ == "__main__":
    main() 