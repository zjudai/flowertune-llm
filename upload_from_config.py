#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import tomli
import subprocess
import sys
from huggingface_hub import HfApi, create_repo, upload_folder
from peft import PeftConfig

def parse_args():
    parser = argparse.ArgumentParser(description="Upload PEFT models to Hugging Face Hub from config file")
    parser.add_argument(
        "--config_file", 
        type=str, 
        default="eval_config.toml",
        help="Path to the TOML config file with model paths"
    )
    parser.add_argument(
        "--username", 
        type=str, 
        required=True,
        help="Your Hugging Face username"
    )
    parser.add_argument(
        "--private", 
        action="store_true",
        help="Make repositories private"
    )
    return parser.parse_args()

def read_config(config_file):
    """Read the TOML configuration file"""
    print(f"读取配置文件: {config_file}")
    with open(config_file, "rb") as f:
        return tomli.load(f)

def get_readme_models():
    """Get the list of models mentioned in README.md's table"""
    models = [
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "meta-llama/Llama-3.1-8B-Instruct",
        "mistralai/Mistral-7B-v0.3",
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "meta-llama/Llama-3.2-1B-Instruct",
        "google/gemma-3-1b-it",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    ]
    print(f"找到README中的{len(models)}个模型")
    return models

def upload_model(model_name, peft_path, username, private=False):
    """Upload a specific model to Hugging Face Hub"""
    print(f"\n开始处理模型: {model_name}")
    # 确认检查点路径存在
    if not os.path.exists(peft_path):
        print(f"错误: 检查点路径不存在: {peft_path}")
        return False
    
    # 提取模型简称作为仓库名
    model_short = model_name.split("/")[-1]
    model_short_lower = model_short.lower()
    repo_name = f"{username}/flowertune-general-nlp-lora-{model_short_lower}"
    
    print(f"正在上传模型: {model_name}")
    print(f"检查点路径: {peft_path}")
    print(f"仓库名称: {repo_name}")
    
    # 检查适配器文件是否存在
    adapter_model_path = os.path.join(peft_path, "adapter_model.safetensors")
    adapter_config_path = os.path.join(peft_path, "adapter_config.json")
    
    if not os.path.exists(adapter_model_path):
        print(f"错误: 适配器模型文件不存在: {adapter_model_path}")
        return False
    
    if not os.path.exists(adapter_config_path):
        print(f"错误: 适配器配置文件不存在: {adapter_config_path}")
        return False
    
    print(f"找到适配器文件: adapter_model.safetensors 和 adapter_config.json")
    
    # 检查PEFT配置
    try:
        peft_config = PeftConfig.from_pretrained(peft_path)
        print(f"PEFT配置加载成功")
    except Exception as e:
        print(f"警告: 无法加载PEFT配置: {str(e)}")
        return False
    
    # 创建模型卡内容
    description = f"This model is a LoRA adapter fine-tuned on {model_name} using the Flower federated learning framework. It was trained on a general NLP dataset (vicgalle/alpaca-gpt4) through distributed learning to improve performance."
    
    model_card = f"""---
base_model: {model_name}
tags:
- peft
- lora
- federated-learning
- flower
datasets:
- vicgalle/alpaca-gpt4
---

# FlowerTune LoRA Model

This is a LoRA adapter for {model_name} fine-tuned with Flower federated learning framework on a general NLP dataset.

## Training Details

- Dataset: vicgalle/alpaca-gpt4
- Training method: Federated LoRA fine-tuning with FlowerTune
- Framework: Flower

{description}

## Links
- FlowerTune Homepage: [https://huggingface.co/zjudai/FlowerTune](https://huggingface.co/zjudai/FlowerTune)
- FlowerTune Collection: [https://huggingface.co/collections/zjudai/flowertune-lora-collection-67ecd5d0dae6145cbf798439](https://huggingface.co/collections/zjudai/flowertune-lora-collection-67ecd5d0dae6145cbf798439)
"""
    
    # 写入模型卡到临时文件
    readme_path = os.path.join(peft_path, "README.md")
    try:
        with open(readme_path, "w") as f:
            f.write(model_card)
        print(f"已创建模型卡README.md")
    except Exception as e:
        print(f"创建模型卡失败: {str(e)}")
        return False
    
    # 创建或获取仓库
    api = HfApi()
    try:
        repo_url = create_repo(
            repo_id=repo_name,
            private=private,
            exist_ok=True,
            repo_type="model"
        )
        print(f"仓库URL: {repo_url}")
    except Exception as e:
        print(f"创建仓库失败: {str(e)}")
        if os.path.exists(readme_path):
            try:
                os.remove(readme_path)
            except:
                pass
        return False
    
    # 上传模型到Hugging Face Hub
    try:
        print("上传模型中...")
        upload_folder(
            folder_path=peft_path,
            repo_id=repo_name,
            repo_type="model",
            commit_message="Upload LoRA adapter"
        )
        print(f"模型 {model_name} 上传成功!")
        print(f"模型可在以下网址访问: https://huggingface.co/{repo_name}")
        return True
    except Exception as e:
        print(f"上传模型失败: {str(e)}")
        return False
    finally:
        # 清理README (如果不是原本就存在的)
        if os.path.exists(readme_path) and not os.path.exists(os.path.join(peft_path, "README.md.bak")):
            try:
                os.remove(readme_path)
            except:
                pass

def main():
    print("开始执行上传脚本")
    args = parse_args()
    print(f"用户名: {args.username}")
    print(f"私有仓库: {args.private}")
    
    # 确保安装了必要的依赖
    try:
        subprocess.run(["pip", "install", "-q", "huggingface_hub", "transformers", "peft", "tomli"], check=True)
        print("依赖检查完成")
    except subprocess.CalledProcessError:
        print("安装依赖失败，请手动安装: pip install huggingface_hub transformers peft tomli")
        return
    
    # 检查是否已登录Hugging Face
    try:
        api = HfApi()
        whoami = api.whoami()
        print(f"已登录Hugging Face，用户: {whoami['name']}")
    except Exception as e:
        print(f"未登录Hugging Face或访问出错: {str(e)}")
        print("请先运行: huggingface-cli login")
        return
    
    # 读取配置文件
    try:
        config_file = args.config_file
        print(f"使用配置文件: {config_file}")
        if not os.path.exists(config_file):
            print(f"错误: 配置文件不存在: {config_file}")
            return
        config = read_config(config_file)
    except Exception as e:
        print(f"读取配置文件失败: {str(e)}")
        return
    
    # 获取README中列出的模型
    readme_models = get_readme_models()
    
    # 处理每个模型
    success_count = 0
    total_models = len(readme_models)
    
    print(f"开始处理{total_models}个模型")
    if "models" not in config:
        print("错误: 配置文件中没有'models'部分")
        return
    
    # 打印配置中的所有模型
    print("配置文件中包含以下模型:")
    for model_key in config.get("models", {}):
        print(f"- {model_key}")
    
    for model_name in readme_models:
        print("-" * 50)
        if model_name in config.get("models", {}):
            peft_path = config["models"][model_name]["peft_path"]
            print(f"找到模型 {model_name} 的路径: {peft_path}")
            result = upload_model(model_name, peft_path, args.username, args.private)
            if result:
                success_count += 1
        else:
            print(f"警告: 在配置文件中未找到模型 {model_name} 的路径")
    
    print("=" * 50)
    print(f"上传完成: {success_count}/{total_models} 个模型上传成功")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"发生未预期的错误: {str(e)}")
        import traceback
        traceback.print_exc() 