#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from huggingface_hub import HfApi, create_repo, upload_folder
from peft import PeftConfig

def parse_args():
    parser = argparse.ArgumentParser(description="Upload PEFT model to Hugging Face Hub")
    parser.add_argument(
        "--username", 
        type=str, 
        default="zjudai",
        help="Your Hugging Face username"
    )
    parser.add_argument(
        "--private", 
        action="store_true",
        help="Make the repository private"
    )
    return parser.parse_args()

def upload_model(username, private=False):
    """Upload a specific model to Hugging Face Hub"""
    # 设置模型参数
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    peft_path = "results/finance_mistralai_Mistral-7B-Instruct-v0.3_20250404_014217/peft_10"
    
    print(f"\n开始处理模型: {model_name}")
    # 确认检查点路径存在
    if not os.path.exists(peft_path):
        print(f"错误: 检查点路径不存在: {peft_path}")
        return False
    
    # 提取模型简称作为仓库名
    model_short = model_name.split("/")[-1]
    model_short_lower = model_short.lower()
    repo_name = f"{username}/flowertune-finance-lora-{model_short_lower}"
    
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
    description = f"This model is a LoRA adapter fine-tuned on {model_name} using the Flower federated learning framework. It was trained on finance NLP datasets through distributed learning. The model achieved scores of FPB: 0.5863, FIQA: 0.6711, TFNS: 0.5863, with an average score of 0.6145."
    
    model_card = f"""---
base_model: {model_name}
tags:
- peft
- lora
- federated-learning
- flower
- finance
---

# FlowerTune Finance LoRA Model

This is a LoRA adapter for {model_name} fine-tuned with Flower federated learning framework on finance NLP datasets.

## Model Performance
- FPB: 0.5863
- FIQA: 0.6711
- TFNS: 0.5863
- Average: 0.6145

## Training Details
- Domain: Finance
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
            commit_message="Upload LoRA adapter for finance domain"
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
    
    # 检查是否已登录Hugging Face
    try:
        api = HfApi()
        whoami = api.whoami()
        print(f"已登录Hugging Face，用户: {whoami['name']}")
    except Exception as e:
        print(f"未登录Hugging Face或访问出错: {str(e)}")
        print("请先运行: huggingface-cli login")
        return
    
    # 上传模型
    success = upload_model(args.username, args.private)
    
    if success:
        print("模型上传成功！")
    else:
        print("模型上传失败。")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"发生未预期的错误: {str(e)}")
        import traceback
        traceback.print_exc() 