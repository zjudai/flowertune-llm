#!/bin/bash

# 安装必要的依赖
pip install -q huggingface_hub transformers peft

# 登录Hugging Face（如果尚未登录）
# huggingface-cli login

# 示例: 上传Mistral-7B-v0.3的微调模型
# 请替换以下参数为您的实际参数
# YOUR_USERNAME: 您的Hugging Face用户名
# results/2025-03-31_14-22-50/peft_5: 您的模型检查点路径
python upload_to_hf.py \
    --checkpoint_path "results/2025-03-31_14-22-50/peft_5" \
    --model_name "mistralai/Mistral-7B-v0.3" \
    --repo_name "YOUR_USERNAME/mistral-7b-v0.3-flowertune-lora" \
    --description "这个模型是在federated learning环境下使用Flower框架微调的LoRA适配器。它在general NLP dataset上训练，展示了分布式学习的优势。"

# 如果您想上传其他模型，请取消注释并修改以下示例
# python upload_to_hf.py \
#     --checkpoint_path "results/YOUR_CHECKPOINT_PATH" \
#     --model_name "Qwen/Qwen2.5-7B-Instruct" \
#     --repo_name "YOUR_USERNAME/qwen-7b-instruct-flowertune-lora" \
#     --description "使用Flower联邦学习框架微调的Qwen2.5-7B-Instruct LoRA模型" 