#!/bin/bash

# 安装必要的依赖
pip install -q huggingface_hub transformers peft tomli

# 检查是否已登录Hugging Face
if ! huggingface-cli whoami &>/dev/null; then
    echo "未登录Hugging Face，请先登录"
    huggingface-cli login
fi

# 接收用户输入
read -p "请输入您的Hugging Face用户名: " HF_USERNAME
read -p "是否将模型仓库设为私有? (y/n，默认为n): " PRIVATE_REPO

# 设置私有标志
PRIVATE_FLAG=""
if [[ "$PRIVATE_REPO" =~ ^[Yy]$ ]]; then
    PRIVATE_FLAG="--private"
fi

# 如果配置文件存在，则从中读取路径上传模型
if [ -f "eval_config.toml" ]; then
    echo "找到eval_config.toml，使用配置文件中的路径上传模型..."
    python upload_from_config.py --username "$HF_USERNAME" $PRIVATE_FLAG
else
    echo "错误: 未找到eval_config.toml文件!"
    exit 1
fi 