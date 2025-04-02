#!/bin/bash

# 安装必要的依赖
pip install -q huggingface_hub transformers peft

# 登录Hugging Face（如果尚未登录）
echo "请确保已经通过huggingface-cli login登录到Hugging Face"
echo "如果尚未登录，请运行 huggingface-cli login 并输入您的token"
read -p "按Enter键继续..."

# 您的Hugging Face用户名
read -p "请输入您的Hugging Face用户名: " HF_USERNAME

# 检查模型路径
echo "正在检查模型目录..."
RESULTS_DIR="results"
if [ ! -d "$RESULTS_DIR" ]; then
    echo "错误: $RESULTS_DIR 目录不存在!"
    exit 1
fi

# 模型列表（从README.md中提取）
declare -a MODELS=(
    "Qwen/Qwen2.5-7B-Instruct"
    "Qwen/Qwen2.5-1.5B-Instruct"
    "mistralai/Mistral-7B-Instruct-v0.3"
    "meta-llama/Llama-3.1-8B-Instruct"
    "mistralai/Mistral-7B-v0.3"
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    "meta-llama/Llama-3.2-1B-Instruct"
    "google/gemma-3-1b-it"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
)

# 上传所有模型
echo "开始上传模型..."
for MODEL in "${MODELS[@]}"
do
    # 提取模型简称作为仓库名
    MODEL_SHORT=$(echo $MODEL | sed 's|.*/||')
    MODEL_SHORT_LOWER=$(echo $MODEL_SHORT | tr '[:upper:]' '[:lower:]')
    REPO_NAME="${HF_USERNAME}/${MODEL_SHORT_LOWER}-flowertune-lora"
    
    # 查找对应的训练日志
    MODEL_SAFE_NAME="${MODEL//\//_}"
    TRAIN_LOG=$(find experiment_results -name "${MODEL_SAFE_NAME}_train.log" | head -1)
    
    if [ -z "$TRAIN_LOG" ]; then
        echo "警告: 未找到 $MODEL 的训练日志"
        echo "跳过上传 $MODEL"
        echo "-----------------------------------------"
        continue
    fi
    
    echo "找到训练日志: $TRAIN_LOG"
    
    # 从训练日志提取实验标识符（通常是时间戳）
    # 尝试从训练日志路径中提取时间戳
    TIMESTAMP=$(echo "$TRAIN_LOG" | grep -o -E '[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2}-[0-9]{2}-[0-9]{2}' || echo "")
    
    if [ -z "$TIMESTAMP" ]; then
        # 如果无法从日志中提取时间戳，查找最新的结果目录
        echo "无法从训练日志中提取时间戳，使用最新的结果目录"
        RESULT_DIR=$(ls -td $RESULTS_DIR/*/ | head -1)
    else
        # 尝试找到对应时间戳的结果目录
        RESULT_DIR=$(find $RESULTS_DIR -maxdepth 1 -type d -name "*$TIMESTAMP*" | head -1)
        
        if [ -z "$RESULT_DIR" ]; then
            echo "未找到与时间戳 $TIMESTAMP 匹配的结果目录，使用最新的结果目录"
            RESULT_DIR=$(ls -td $RESULTS_DIR/*/ | head -1)
        fi
    fi
    
    echo "使用结果目录: $RESULT_DIR"
    
    # 查找peft检查点目录
    PEFT_DIRS=$(find $RESULT_DIR -name "peft_*" -type d)
    
    if [ -z "$PEFT_DIRS" ]; then
        echo "错误: 在 $RESULT_DIR 中未找到peft检查点目录!"
        echo "跳过上传 $MODEL"
        echo "-----------------------------------------"
        continue
    fi
    
    # 提取最高的检查点编号
    MAX_PEFT_NUMBER=$(echo "$PEFT_DIRS" | grep -o 'peft_[0-9]*' | sort -t_ -k2 -nr | head -1 | cut -d_ -f2)
    CHECKPOINT_PATH="${RESULT_DIR}peft_${MAX_PEFT_NUMBER}"
    
    if [ -d "$CHECKPOINT_PATH" ]; then
        echo "正在上传模型: $MODEL"
        echo "检查点路径: $CHECKPOINT_PATH"
        echo "仓库名称: $REPO_NAME"
        
        DESCRIPTION="这个模型是在federated learning环境下使用Flower框架微调的LoRA适配器，基于$MODEL模型。它在general NLP dataset (vicgalle/alpaca-gpt4)上训练，通过分布式学习实现性能提升。"
        
        # 上传模型
        python upload_to_hf.py \
            --checkpoint_path "$CHECKPOINT_PATH" \
            --model_name "$MODEL" \
            --repo_name "$REPO_NAME" \
            --description "$DESCRIPTION"
        
        echo "模型 $MODEL 上传完成!"
        echo "-----------------------------------------"
    else
        echo "警告: 未找到 $MODEL 的检查点路径: $CHECKPOINT_PATH"
        echo "跳过上传 $MODEL"
        echo "-----------------------------------------"
    fi
done

echo "所有模型上传完成!" 