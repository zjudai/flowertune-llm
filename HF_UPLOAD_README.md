# 批量上传微调模型到Hugging Face Hub

本指南介绍如何使用提供的脚本批量上传您使用Flower框架微调的LoRA模型到Hugging Face Hub。

## 准备工作

1. 确保已经安装了必要的依赖：
   ```bash
   pip install huggingface_hub transformers peft
   ```

2. 登录到Hugging Face：
   ```bash
   huggingface-cli login
   ```
   按照提示输入您的访问令牌(token)。您可以从[此处](https://huggingface.co/settings/tokens)获取令牌。

## 可用脚本

我们提供了三个脚本用于不同的上传需求：

1. **upload_to_hf.py** - 基础上传脚本，用于上传单个模型
2. **upload_all_models.sh** - 自动批量上传所有模型的脚本
3. **upload_selected_models.sh** - 交互式脚本，让您选择要上传的特定模型

### 方法1：上传单个模型

使用`upload_to_hf.py`上传单个模型：

```bash
python upload_to_hf.py \
    --checkpoint_path "results/YOUR_CHECKPOINT_PATH/peft_5" \
    --model_name "MODEL_NAME" \
    --repo_name "YOUR_USERNAME/MODEL_REPO_NAME" \
    --description "模型描述" \
    [--private]  # 可选，设置仓库为私有
```

### 方法2：批量上传所有模型

使用`upload_all_models.sh`自动上传README中列出的所有模型：

```bash
chmod +x upload_all_models.sh
./upload_all_models.sh
```

此脚本会：
- 要求您输入Hugging Face用户名
- 自动查找每个模型的训练日志和PEFT检查点
- 为每个模型创建合适的仓库名称
- 批量上传所有找到检查点的模型

### 方法3：上传选定的模型（推荐）

使用`upload_selected_models.sh`交互式选择要上传的模型：

```bash
chmod +x upload_selected_models.sh
./upload_selected_models.sh
```

此脚本提供了更多灵活性：
- 显示所有可用模型并让您选择要上传的模型
- 允许手动指定检查点路径（如果自动查找失败）
- 允许自定义模型描述
- 可以选择将仓库设为私有

## 模型列表

以下是可以上传的模型列表：

1. Qwen/Qwen2.5-7B-Instruct
2. Qwen/Qwen2.5-1.5B-Instruct
3. mistralai/Mistral-7B-Instruct-v0.3
4. meta-llama/Llama-3.1-8B-Instruct
5. mistralai/Mistral-7B-v0.3
6. TinyLlama/TinyLlama-1.1B-Chat-v1.0
7. meta-llama/Llama-3.2-1B-Instruct
8. google/gemma-3-1b-it
9. deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

## 常见问题

1. **找不到训练日志或检查点？**
   - 使用`upload_selected_models.sh`脚本，它允许您手动指定检查点路径

2. **上传失败？**
   - 确保您已登录Hugging Face
   - 检查网络连接
   - 确认您有足够的权限创建仓库

3. **想修改仓库名称格式？**
   - 编辑脚本中的`REPO_NAME="${HF_USERNAME}/${MODEL_SHORT_LOWER}-flowertune-lora"`行

4. **上传后如何使用模型？**
   请参考UPLOAD_GUIDE.md文件中的"使用上传的模型"部分。 