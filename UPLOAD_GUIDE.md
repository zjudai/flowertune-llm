# 如何上传微调模型到Hugging Face Hub

本指南将帮助您将使用Flower框架微调的LoRA模型上传到Hugging Face Hub。

## 准备工作

1. **Hugging Face账号**: 您需要一个Hugging Face账号。如果没有，请前往[Hugging Face](https://huggingface.co/)注册。

2. **登录Hugging Face**:
   ```bash
   huggingface-cli login
   ```
   按照提示输入您的Hugging Face令牌(token)。您可以从[此处](https://huggingface.co/settings/tokens)获取令牌。

3. **安装必要依赖**:
   ```bash
   pip install huggingface_hub transformers peft
   ```

## 上传您的模型

我们提供了两个脚本来简化上传过程：

1. `upload_to_hf.py`: 主要的上传脚本
2. `upload_example.sh`: 使用示例

### 方法1: 使用示例脚本

1. 打开`upload_example.sh`并修改以下参数：
   - `checkpoint_path`: 您的LoRA模型检查点路径
   - `model_name`: 基础模型名称
   - `repo_name`: 您的Hugging Face仓库名称（格式为"用户名/模型名"）
   - `description`: 模型描述

2. 使脚本可执行并运行：
   ```bash
   chmod +x upload_example.sh
   ./upload_example.sh
   ```

### 方法2: 直接使用Python脚本

您也可以直接使用Python脚本：

```bash
python upload_to_hf.py \
    --checkpoint_path "results/YOUR_MODEL_PATH" \
    --model_name "MODEL_NAME" \
    --repo_name "USERNAME/REPO_NAME" \
    --description "模型描述"
```

参数说明：
- `--checkpoint_path`: LoRA检查点目录路径
- `--model_name`: 基础模型名称
- `--repo_name`: Hugging Face仓库名称
- `--private`: （可选）设置仓库为私有
- `--description`: （可选）模型描述

## 示例

假设您想上传基于"mistralai/Mistral-7B-v0.3"的微调模型：

```bash
python upload_to_hf.py \
    --checkpoint_path "results/2025-03-31_14-22-50/peft_5" \
    --model_name "mistralai/Mistral-7B-v0.3" \
    --repo_name "your-username/mistral-7b-flowertune-lora" \
    --description "基于Flower框架的联邦学习微调Mistral-7B模型"
```

## 上传后检查

上传完成后，您可以在以下URL查看您的模型：
```
https://huggingface.co/YOUR_USERNAME/REPO_NAME
```

## 使用上传的模型

上传后，您可以使用以下代码加载模型：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

# 加载基础模型和tokenizer
model_name = "mistralai/Mistral-7B-v0.3"  # 替换为您使用的基础模型
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 加载您的LoRA适配器
adapter_name = "your-username/mistral-7b-flowertune-lora"  # 替换为您的仓库名
model = PeftModel.from_pretrained(model, adapter_name)

# 现在您可以使用模型进行推理
inputs = tokenizer("请输入您的问题:", return_tensors="pt")
outputs = model.generate(**inputs, max_length=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
``` 