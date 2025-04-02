# 使用配置文件上传微调模型

本指南介绍如何使用`eval_config.toml`配置文件中的精确路径上传微调模型到Hugging Face Hub。

## 优势

使用配置文件上传有以下优势:

1. **精确路径**: 直接使用`eval_config.toml`中定义的路径，无需自动查找
2. **批量处理**: 自动上传README中表格列出的9个模型
3. **高可靠性**: 减少错误，确保每个模型使用正确的检查点

## 使用方法

### 准备工作

1. 确保已安装必要依赖:
   ```bash
   pip install huggingface_hub transformers peft tomli
   ```

2. 登录Hugging Face:
   ```bash
   huggingface-cli login
   ```

### 方式1: 使用Shell脚本（推荐）

最简单的方法是运行提供的shell脚本:

```bash
chmod +x upload_from_config.sh
./upload_from_config.sh
```

脚本会:
1. 检查必要的依赖并安装
2. 确认您已登录Hugging Face
3. 询问您的Hugging Face用户名
4. 询问是否将仓库设为私有
5. 使用`eval_config.toml`中的路径上传模型

### 方式2: 直接使用Python脚本

您也可以直接使用Python脚本:

```bash
python upload_from_config.py --username YOUR_USERNAME [--private]
```

参数说明:
- `--username`: 您的Hugging Face用户名
- `--config_file`: 配置文件路径（默认为"eval_config.toml"）
- `--private`: 设置仓库为私有（可选）

## 配置文件内容

脚本会从`eval_config.toml`文件中读取以下格式的配置:

```toml
[models."Qwen/Qwen2.5-7B-Instruct"]
peft_path = "/path/to/model/checkpoint"
```

脚本将上传README.md表格中的9个模型:

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

**问题**: 上传失败，提示"not enough rights to create this repository"
**解答**: 确保您已正确登录Hugging Face并有足够的权限创建仓库。

**问题**: 某些模型在配置文件中找不到路径
**解答**: 确保`eval_config.toml`中包含了README.md表格中所有模型的路径。

**问题**: 上传后模型无法正常加载
**解答**: 确保配置文件中指定的路径包含完整有效的LoRA适配器文件（adapter_model.safetensors和adapter_config.json）。 