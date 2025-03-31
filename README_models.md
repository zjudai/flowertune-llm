# 模型批量下载工具

这个工具用于批量下载Hugging Face模型，支持并行下载，避免每次运行代码时重新下载模型。

## 功能特点

- 支持批量下载多个模型
- 支持并行下载，提高下载效率
- 支持模型量化配置
- 详细的日志输出
- 支持通过YAML配置文件定义多个模型

## 使用方法

### 1. 下载单个模型

```bash
# 下载单个模型（不量化）
python download_models.py --model "meta-llama/Llama-3.2-1B-Instruct"

# 下载单个模型并应用4位量化
python download_models.py --model "meta-llama/Llama-3.2-1B-Instruct" --quantization 4

# 下载单个模型并指定保存目录
python download_models.py --model "meta-llama/Llama-3.2-1B-Instruct" --output-dir "./models"
```

### 2. 批量下载多个模型

准备一个YAML配置文件（例如`models_config.yaml`）：

```yaml
models:
  - name: "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    quantization: 4
  - name: "meta-llama/Llama-3.2-1B-Instruct"
    quantization: 4
  # 更多模型...
```

然后使用配置文件下载：

```bash
# 使用默认的并行数（4）下载
python download_models.py --config models_config.yaml

# 指定最大并行下载数为8
python download_models.py --config models_config.yaml --workers 8

# 指定输出目录
python download_models.py --config models_config.yaml --output-dir "./downloaded_models"
```

## 参数说明

- `--model`: 要下载的单个模型名称
- `--quantization`: 量化位数（4或8）
- `--output-dir`: 模型保存目录
- `--config`: 包含多个模型的配置文件路径
- `--workers`: 最大并行下载数量（默认为4）

## 配置文件格式

YAML格式的配置文件包含一个`models`列表，每个模型可以包含以下参数：

- `name`: 模型名称（必需）
- `quantization`: 量化位数（可选，4或8）
- `output_dir`: 保存目录（可选）

## 注意事项

1. 确保已安装所有必要的依赖：
   ```bash
   pip install transformers torch bitsandbytes pyyaml huggingface-hub
   ```

2. 如果使用量化参数，确保您的环境支持相应的量化方法

3. 对于需要登录的模型：
   ```bash
   huggingface-cli login
   ``` 