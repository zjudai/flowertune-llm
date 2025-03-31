# FlowerTune Evaluation for General NLP Tasks

This repository contains code for evaluating language models on various NLP tasks, with a focus on MMLU benchmark.

## Installation

1. Create a new conda environment (recommended):
```bash
conda create -n flwr-nlp python=3.10
conda activate flwr-nlp
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

To evaluate a model on MMLU benchmark:

```bash
python eval.py \
    --base-model-name-path "model_name" \
    --run-name "experiment_name" \
    --category "stem,social_sciences,humanities,other" \
    --batch-size 16 \
    --quantization 4
```

### Parameters

- `--base-model-name-path`: Path or name of the base model to evaluate
- `--run-name`: Name for this evaluation run
- `--peft-path`: (Optional) Path to PEFT weights
- `--datasets`: Comma-separated list of datasets to evaluate (default: "mmlu")
- `--category`: Comma-separated list of MMLU categories to evaluate
- `--batch-size`: Batch size for inference (default: 16)
- `--quantization`: Quantization bits (4 or 8, default: 4)
- `--device`: Device to run on (default: "cuda" if available, else "cpu")

### Example Commands

1. Evaluate a 4-bit quantized model:
```bash
python eval.py \
    --base-model-name-path "meta-llama/Llama-2-7b-chat-hf" \
    --run-name "llama2-7b-4bit" \
    --category "stem" \
    --quantization 4
```

2. Evaluate with PEFT weights:
```bash
python eval.py \
    --base-model-name-path "meta-llama/Llama-2-7b-chat-hf" \
    --run-name "llama2-7b-peft" \
    --peft-path "path/to/peft/weights" \
    --category "stem,social_sciences" \
    --quantization 4
```

## Troubleshooting

1. **Model Loading Issues**
   - Ensure you have the latest version of transformers and peft
   - Check if the model requires custom code (trust_remote_code=True is now set by default)
   - Verify model path and access permissions

2. **Quantization Issues**
   - For 4-bit quantization, ensure you have enough GPU memory
   - Try using 8-bit quantization if 4-bit fails
   - Check if the model supports quantization

3. **PEFT Loading Issues**
   - Verify PEFT weights are compatible with the base model
   - Check if the PEFT path exists and is accessible
   - Ensure PEFT weights were trained with the same model architecture

4. **Memory Issues**
   - Reduce batch size
   - Use gradient checkpointing
   - Enable low_cpu_mem_usage (already enabled by default)

## Results

Results will be saved in the `experiment_results` directory with the following format:
- `{model_name}_{run_name}_eval.log`: Detailed evaluation log
- `model_results_{timestamp}.csv`: Summary of results
