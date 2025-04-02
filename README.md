# FlowerTune LLM on General NLP Dataset

This directory conducts federated instruction tuning with pretrained language models on a general NLP dataset [vicgalle/alpaca-gpt4](https://huggingface.co/datasets/vicgalle/alpaca-gpt4).
We use [Flower Datasets](https://flower.dev/docs/datasets/) to download, partition and preprocess the dataset.
Flower's Simulation Engine is used to simulate the LLM fine-tuning process in a federated way,
which allows users to perform the training on a single GPU.

## Important Links

- GitHub: [https://github.com/zjudai/flwr-nlp](https://github.com/zjudai/flwr-nlp)
- HuggingFace: [https://huggingface.co/zjudai/FlowerTune](https://huggingface.co/zjudai/FlowerTune)
- Model Collection: [FlowerTune LoRA Collection](https://huggingface.co/collections/zjudai/flowertune-lora-collection-67ecd5d0dae6145cbf798439)

## Experimental Setup

The dataset is divided into 20 partitions in an IID fashion, a partition is assigned to each ClientApp.
We randomly sample a fraction (0.1) of the total nodes to participate in each round, for a total of `10` rounds.
All settings are defined in `pyproject.toml`.

## Methodology

This baseline performs federated LLM fine-tuning with [LoRA](https://arxiv.org/abs/2106.09685) using the [🤗PEFT](https://huggingface.co/docs/peft/en/index) library.
The clients' models are aggregated with `FedAvg` strategy.
This provides a baseline performance for general NLP tasks with evaluation on MMLU benchmark.

### Example: Qwen2.5-7B-Instruct

For example, with the **Qwen/Qwen2.5-7B-Instruct** model we adopted the following fine-tuning methodology:

- **Precision**: `bf16` for model weights.
- **Quantization**: `4-bit` quantization for reduced memory usage.
- **LoRA Configuration**:
  - Rank (r): `32`
  - Alpha: `64`
- **Training Configuration**:
  - Batch size: `8`
  - Maximum number of steps: `10`
  - Total number of rounds: `10`
  - Fraction fit per round: `0.1`
- **Learning Rate Scheduler**:
  - Maximum LR: `5e-5`
  - Minimum LR: `1e-6`
  - Constant learning rate scheduler over steps
- **Strategy**: `FedAvg`

## Environment and Execution

### Environment Setup

Project dependencies are defined in `pyproject.toml`. Install them in an activated Python environment with:

```shell
python -m pip install --upgrade pip wheel setuptools packaging

pip install -e .
```

### Running the Training and Evaluation

We use a wrapper script `run_all_experiments.sh` to handle both training and evaluation processes:

```bash
# Example of running experiments
./run_all_experiments.sh --model Qwen/Qwen2.5-7B-Instruct --task general_nlp
```

The wrapper script sets up the proper environment, including:
- Activating the conda environment
- Setting up proxy configurations if needed
- Executing the main experiment runner script with the provided parameters

The actual experiment workflow is implemented in `run_experiments.py`, which is called by the wrapper script.

### Model Saving

The global PEFT model checkpoints are saved every 5 rounds after aggregation on the server side as default, which can be specified with `train.save-every-round` under [tool.flwr.app.config] entry in `pyproject.toml`.

## Evaluation Results

The evaluation was conducted on the MMLU (Massive Multitask Language Understanding) benchmark, which tests knowledge across various domains:

| **Model** | **STEM** | **Social Sciences** | **Humanities** | **Average** | **Comm. Costs** |
|-----------|----------|---------------------|----------------|-------------|-----------------|
| Qwen/Qwen2.5-7B-Instruct | 52.52% | 79.27% | 60.32% | 64.04% | 1540.55 MB |
| Qwen/Qwen2.5-1.5B-Instruct | 47.13% | 62.30% | 50.54% | 53.32% | 665.55 MB |
| mistralai/Mistral-7B-Instruct-v0.3 | 29.94% | 54.27% | 44.93% | 43.05% | 2080.62 MB |
| meta-llama/Llama-3.1-8B-Instruct | 22.87% | 39.55% | 32.05% | 31.49% | 2080.62 MB |
| TinyLlama/TinyLlama-1.1B-Chat-v1.0 | 14.18% | 21.61% | 21.91% | 19.23% | 687.93 MB |
| meta-llama/Llama-3.2-1B-Instruct | 12.88% | 17.61% | 6.16% | 12.22% | 520.31 MB |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | 0.54% | 0.00% | 0.04% | 0.19% | 665.55 MB |

## Hardware Details

For this experiment, I utilized a GPU-enabled virtual machine.

| **Component** | **Specification**    |
|---------------|----------------------|
| **GPU**       | 1 × GPU with 16+ GB  |
| **vCPUs**     | 6                    |
| **Memory**    | 16+ GB               |