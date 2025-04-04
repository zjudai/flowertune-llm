---
base_model: mistralai/Mistral-7B-Instruct-v0.3
tags:
- peft
- lora
- federated-learning
- flower
---

# FlowerTune LoRA Model

This is a LoRA adapter for mistralai/Mistral-7B-Instruct-v0.3 fine-tuned with Flower federated learning framework on a general NLP dataset.

## Training Details

- Dataset: vicgalle/alpaca-gpt4
- Training method: Federated LoRA fine-tuning with FlowerTune
- Framework: Flower

这个模型是在federated learning环境下使用Flower框架微调的LoRA适配器。它在金融场景NLP数据上训练，表现优异，达到了0.5863的准确率和0.6711的F1分数。
