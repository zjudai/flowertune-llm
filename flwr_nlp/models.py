"""flwr-nlp: A Flower / FlowerTune app."""
# flwr-nlp: 基于Flower框架的联邦学习自然语言处理应用

import math

import torch
from omegaconf import DictConfig
from collections import OrderedDict
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from peft.utils import prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from flwr.common.typing import NDArrays


def cosine_annealing(
    current_round: int,
    total_round: int,
    lrate_max: float = 0.001,
    lrate_min: float = 0.0,
) -> float:
    """Implement cosine annealing learning rate schedule."""
    # 实现余弦退火学习率调度
    # 参数:
    #   current_round: 当前轮次
    #   total_round: 总轮次
    #   lrate_max: 最大学习率，默认为0.001
    #   lrate_min: 最小学习率，默认为0.0
    # 返回:
    #   当前轮次对应的学习率
    cos_inner = math.pi * current_round / total_round
    return lrate_min + 0.5 * (lrate_max - lrate_min) * (1 + math.cos(cos_inner))


def get_model(model_cfg: DictConfig):
    """Load model with appropriate quantization config and other optimizations.
    """
    # 加载模型并应用适当的量化配置和其他优化
    # 参数:
    #   model_cfg: 包含模型配置的字典配置对象
    # 返回:
    #   配置好的PEFT模型
    
    # 根据配置选择量化方法
    if model_cfg.quantization == 4:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)  # 使用4位量化
    elif model_cfg.quantization == 8:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)  # 使用8位量化
    else:
        raise ValueError(
            f"Use 4-bit or 8-bit quantization. You passed: {model_cfg.quantization}/"
        )

    # 加载预训练的因果语言模型
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.name,
        quantization_config=quantization_config,  # 应用量化配置
        torch_dtype=torch.bfloat16,  # 使用bfloat16数据类型
        low_cpu_mem_usage=True,  # 启用低CPU内存使用
    )

    # 为k位量化训练准备模型
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=model_cfg.gradient_checkpointing
    )

    # 配置LoRA参数高效微调
    peft_config = LoraConfig(
        r=model_cfg.lora.peft_lora_r,  # LoRA秩
        lora_alpha=model_cfg.lora.peft_lora_alpha,  # LoRA alpha参数
        lora_dropout=0.075,  # LoRA dropout率
        task_type="CAUSAL_LM",  # 任务类型：因果语言模型
    )

    # 如果启用梯度检查点，禁用模型缓存以节省内存
    if model_cfg.gradient_checkpointing:
        model.config.use_cache = False

    # 返回应用了PEFT配置的模型
    return get_peft_model(model, peft_config)


def set_parameters(model, parameters: NDArrays) -> None:
    """Change the parameters of the model using the given ones."""
    # 使用给定参数更新模型参数
    # 参数:
    #   model: 待更新的模型
    #   parameters: 新参数值（NumPy数组格式）
    # 返回:
    #   无
    peft_state_dict_keys = get_peft_model_state_dict(model).keys()  # 获取PEFT模型状态字典的键
    params_dict = zip(peft_state_dict_keys, parameters)  # 将键与新参数值配对
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})  # 创建新的状态字典
    set_peft_model_state_dict(model, state_dict)  # 设置PEFT模型的状态字典


def get_parameters(model) -> NDArrays:
    """Return the parameters of the current net."""
    # 获取当前模型的参数
    # 参数:
    #   model: 待获取参数的模型
    # 返回:
    #   模型参数列表（NumPy数组格式）
    state_dict = get_peft_model_state_dict(model)  # 获取PEFT模型状态字典
    return [val.cpu().numpy() for _, val in state_dict.items()]  # 将参数转换为NumPy数组并返回
