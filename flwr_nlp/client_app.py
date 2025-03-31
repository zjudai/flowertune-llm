"""flwr-nlp: A Flower / FlowerTune app."""
# flwr-nlp: 基于Flower框架的联邦学习自然语言处理应用

import os
import warnings
from typing import Dict, Tuple

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from flwr.common.config import unflatten_dict
from flwr.common.typing import NDArrays, Scalar
from omegaconf import DictConfig

from transformers import TrainingArguments
from trl import SFTTrainer

from flwr_nlp.dataset import (
    get_tokenizer_and_data_collator_and_propt_formatting,
    load_data,
    replace_keys,
)
from flwr_nlp.models import (
    cosine_annealing,
    get_model,
    set_parameters,
    get_parameters,
)

# 避免警告信息
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["RAY_DISABLE_DOCKER_CPU_WARNING"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)


# pylint: disable=too-many-arguments
# pylint: disable=too-many-instance-attributes
class FlowerClient(NumPyClient):
    """Standard Flower client for CNN training."""
    # 标准的Flower客户端，用于语言模型训练

    def __init__(
        self,
        model_cfg: DictConfig,
        train_cfg: DictConfig,
        trainset,
        tokenizer,
        formatting_prompts_func,
        data_collator,
        num_rounds,
    ):  # pylint: disable=too-many-arguments
        # 初始化客户端
        # 参数:
        #   model_cfg: 模型配置
        #   train_cfg: 训练配置
        #   trainset: 训练数据集
        #   tokenizer: 分词器
        #   formatting_prompts_func: 提示格式化函数
        #   data_collator: 数据整理器
        #   num_rounds: 总训练轮次
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 设置设备(GPU或CPU)
        self.train_cfg = train_cfg  # 保存训练配置
        self.training_argumnets = TrainingArguments(**train_cfg.training_arguments)  # 创建训练参数对象
        self.tokenizer = tokenizer  # 保存分词器
        self.formatting_prompts_func = formatting_prompts_func  # 保存提示格式化函数
        self.data_collator = data_collator  # 保存数据整理器
        self.num_rounds = num_rounds  # 保存总训练轮次
        self.trainset = trainset  # 保存训练数据集

        # 实例化模型
        self.model = get_model(model_cfg)

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict]:
        """Implement distributed fit function for a given client."""
        # 实现给定客户端的分布式拟合函数
        # 参数:
        #   parameters: 全局模型参数
        #   config: 配置字典
        # 返回:
        #   (更新后的模型参数, 训练样本数量, 训练指标)
        
        # 使用全局参数设置本地模型
        set_parameters(self.model, parameters)

        # 计算新的学习率（使用余弦退火调度）
        new_lr = cosine_annealing(
            int(config["current_round"]),
            self.num_rounds,
            self.train_cfg.learning_rate_max,
            self.train_cfg.learning_rate_min,
        )

        # 更新训练参数
        self.training_argumnets.learning_rate = new_lr
        self.training_argumnets.output_dir = config["save_path"]

        # 构建训练器
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=self.training_argumnets,
            max_seq_length=self.train_cfg.seq_length,
            train_dataset=self.trainset,
            formatting_func=self.formatting_prompts_func,
            data_collator=self.data_collator,
        )

        # 执行本地训练
        results = trainer.train()

        # 返回更新后的模型参数、训练样本数量和训练损失
        return (
            get_parameters(self.model),
            len(self.trainset),
            {"train_loss": results.training_loss},
        )


def client_fn(context: Context) -> FlowerClient:
    """Create a Flower client representing a single organization."""
    # 创建代表单个组织的Flower客户端
    # 参数:
    #   context: 包含配置信息的上下文对象
    # 返回:
    #   配置好的FlowerClient实例
    
    # 从上下文获取配置信息
    partition_id = context.node_config["partition-id"]  # 分区ID
    num_partitions = context.node_config["num-partitions"]  # 总分区数
    num_rounds = context.run_config["num-server-rounds"]  # 服务器轮次
    cfg = DictConfig(replace_keys(unflatten_dict(context.run_config)))  # 转换配置为DictConfig格式

    # 获取客户端的数据分区
    client_trainset = load_data(partition_id, num_partitions, cfg.static.dataset.name)
    # 获取分词器、数据整理器和提示格式化函数
    (
        tokenizer,
        data_collator,
        formatting_prompts_func,
    ) = get_tokenizer_and_data_collator_and_propt_formatting(cfg.model.name)

    # 创建并返回FlowerClient实例
    return FlowerClient(
        cfg.model,
        cfg.train,
        client_trainset,
        tokenizer,
        formatting_prompts_func,
        data_collator,
        num_rounds,
    ).to_client()


# Flower ClientApp
app = ClientApp(client_fn)  # 创建Flower客户端应用
