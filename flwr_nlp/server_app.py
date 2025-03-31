"""flwr-nlp: A Flower / FlowerTune app."""
# flwr-nlp: 基于Flower框架的联邦学习自然语言处理应用

import os
from datetime import datetime

from flwr.common import Context, ndarrays_to_parameters
from flwr.common.config import unflatten_dict
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from omegaconf import DictConfig

from flwr_nlp.models import get_model, get_parameters, set_parameters
from flwr_nlp.dataset import replace_keys
from flwr_nlp.strategy import FlowerTuneLlm


# 获取在策略的evaluate()方法中执行的函数
# 这里我们用它来保存全局模型检查点
def get_evaluate_fn(model_cfg, save_every_round, total_round, save_path):
    """Return an evaluation function for saving global model."""
    # 返回一个用于保存全局模型的评估函数
    # 参数:
    #   model_cfg: 模型配置
    #   save_every_round: 保存模型的轮次间隔
    #   total_round: 总训练轮次
    #   save_path: 保存路径
    # 返回:
    #   evaluate函数, 用于保存模型

    def evaluate(server_round: int, parameters, config):
        # 保存模型
        # 参数:
        #   server_round: 当前服务器轮次
        #   parameters: 当前全局模型参数
        #   config: 配置字典
        # 返回:
        #   评估分数(这里为0.0)和空字典
        
        # 在特定轮次保存模型（最后一轮或每save_every_round轮）
        if server_round != 0 and (
            server_round == total_round or server_round % save_every_round == 0
        ):
            # 清理CUDA内存
            import torch
            import gc
            
            # 执行垃圾回收
            gc.collect()
            
            # 如果CUDA可用，清空缓存
            if torch.cuda.is_available():
                # 打印清理前的GPU内存
                print(f"GPU memory before cleanup: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
                # 释放缓存
                torch.cuda.empty_cache()
                # 确保所有CUDA操作完成
                torch.cuda.synchronize()
                # 打印清理后的GPU内存
                print(f"GPU memory after cleanup: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            
            # 初始化模型
            model = get_model(model_cfg)
            set_parameters(model, parameters)

            # 保存模型到指定路径
            model.save_pretrained(f"{save_path}/peft_{server_round}")
            
            # 保存后再次清理内存
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return 0.0, {}

    return evaluate


def get_on_fit_config(save_path):
    """Return a function that will be used to construct the config that the
    client's fit() method will receive."""
    # 返回一个函数，用于构建客户端fit()方法将接收的配置
    # 参数:
    #   save_path: 保存路径
    # 返回:
    #   fit_config_fn函数, 用于生成训练配置

    def fit_config_fn(server_round: int):
        # 为每轮训练生成配置
        # 参数:
        #   server_round: 当前服务器轮次
        # 返回:
        #   包含当前轮次和保存路径的配置字典
        fit_config = {}
        fit_config["current_round"] = server_round
        fit_config["save_path"] = save_path
        return fit_config

    return fit_config_fn


def fit_weighted_average(metrics):
    """Aggregate (federated) evaluation metrics."""
    # 聚合联邦评估指标
    # 参数:
    #   metrics: 包含样本数和训练损失的列表
    # 返回:
    #   聚合后的指标字典
    
    # 将每个客户端的损失乘以样本数
    losses = [num_examples * m["train_loss"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # 聚合并返回自定义指标（加权平均）
    return {"train_loss": sum(losses) / sum(examples)}


def server_fn(context: Context):
    """Construct components that set the ServerApp behaviour."""
    # 构建设置ServerApp行为的组件
    # 参数:
    #   context: 包含配置信息的上下文对象
    # 返回:
    #   包含策略和配置的ServerAppComponents对象
    
    # 根据当前时间戳创建输出目录
    current_time = datetime.now()
    folder_name = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(os.getcwd(), f"results/{folder_name}")
    os.makedirs(save_path, exist_ok=True)

    # 从配置中读取信息
    num_rounds = context.run_config["num-server-rounds"]
    cfg = DictConfig(replace_keys(unflatten_dict(context.run_config)))

    # 获取初始模型权重
    init_model = get_model(cfg.model)
    init_model_parameters = get_parameters(init_model)
    init_model_parameters = ndarrays_to_parameters(init_model_parameters)

    # 定义策略
    strategy = FlowerTuneLlm(
        fraction_fit=cfg.strategy.fraction_fit,  # 每轮训练的客户端比例
        fraction_evaluate=cfg.strategy.fraction_evaluate,  # 每轮评估的客户端比例
        on_fit_config_fn=get_on_fit_config(save_path),  # 训练配置生成函数
        fit_metrics_aggregation_fn=fit_weighted_average,  # 训练指标聚合函数
        initial_parameters=init_model_parameters,  # 初始模型参数
        evaluate_fn=get_evaluate_fn(
            cfg.model, cfg.train.save_every_round, num_rounds, save_path
        ),  # 评估函数
    )
    config = ServerConfig(num_rounds=num_rounds)  # 服务器配置

    return ServerAppComponents(strategy=strategy, config=config)


# Flower ServerApp
app = ServerApp(server_fn=server_fn)  # 创建Flower服务器应用
