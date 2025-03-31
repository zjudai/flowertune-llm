"""flwr-nlp: A Flower / FlowerTune app."""
# flwr-nlp: 基于Flower框架的联邦学习自然语言处理应用

from io import BytesIO
from logging import INFO, WARN
from typing import List, Tuple, Union

from flwr.common import FitIns, FitRes, Parameters, log, parameters_to_ndarrays
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg, FedOpt


class FlowerTuneLlm(FedAvg):
    """Customised FedAvg strategy implementation.
    
    This class behaves just like FedAvg but also tracks the communication
    costs associated with `fit` over FL rounds.
    """
    # 自定义FedAvg策略实现
    # 该类表现得像标准FedAvg，但额外跟踪了FL轮次中与`fit`相关的通信成本
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.comm_tracker = CommunicationTracker()  # 初始化通信跟踪器

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ):
        """Configure the next round of training."""
        # 配置下一轮训练
        # 参数:
        #   server_round: 服务器轮次
        #   parameters: 全局模型参数
        #   client_manager: 客户端管理器
        # 返回:
        #   配置好的客户端和fit指令列表
        return_clients = super().configure_fit(server_round, parameters, client_manager)

        # 测量通信成本
        fit_ins_list = [fit_ins for _, fit_ins in return_clients]
        self.comm_tracker.track(fit_ins_list)

        return return_clients

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ):
        """Aggregate fit results using weighted average."""
        # 使用加权平均聚合拟合结果
        # 参数:
        #   server_round: 服务器轮次
        #   results: 客户端训练结果列表
        #   failures: 失败的客户端或异常列表
        # 返回:
        #   聚合后的参数和指标
        
        # 测量通信成本
        fit_res_list = [fit_res for _, fit_res in results]
        self.comm_tracker.track(fit_res_list)

        parameters_aggregated, metrics_aggregated = super().aggregate_fit(
            server_round, results, failures
        )

        return parameters_aggregated, metrics_aggregated


class CommunicationTracker:
    """Communication costs tracker over FL rounds."""
    # FL轮次通信成本跟踪器
    
    def __init__(self):
        self.curr_comm_cost = 0.0  # 当前累计通信成本(MB)

    @staticmethod
    def _compute_bytes(parameters):
        # 计算参数的字节大小
        return sum([BytesIO(t).getbuffer().nbytes for t in parameters.tensors])

    def track(self, fit_list: List[Union[FitIns, FitRes]]):
        # 跟踪一组fit指令或结果的通信成本
        # 参数:
        #   fit_list: fit指令或结果列表
        size_bytes_list = [
            self._compute_bytes(fit_ele.parameters)
            for fit_ele in fit_list
        ]
        comm_cost = sum(size_bytes_list) / 1024**2  # 转换为MB

        self.curr_comm_cost += comm_cost  # 累加通信成本
        log(
            INFO,
            "Communication budget: used %.2f MB (+%.2f MB this round) / 200,000 MB",
            self.curr_comm_cost,
            comm_cost,
        )

        # 如果超过预设的通信预算(200,000 MB)，发出警告
        if self.curr_comm_cost > 2e5:
            log(
                WARN,
                "The accumulated communication cost has exceeded 200,000 MB. "
                "Please consider reducing it if you plan to participate "
                "FlowerTune LLM Leaderboard.",
            )
