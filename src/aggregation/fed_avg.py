"""
FedAvg - 联邦平均聚合
经典的联邦学习聚合算法，对所有客户端参数进行简单平均
参考: McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data", AISTATS 2017
"""
import torch
from collections import OrderedDict, defaultdict


class FedAvg:
    def __init__(self, config: dict):
        self.config = config
        self.name = "FedAvg"

    def aggregate(self, client_parts_list: list, device: torch.device) -> dict:
        """
        对客户端参数进行简单平均聚合

        参数:
            client_parts_list: 客户端参数列表
            device: 计算设备

        返回:
            aggregated_parts: 聚合后的参数字典
        """
        if not client_parts_list:
            return {}

        # 收集所有参数
        collected_tensors = defaultdict(lambda: defaultdict(list))
        part_keys = client_parts_list[0].keys()

        for client_parts in client_parts_list:
            for part_name in part_keys:
                for key, param_tensor in client_parts[part_name].items():
                    collected_tensors[part_name][key].append(param_tensor)

        # 简单平均聚合
        aggregated_parts = OrderedDict()
        for part_name, keys_dict in collected_tensors.items():
            aggregated_parts[part_name] = OrderedDict()
            for key, tensor_list in keys_dict.items():
                aggregated_tensor = torch.stack(tensor_list).mean(dim=0)
                aggregated_parts[part_name][key] = aggregated_tensor.to(device)

        return aggregated_parts
