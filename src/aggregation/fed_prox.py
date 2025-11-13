"""
FedProx - 带近端项的联邦学习
"""
import torch
from collections import OrderedDict, defaultdict


class FedProx:
    def __init__(self, config: dict):
        self.config = config
        self.mu = config.get('aggregation', {}).get('mu', 0.01)
        self.name = "FedProx"
        self.global_params = None

    def aggregate(self, client_parts_list: list, device: torch.device) -> dict:
        """FedProx聚合"""
        if not client_parts_list:
            return {}

        num_clients = len(client_parts_list)
        part_keys = client_parts_list[0].keys()

        if self.global_params is None:
            return self._fedavg_aggregate(client_parts_list, device, part_keys)

        # 计算每个客户端与全局模型的距离
        distances = []
        for client_parts in client_parts_list:
            dist = 0.0
            for part_name in part_keys:
                for key, param_tensor in client_parts[part_name].items():
                    if key in self.global_params[part_name]:
                        diff = param_tensor - self.global_params[part_name][key]
                        dist += torch.sum(diff ** 2).item()
            distances.append(dist)

        # 转换距离为权重
        max_dist = max(distances) if max(distances) > 0 else 1.0
        weights = [1.0 / (1.0 + self.mu * (d / max_dist)) for d in distances]
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        # 加权聚合
        collected_tensors = defaultdict(lambda: defaultdict(list))
        for client_parts in client_parts_list:
            for part_name in part_keys:
                for key, param_tensor in client_parts[part_name].items():
                    collected_tensors[part_name][key].append(param_tensor)

        aggregated_parts = OrderedDict()
        for part_name, keys_dict in collected_tensors.items():
            aggregated_parts[part_name] = OrderedDict()
            for key, tensor_list in keys_dict.items():
                weighted_sum = sum(w * t for w, t in zip(weights, tensor_list))
                aggregated_parts[part_name][key] = weighted_sum.to(device)

        self.global_params = {
            part_name: {k: v.clone() for k, v in params.items()}
            for part_name, params in aggregated_parts.items()
        }

        return aggregated_parts

    def _fedavg_aggregate(self, client_parts_list, device, part_keys):
        """标准FedAvg聚合（用于第一轮）"""
        collected_tensors = defaultdict(lambda: defaultdict(list))

        for client_parts in client_parts_list:
            for part_name in part_keys:
                for key, param_tensor in client_parts[part_name].items():
                    collected_tensors[part_name][key].append(param_tensor)

        aggregated_parts = OrderedDict()
        for part_name, keys_dict in collected_tensors.items():
            aggregated_parts[part_name] = OrderedDict()
            for key, tensor_list in keys_dict.items():
                aggregated_tensor = torch.stack(tensor_list).mean(dim=0)
                aggregated_parts[part_name][key] = aggregated_tensor.to(device)

        self.global_params = {
            part_name: {k: v.clone() for k, v in params.items()}
            for part_name, params in aggregated_parts.items()
        }

        return aggregated_parts
