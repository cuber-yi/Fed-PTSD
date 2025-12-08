import torch
from collections import OrderedDict, defaultdict
import math


class FedAdagrad:
    def __init__(self, config: dict):
        self.config = config
        self.name = "FedAdagrad"
        # 默认参数
        self.beta1 = config.get('aggregation', {}).get('beta1', 0.9)  # 动量因子
        self.epsilon = config.get('aggregation', {}).get('epsilon', 1e-8)
        self.server_lr = config.get('aggregation', {}).get('server_lr', 0.01)

        self.global_params = None
        self.exp_avg = None  # 动量 (m)
        self.sum_sq_grad = None  # 累积平方梯度 (v)

    def aggregate(self, client_parts_list: list, device: torch.device) -> dict:
        if not client_parts_list:
            return {}

        # 1. 简单的平均聚合得到 pseudo-gradient
        collected_tensors = defaultdict(lambda: defaultdict(list))
        part_keys = client_parts_list[0].keys()

        for client_parts in client_parts_list:
            for part_name in part_keys:
                for key, param_tensor in client_parts[part_name].items():
                    collected_tensors[part_name][key].append(param_tensor)

        avg_parts = OrderedDict()
        for part_name, keys_dict in collected_tensors.items():
            avg_parts[part_name] = OrderedDict()
            for key, tensor_list in keys_dict.items():
                avg_parts[part_name][key] = torch.stack(tensor_list).mean(dim=0).to(device)

        # 初始化全局状态
        if self.global_params is None:
            self.global_params = {pn: {k: v.clone() for k, v in p.items()} for pn, p in avg_parts.items()}
            self.exp_avg = {pn: {k: torch.zeros_like(v) for k, v in p.items()} for pn, p in avg_parts.items()}
            self.sum_sq_grad = {pn: {k: torch.zeros_like(v) for k, v in p.items()} for pn, p in avg_parts.items()}
            return avg_parts

        # 2. Server-side Adagrad Update
        final_parts = OrderedDict()
        for part_name in avg_parts:
            final_parts[part_name] = OrderedDict()
            for key, avg_tensor in avg_parts[part_name].items():
                old_tensor = self.global_params[part_name][key]
                # 伪梯度: g = w_old - w_avg (注意方向)
                grad = old_tensor - avg_tensor

                # Update momentum
                self.exp_avg[part_name][key] = self.beta1 * self.exp_avg[part_name][key] + (1 - self.beta1) * grad

                # Update sum squared gradients (Adagrad specific: simply accumulate)
                self.sum_sq_grad[part_name][key] += grad ** 2

                # Update parameters
                m = self.exp_avg[part_name][key]
                v = self.sum_sq_grad[part_name][key]

                denom = v.sqrt().add_(self.epsilon)
                new_tensor = old_tensor - self.server_lr * (m / denom)

                final_parts[part_name][key] = new_tensor
                self.global_params[part_name][key] = new_tensor.clone()

        return final_parts
