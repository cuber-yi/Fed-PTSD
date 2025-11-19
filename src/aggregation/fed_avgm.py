import torch
from collections import OrderedDict, defaultdict


class FedAvgM:
    def __init__(self, config: dict):
        self.config = config
        self.name = "FedAvgM"
        # 动量因子 (beta) 和 服务器学习率 (server_lr)
        # 建议在 config.yaml 的 aggregation 下添加这些参数，这里给默认值
        self.beta = config.get('aggregation', {}).get('beta', 0.9)
        self.server_lr = config.get('aggregation', {}).get('server_lr', 1.0)

        self.global_params = None
        self.velocity = None  # 动量缓冲区

    def aggregate(self, client_parts_list: list, device: torch.device) -> dict:
        if not client_parts_list:
            return {}

        # 1. 计算标准的 FedAvg 聚合 (theta_avg)
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

        # 如果是第一轮，直接初始化 global_params 和 velocity
        if self.global_params is None:
            self.global_params = {
                pn: {k: v.clone() for k, v in p.items()}
                for pn, p in avg_parts.items()
            }
            self.velocity = {
                pn: {k: torch.zeros_like(v) for k, v in p.items()}
                for pn, p in avg_parts.items()
            }
            return avg_parts

        # 2. 计算伪梯度 (Pseudo-gradient): g = theta_old - theta_avg
        # 3. 更新动量: v = beta * v + g
        # 4. 更新参数: theta_new = theta_old - server_lr * v

        final_parts = OrderedDict()
        for part_name in avg_parts:
            final_parts[part_name] = OrderedDict()
            for key, avg_tensor in avg_parts[part_name].items():
                old_tensor = self.global_params[part_name][key]

                # 伪梯度 (注意方向，这里假设 avg 是经过 SGD 更新后的，所以 old - avg 代表梯度方向)
                grad = old_tensor - avg_tensor

                # 更新动量
                if key not in self.velocity[part_name]:
                    self.velocity[part_name][key] = torch.zeros_like(grad)

                self.velocity[part_name][key] = self.beta * self.velocity[part_name][key] + grad

                # 应用更新
                new_tensor = old_tensor - self.server_lr * self.velocity[part_name][key]
                final_parts[part_name][key] = new_tensor

                # 更新服务器状态
                self.global_params[part_name][key] = new_tensor.clone()

        return final_parts
