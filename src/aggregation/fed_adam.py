import torch
from collections import OrderedDict, defaultdict
import math


class FedAdam:
    def __init__(self, config: dict):
        self.config = config
        self.name = "FedAdam"
        # Adam 参数
        self.beta1 = config.get('aggregation', {}).get('beta1', 0.9)
        self.beta2 = config.get('aggregation', {}).get('beta2', 0.99)
        self.epsilon = config.get('aggregation', {}).get('epsilon', 1e-8)
        self.server_lr = config.get('aggregation', {}).get('server_lr', 0.01)  # FedAdam 的 server_lr 通常较小

        self.global_params = None
        self.exp_avg = None
        self.exp_avg_sq = None
        self.t = 0

    def aggregate(self, client_parts_list: list, device: torch.device) -> dict:
        if not client_parts_list:
            return {}

        # 1. 计算 FedAvg
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

        if self.global_params is None:
            self.global_params = {
                pn: {k: v.clone() for k, v in p.items()} for pn, p in avg_parts.items()
            }
            self.exp_avg = {
                pn: {k: torch.zeros_like(v) for k, v in p.items()} for pn, p in avg_parts.items()
            }
            self.exp_avg_sq = {
                pn: {k: torch.zeros_like(v) for k, v in p.items()} for pn, p in avg_parts.items()
            }
            return avg_parts

        self.t += 1

        final_parts = OrderedDict()
        for part_name in avg_parts:
            final_parts[part_name] = OrderedDict()
            for key, avg_tensor in avg_parts[part_name].items():
                old_tensor = self.global_params[part_name][key]
                grad = old_tensor - avg_tensor

                # Adam Update
                if key not in self.exp_avg[part_name]:
                    self.exp_avg[part_name][key] = torch.zeros_like(grad)
                    self.exp_avg_sq[part_name][key] = torch.zeros_like(grad)

                m = self.exp_avg[part_name][key]
                v = self.exp_avg_sq[part_name][key]

                m = self.beta1 * m + (1 - self.beta1) * grad
                v = self.beta2 * v + (1 - self.beta2) * (grad ** 2)

                self.exp_avg[part_name][key] = m
                self.exp_avg_sq[part_name][key] = v

                # Bias correction (可选，标准Adam有，FedOpt论文中有时省略，这里加上)
                bias_correction1 = 1 - self.beta1 ** self.t
                bias_correction2 = 1 - self.beta2 ** self.t
                step_size = self.server_lr / bias_correction1

                denom = (v.sqrt() / math.sqrt(bias_correction2)).add_(self.epsilon)

                new_tensor = old_tensor - step_size * (m / denom)

                final_parts[part_name][key] = new_tensor
                self.global_params[part_name][key] = new_tensor.clone()

        return final_parts
