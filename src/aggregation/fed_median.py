import torch
from collections import OrderedDict, defaultdict

class FedMedian:
    def __init__(self, config: dict):
        self.config = config
        self.name = "FedMedian"

    def aggregate(self, client_parts_list: list, device: torch.device) -> dict:
        """
        Coordinate-wise Median Aggregation
        """
        if not client_parts_list:
            return {}

        collected_tensors = defaultdict(lambda: defaultdict(list))
        part_keys = client_parts_list[0].keys()

        # 收集所有参数
        for client_parts in client_parts_list:
            for part_name in part_keys:
                for key, param_tensor in client_parts[part_name].items():
                    collected_tensors[part_name][key].append(param_tensor)

        aggregated_parts = OrderedDict()
        for part_name, keys_dict in collected_tensors.items():
            aggregated_parts[part_name] = OrderedDict()
            for key, tensor_list in keys_dict.items():
                # stack shape: [num_clients, *param_shape]
                stacked_params = torch.stack(tensor_list)
                # 计算中位数 (dim=0 是客户端维度)
                # torch.median 返回 (values, indices)，我们只需要 values
                median_tensor = torch.median(stacked_params, dim=0).values
                aggregated_parts[part_name][key] = median_tensor.to(device)

        return aggregated_parts
