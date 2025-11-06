import torch
from collections import OrderedDict, defaultdict
from utils.model_utils import get_model_class
from client import _classify_xpatch_param


class Server:
    def __init__(self, config: dict, device: torch.device):
        self.config = config
        self.device = device

        self.model_name = self.config['model']['name']
        model_params = self.config['model']['config']

        ModelClass = get_model_class(self.model_name)
        self.global_model = ModelClass(**model_params).to(self.device)

    def get_global_model_parts(self) -> dict:
        """
        将当前的全局模型拆分为多个部分以供分发。
        """
        is_xpatch_pFL = self.model_name.lower() == 'xpatch'

        if is_xpatch_pFL:
            parts = {'common': OrderedDict(), 'seasonal': OrderedDict(), 'trend': OrderedDict()}

            # 只遍历共享参数
            for name, param in self.global_model.named_parameters():
                part_name = _classify_xpatch_param(name)
                if part_name != 'personal':
                    parts[part_name][name] = param.data.clone()

            return parts

        else:
            # 返回完整模型
            return {'full_model': self.global_model.state_dict()}

    def aggregate_parameters(self, client_parts_list: list) -> dict:
        # 收集所有参数
        collected_tensors = defaultdict(lambda: defaultdict(list))

        # 获取所有参数部分的键
        if not client_parts_list:
            return {}
        part_keys = client_parts_list[0].keys()

        for client_parts in client_parts_list:
            for part_name in part_keys:
                for key, param_tensor in client_parts[part_name].items():
                    collected_tensors[part_name][key].append(param_tensor)

        # 分别聚合
        aggregated_parts = OrderedDict()
        for part_name, keys_dict in collected_tensors.items():
            aggregated_parts[part_name] = OrderedDict()
            for key, tensor_list in keys_dict.items():
                # 堆叠并沿 dim=0 (客户端维度) 求平均
                aggregated_tensor = torch.stack(tensor_list).mean(dim=0)
                aggregated_parts[part_name][key] = aggregated_tensor.to(self.device)

        return aggregated_parts

    def update_global_model(self, aggregated_parts: dict):
        """用聚合后的新参数更新全局模型"""
        current_state_dict = self.global_model.state_dict()

        for part_name, params_dict in aggregated_parts.items():
            current_state_dict.update(params_dict)

        self.global_model.load_state_dict(current_state_dict)
