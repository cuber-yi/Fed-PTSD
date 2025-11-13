import torch
from collections import OrderedDict, defaultdict
from utils.model_utils import get_model_class
from client import _classify_xpatch_param

from src.aggregation.fed_avg import FedAvg
from src.aggregation.fed_prox import FedProx


class Server:
    def __init__(self, config: dict, device: torch.device):
        self.config = config
        self.device = device

        self.model_name = self.config['model']['name']
        model_params = self.config['model']['config']

        ModelClass = get_model_class(self.model_name)
        self.global_model = ModelClass(**model_params).to(self.device)

        # 初始化聚合策略
        aggregation_name = config.get('aggregation', {}).get('name', 'fedavg').lower()
        self.aggregator = self._get_aggregator(aggregation_name)

    def _get_aggregator(self, name: str):
        """根据配置选择聚合策略"""
        aggregators = {
            'fedavg': FedAvg,
            'fedprox': FedProx,
        }

        if name not in aggregators:
            print(f"[Warning] 未知聚合策略 '{name}', 使用默认 FedAvg")
            name = 'fedavg'

        return aggregators[name](self.config)

    def get_global_model_parts(self) -> dict:
        """
        将当前的全局模型拆分为多个部分以供分发。
        """
        is_xpatch_pFL = self.model_name.lower() == 'xpatch'

        if is_xpatch_pFL:
            parts = {'seasonal': OrderedDict(), 'trend': OrderedDict()}

            # 只遍历共享参数
            for name, param in self.global_model.named_parameters():
                part_name = _classify_xpatch_param(name)
                if part_name != 'personal':
                    parts[part_name][name] = param.data.clone()

            return parts

        else:
            # 返回完整模型
            return {'full_model': self.global_model.state_dict()}

    def aggregate_parameters(self, client_parts_list: list, client_losses: list = None) -> dict:
        return self.aggregator.aggregate(client_parts_list, self.device)

    def update_global_model(self, aggregated_parts: dict):
        """用聚合后的新参数更新全局模型"""
        current_state_dict = self.global_model.state_dict()

        for part_name, params_dict in aggregated_parts.items():
            current_state_dict.update(params_dict)

        self.global_model.load_state_dict(current_state_dict)

    def get_aggregator_info(self):
        """获取聚合器信息（用于日志和调试）"""
        if hasattr(self.aggregator, 'get_weights_info'):
            return self.aggregator.get_weights_info()
        return None
