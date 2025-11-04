import torch
from collections import OrderedDict
from utils.model_utils import get_model_class


class Server:
    def __init__(self, config: dict, device: torch.device):
        self.config = config
        self.device = device

        model_name = self.config['model']['name']
        model_params = self.config['model']['config']

        ModelClass = get_model_class(model_name)
        self.global_model = ModelClass(**model_params).to(self.device)
        print(f"服务端部署模型: {model_name}")

    def aggregate_parameters(self, client_parameters_list: list):
        """
        直接对客户端上传的完整模型参数进行平均。
        """

        # 初始化一个新的 state_dict 用于存放聚合后的参数
        aggregated_params = OrderedDict()

        # 获取参数的键
        sample_params = client_parameters_list[0]
        keys = sample_params.keys()

        for key in keys:
            # 收集所有客户端对当前参数的值，并计算平均值
            aggregated_tensor = torch.stack([params[key] for params in client_parameters_list]).mean(dim=0)
            aggregated_params[key] = aggregated_tensor.to(self.device)

        return aggregated_params

    def update_global_model(self, new_params: OrderedDict):
        """用聚合后的新参数直接更新全局模型"""
        if not new_params:
            return

        # 获取当前全局模型的完整状态字典
        current_state_dict = self.global_model.state_dict()
        # 将聚合后的可训练参数加载进去
        current_state_dict.update(new_params)
        # 加载更新后的状态字典
        self.global_model.load_state_dict(current_state_dict)
