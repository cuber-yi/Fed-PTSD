import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import OrderedDict
import numpy as np
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error
from utils.model_utils import get_model_class
from utils.reporting_utils import save_client_results


def _classify_xpatch_param(key: str) -> str:
    """
    根据xPatch模型结构，将参数键分为三类：
    1. 'seasonal': 季节性流 (Network.py中的非线性流)
    2. 'trend': 趋势流 (Network.py中的线性流)
    3. 'personal': 个性化头 (Network.py中的fc8，RevIN 和 Decomp 模块)
    """
    # 个性化头
    if key.startswith('net.fc8') or \
            key.startswith('revin_layer') or \
            key.startswith('decomp'):
        return 'personal'

    # 季节性流 (Non-linear Stream in network.py)
    if key.startswith('net.fc1') or \
            key.startswith('net.bn1') or \
            key.startswith('net.conv1') or \
            key.startswith('net.bn2') or \
            key.startswith('net.fc2') or \
            key.startswith('net.conv2') or \
            key.startswith('net.bn3') or \
            key.startswith('net.fc3') or \
            key.startswith('net.fc4'):
        return 'seasonal'

    # 趋势流 (Linear Stream in network.py)
    if key.startswith('net.fc5') or \
            key.startswith('net.ln1') or \
            key.startswith('net.fc6') or \
            key.startswith('net.ln2') or \
            key.startswith('net.fc7'):
        return 'trend'

    return 'common'


class Client:
    def __init__(self, client_id: int, dataloader: DataLoader, config: dict, device: torch.device):
        self.client_id = client_id
        self.dataloader = dataloader
        self.config = config
        self.device = device

        # --- 使用模型工厂动态创建本地模型 ---
        self.model_name = self.config['model']['name']
        model_params = self.config['model']['config']

        ModelClass = get_model_class(self.model_name)
        self.model = ModelClass(**model_params).to(self.device)

        self.dp_enabled = self.config.get('privacy', {}).get('enabled', False)
        if self.dp_enabled:
            self.dp_clipping_norm = self.config['privacy']['clipping_norm']
            self.dp_noise_sigma = self.config['privacy']['noise_sigma']
            print(f"[Client {client_id}] 差分隐私已启用. Clip={self.dp_clipping_norm}, Sigma={self.dp_noise_sigma}")

    def set_global_model(self, global_parts: dict):
        """
        从服务器接收拆分后的全局模型。
        """
        is_xpatch_pFL = self.model_name.lower() == 'xpatch'

        if is_xpatch_pFL:
            local_state_dict = self.model.state_dict()
            local_state_dict.update(global_parts['seasonal'])
            local_state_dict.update(global_parts['trend'])
            # 加载合并后的状态
            self.model.load_state_dict(local_state_dict)

        else:
            # --- 加载完整模型 ---
            self.model.load_state_dict(global_parts['full_model'])

    def local_train(self):
        """执行本地训练 (同时训练共享层和个性化层)"""
        self.model.train()
        local_epochs = self.config['federation']['local_epochs']
        lr = self.config['training']['lr']
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        for epoch in range(local_epochs):
            for x_batch, y_batch in self.dataloader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(x_batch)
                loss = criterion(outputs, y_batch.squeeze(-1))
                loss.backward()
                if self.dp_enabled:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.dp_clipping_norm)
                optimizer.step()

    def _add_noise_to_part(self, part_dict: OrderedDict, sigma: float) -> OrderedDict:
        """为参数字典中的每个张量添加高斯噪声"""
        if sigma == 0:
            return part_dict

        noisy_part = OrderedDict()
        for key, param in part_dict.items():
            noise = torch.randn_like(param) * sigma
            noisy_part[key] = param + noise
        return noisy_part

    def get_local_parameters(self) -> dict:
        """
        返回本地训练后的模型参数，拆分为多个部分。
        """
        is_xpatch_pFL = self.model_name.lower() == 'xpatch'

        if is_xpatch_pFL:
            parts = {
                'common': OrderedDict(),
                'seasonal': OrderedDict(),
                'trend': OrderedDict(),
                'personal': OrderedDict()
            }

            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    part_name = _classify_xpatch_param(name)
                    parts[part_name][name] = param.data.clone()

            if self.dp_enabled:
                noisy_seasonal = self._add_noise_to_part(parts['seasonal'], self.dp_noise_sigma['seasonal'])
                noisy_trend = self._add_noise_to_part(parts['trend'], self.dp_noise_sigma['trend'])

                # 只返回添加噪声后的共享部分
                return {'seasonal': noisy_seasonal, 'trend': noisy_trend}

            # 只返回共享部分
            return {'seasonal': parts['seasonal'], 'trend': parts['trend']}

        else:
            # --- 返回完整模型 ---
            full_params = OrderedDict()
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    full_params[name] = param.data.clone()
            return {'full_model': full_params}

    def evaluate(self, save_dir: str):
        # --- 1. 保存最终模型 ---
        model_path = os.path.join(save_dir, 'models', f'client_{self.client_id}_final_model.pth')
        torch.save(self.model.state_dict(), model_path)

        # --- 2. 评估模型性能 ---
        self.model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for x_batch, y_batch in self.dataloader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                outputs = self.model(x_batch)
                all_preds.append(outputs.cpu().numpy())
                all_true.append(y_batch.squeeze(-1).cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_true = np.concatenate(all_true)

        mae = mean_absolute_error(all_true, all_preds)
        rmse = np.sqrt(mean_squared_error(all_true, all_preds))

        # --- 3. 获取样本用于绘图 ---
        x_sample, y_sample = next(iter(self.dataloader))
        x_sample, y_sample = x_sample[0:1].to(self.device), y_sample[0:1]
        with torch.no_grad():
            y_pred_sample = self.model(x_sample).cpu().numpy().flatten()
        y_true_sample = y_sample.numpy().flatten()

        # --- 4. 调用外部函数保存结果和绘图 ---
        metrics = {'MAE': mae, 'RMSE': rmse}
        save_client_results(
            save_dir=save_dir,
            client_id=self.client_id,
            metrics=metrics,
            y_true=y_true_sample,
            y_pred=y_pred_sample
        )

        return mae, rmse
