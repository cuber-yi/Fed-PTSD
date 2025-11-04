import yaml
import torch
from pathlib import Path


def load_config(path='config/config.yaml'):
    config_path = Path(path)
    # 1. 加载主配置文件
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 2. 获取模型名称并加载模型配置文件
    model_name = config.get('model', {}).get('name')
    model_config_path = config_path.parent / f"{model_name}.yaml"

    with open(model_config_path, 'r', encoding='utf-8') as f:
        model_config = yaml.safe_load(f)

    # 3. 合并配置: 将模型配置合并到主配置的 'model' 键下
    config['model'].update(model_config)

    # 4. 后处理
    if config['data']['device'] == 'auto':
        config['data']['device'] = "cuda" if torch.cuda.is_available() else "cpu"

    # 将数据输入维度注入到模型配置中
    if 'config' in config['model'] and config['model']['config'] is not None:
        config['model']['config']['enc_in'] = config['data']['enc_in']
        config['model']['config']['pred_len'] = config['data']['pre_len']
        config['model']['config']['seq_len'] = config['data']['window_size']

    return config

