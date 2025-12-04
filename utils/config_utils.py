import yaml
import torch
from pathlib import Path


def load_config(path='config/config.yaml'):
    config_path = Path(path)

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 获取模型名称并加载模型配置文件
    model_name = config.get('model', {}).get('name')
    if model_name:
        model_config_path = config_path.parent / f"{model_name}.yaml"

        if model_config_path.exists():
            with open(model_config_path, 'r', encoding='utf-8') as f:
                model_config = yaml.safe_load(f)
            # 合并配置: 将模型配置合并到主配置的 'model' 键下
            if model_config:
                config['model'].update(model_config)
        else:
            print(f"[Config] 提示: 未找到 {model_name}.yaml，将使用默认或代码注入的参数。")

    # 确保 config['model']['config'] 存在，以便后续注入参数
    if 'config' not in config['model'] or config['model']['config'] is None:
        config['model']['config'] = {}

    if config['data']['device'] == 'auto':
        config['data']['device'] = "cuda" if torch.cuda.is_available() else "cpu"

    # 将数据输入维度注入到模型配置中
    if 'config' in config['model']:
        config['model']['config']['enc_in'] = config['data']['enc_in']
        config['model']['config']['pred_len'] = config['data']['pre_len']
        config['model']['config']['seq_len'] = config['data']['window_size']

    return config
