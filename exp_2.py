import os
import torch
import numpy as np
import random
import copy
import datetime
import yaml
import pandas as pd
from pathlib import Path
from utils.config_utils import load_config
from utils.data_loader import setup_clients_by_sheet
from utils.reporting_utils import save_summary_report
from client import Client
from server import Server


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_single_experiment(file_path, strategy_name, agg_method, pfl_enabled, window_size, pre_len, base_config,
                          parent_dir):
    """
    :param strategy_name: 用于显示的策略名称 (如 'Ours', 'FedProx')
    :param agg_method: 聚合器名称 ('fedavg', 'fedprox', 'fedavgm'...)
    :param pfl_enabled: 是否开启个性化 (True/False)
    """

    config = copy.deepcopy(base_config)

    # --- 配置构建 ---
    config['model']['name'] = 'xpatch'
    config['model']['config'] = {}  # 清空防止污染

    # 加载 xPatch 默认参数
    model_config_path = Path('config/xpatch.yaml')
    if model_config_path.exists():
        with open(model_config_path, 'r', encoding='utf-8') as f:
            model_cfg = yaml.safe_load(f)
            if 'config' in model_cfg:
                config['model']['config'].update(model_cfg['config'])
            else:
                config['model']['config'].update(model_cfg)

    # 关键对比参数设置
    config['clustering']['enabled'] = False  # 实验1.2 暂不开启聚类，专注于聚合策略对比
    config['model']['pfl_enabled'] = pfl_enabled
    config['aggregation']['name'] = agg_method

    # 为特定算法设置超参数
    if agg_method == 'fedprox':
        config['aggregation']['mu'] = 0.01
    elif agg_method == 'fedavgm':
        config['aggregation']['server_lr'] = 1.0
        config['aggregation']['beta'] = 0.9
    elif agg_method == 'fedadam':
        config['aggregation']['server_lr'] = 0.01
        config['aggregation']['beta1'] = 0.9
        config['aggregation']['beta2'] = 0.99

    # 基础设置
    config['privacy']['enabled'] = False
    config['federation']['num_rounds'] = 50
    config['data']['mode'] = 'single_file_multi_client'
    config['data']['single_file'] = file_path
    config['data']['window_size'] = window_size
    config['data']['pre_len'] = pre_len

    # 维度注入
    config['model']['config']['enc_in'] = config['data']['enc_in']
    config['model']['config']['pred_len'] = config['data']['pre_len']
    config['model']['config']['seq_len'] = config['data']['window_size']

    device_str = config['data']['device']
    device = torch.device(device_str if device_str != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu'))

    # --- 打印信息 ---
    file_name = os.path.basename(file_path)
    print(f"\n{'=' * 80}")
    print(f" >>> Dataset=[{file_name}] | Strategy=[{strategy_name}]")
    print(f" >>> Agg=[{agg_method}], PFL=[{pfl_enabled}], Window={window_size}")
    print(f"{'=' * 80}")

    exp_dir = os.path.join(parent_dir, f"{file_name}_{strategy_name}")
    os.makedirs(os.path.join(exp_dir, 'results'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'models'), exist_ok=True)

    # --- 初始化 ---
    seed = config.get('seed', 42)
    set_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)

    try:
        client_dataloaders = setup_clients_by_sheet(
            file_path=file_path,
            window_size=window_size,
            pre_len=pre_len,
            batch_size=config['federation']['batch_size'],
            max_capacity=config['data']['max_capacity'],
            generator=g
        )
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

    if not client_dataloaders: return None

    num_clients = len(client_dataloaders)
    clients = [Client(client_id=i, dataloader=dl, config=config, device=device) for i, dl in
               enumerate(client_dataloaders)]
    server = Server(config=config, num_total_clients=num_clients, device=device)

    # --- 训练 ---
    for comm_round in range(config['federation']['num_rounds']):
        client_parts_dict = {}
        client_losses_dict = {}

        for client in clients:
            global_parts = server.get_global_model_parts(client.client_id)
            client.set_global_model(copy.deepcopy(global_parts))

            loss = client.local_train()
            local_parts = client.get_local_parameters()

            client_parts_dict[client.client_id] = local_parts
            client_losses_dict[client.client_id] = loss

        server.aggregate_parameters(client_parts_dict, client_losses_dict)

        if (comm_round + 1) % 10 == 0:
            print(f"  Round {comm_round + 1} Loss: {loss:.4f}")

    # --- 评估 ---
    all_metrics = []
    for client in clients:
        final_parts = server.get_global_model_parts(client.client_id)
        client.set_global_model(copy.deepcopy(final_parts))
        mae, rmse = client.evaluate(save_dir=exp_dir)
        all_metrics.append({'client_id': client.client_id, 'MAE': mae, 'RMSE': rmse})

    avg_mae = np.mean([m['MAE'] for m in all_metrics])
    avg_rmse = np.mean([m['RMSE'] for m in all_metrics])

    return {'Dataset': file_name, 'Strategy': strategy_name, 'MAE': avg_mae, 'RMSE': avg_rmse}


def main():
    base_config = load_config('config/config.yaml')

    # 1. 定义数据计划
    files_plan = [
        {'path': 'data/batch-1.xlsx', 'win': 50, 'pre': 200},
        {'path': 'data/batch-2.xlsx', 'win': 50, 'pre': 200},
        {'path': 'data/batch-3.xlsx', 'win': 50, 'pre': 200},
        {'path': 'data/batch-4.xlsx', 'win': 100, 'pre': 500},
        {'path': 'data/batch-5.xlsx', 'win': 100, 'pre': 500},
    ]

    # 2. 定义对比策略
    strategies = [
        {'name': 'Ours (PFL)', 'agg': 'fedavg', 'pfl': True},
        {'name': 'FedAvg', 'agg': 'fedavg', 'pfl': False},
        {'name': 'FedProx', 'agg': 'fedprox', 'pfl': False},
        {'name': 'FedAvgM', 'agg': 'fedavgm', 'pfl': False},
        {'name': 'FedAdam', 'agg': 'fedadam', 'pfl': False},
    ]

    timestamp = datetime.datetime.now().strftime("%m%d-%H%M")
    parent_dir = os.path.join(base_config['results']['save_dir_prefix'], f"exp_2_{timestamp}")
    os.makedirs(parent_dir, exist_ok=True)

    print(f"Start Exp 1.2. Saving to: {parent_dir}")

    results = []
    for plan in files_plan:
        for strat in strategies:
            # 检查文件
            if not os.path.exists(plan['path']): continue

            res = run_single_experiment(
                file_path=plan['path'],
                strategy_name=strat['name'],
                agg_method=strat['agg'],
                pfl_enabled=strat['pfl'],
                window_size=plan['win'],
                pre_len=plan['pre'],
                base_config=base_config,
                parent_dir=parent_dir
            )
            if res: results.append(res)

    if results:
        df = pd.DataFrame(results)
        csv_path = os.path.join(parent_dir, 'exp_1_2_summary.csv')
        df.to_csv(csv_path, index=False)
        print("\n实验 1.2 汇总:")
        print(df.to_string(index=False))
        print(f"Saved to {csv_path}")


if __name__ == '__main__':
    main()