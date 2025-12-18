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
from utils.data_loader import setup_clients_multi_file_by_sheet
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


def run_privacy_experiment(dataset_group_name, files, cluster_config, privacy_setting, base_config, parent_dir):
    config = copy.deepcopy(base_config)

    # --- 1. 基础模型与聚合配置 ---
    config['model']['name'] = 'xpatch'
    config['model']['pfl_enabled'] = True
    config['aggregation'] = {'name': 'fedavgm', 'beta': 0.9, 'server_lr': 1.0}

    # --- 2. 注入特定的聚类配置 ---
    config['clustering']['enabled'] = True
    config['clustering']['method'] = cluster_config['method']
    config['clustering']['num_clusters'] = cluster_config['num_clusters']

    # 获取聚类控制参数
    WARMUP_ROUNDS = config['clustering'].get('warmup_rounds', 9)
    RECLUSTER_INTERVAL = config['clustering'].get('recluster_every_n_rounds', 100)

    # --- 3. 隐私设置 ---
    noise_level_name = privacy_setting['name']
    if privacy_setting['enabled']:
        config['privacy']['enabled'] = True
        config['privacy']['clipping_norm'] = 1.5
        config['privacy']['noise_sigma'] = privacy_setting['sigma_dict']
    else:
        config['privacy']['enabled'] = False

    # --- 4. 数据配置 ---
    config['data']['mode'] = 'multi_file_all_sheets'
    config['data']['files'] = files

    # 自动判断窗口大小
    if 'MIT' in dataset_group_name:
        config['data']['window_size'] = 100
        config['data']['pre_len'] = 500
    else:
        # XJTU
        config['data']['window_size'] = 50
        config['data']['pre_len'] = 200

    if 'config' not in config['model']: config['model']['config'] = {}
    config['model']['config']['enc_in'] = config['data']['enc_in']
    config['model']['config']['pred_len'] = config['data']['pre_len']
    config['model']['config']['seq_len'] = config['data']['window_size']

    model_config_path = Path('config') / "xpatch.yaml"
    if model_config_path.exists():
        with open(model_config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
            if cfg and 'config' in cfg:
                config['model']['config'].update(cfg['config'])

    device = torch.device(config['data']['device'])

    # 生成实验名
    clus_tag = f"{config['clustering']['method']}_K{config['clustering']['num_clusters']}"
    exp_sub_name = f"{dataset_group_name}_{clus_tag}_{noise_level_name}"

    # 路径标准化
    exp_dir = os.path.join(parent_dir, exp_sub_name)
    exp_dir = os.path.normpath(exp_dir)

    print(f"\n{'=' * 80}")
    print(f" >>> 执行隐私实验: {dataset_group_name} | {noise_level_name}")
    print(f" >>> 聚类策略: {clus_tag} (Warmup={WARMUP_ROUNDS})")
    print(f" >>> 隐私噪声: {privacy_setting.get('sigma_dict', 'None')}")
    print(f"{'=' * 80}")

    os.makedirs(os.path.join(exp_dir, 'results'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'models'), exist_ok=True)
    # 不再创建 plots 文件夹

    set_seed(config.get('seed', 42))
    g = torch.Generator()
    g.manual_seed(config.get('seed', 42))

    client_dataloaders = setup_clients_multi_file_by_sheet(
        file_paths=files,
        window_size=config['data']['window_size'],
        pre_len=config['data']['pre_len'],
        batch_size=config['federation']['batch_size'],
        max_capacity=config['data']['max_capacity'],
        generator=g
    )

    if not client_dataloaders: return None

    num_clients = len(client_dataloaders)
    clients = [Client(client_id=i, dataloader=dl, config=config, device=device)
               for i, dl in enumerate(client_dataloaders)]
    server = Server(config=config, num_total_clients=num_clients, device=device)

    num_rounds = config['federation']['num_rounds']
    last_round_client_parts = {}

    for comm_round in range(num_rounds):
        # --- 聚类控制逻辑 (无绘图) ---
        if config['clustering']['enabled']:
            # Round 0 初始聚类 (仅执行逻辑，不保存图)
            if comm_round == 0:
                init_parts = server.get_global_model_parts(0)
                temp_parts = {}
                for c in clients:
                    c.set_global_model(copy.deepcopy(init_parts))
                    c.local_train()
                    temp_parts[c.client_id] = c.get_local_parameters()

                try:
                    server.recluster_clients(temp_parts)
                except Exception as e:
                    print(f"  [Warning] 初始聚类失败: {e}")

            if comm_round < WARMUP_ROUNDS:
                pass
            else:
                is_first = (comm_round == WARMUP_ROUNDS)
                is_interval = ((comm_round - WARMUP_ROUNDS) % RECLUSTER_INTERVAL == 0)

                if (is_first or is_interval) and last_round_client_parts:
                    print(f"  [Cluster] Round {comm_round}: Re-clustering...")
                    try:
                        server.recluster_clients(last_round_client_parts)
                    except Exception as e:
                        print(f"  [Error] 聚类失败 (可能由于隐私噪声): {e}")

        client_parts_dict = {}
        client_losses_dict = {}

        for client in clients:
            global_parts = server.get_global_model_parts(client.client_id)
            client.set_global_model(copy.deepcopy(global_parts))
            loss = client.local_train()

            # 获取的参数已包含隐私噪声(如果启用)
            local_parts = client.get_local_parameters()

            client_parts_dict[client.client_id] = local_parts
            client_losses_dict[client.client_id] = loss

        last_round_client_parts = copy.deepcopy(client_parts_dict)
        server.aggregate_parameters(client_parts_dict, client_losses_dict)

        if (comm_round + 1) % 10 == 0:
            print(f"  Round {comm_round + 1}/{num_rounds} Avg Loss: {np.mean(list(client_losses_dict.values())):.4f}")

    print("正在评估...")
    all_metrics = []
    for client in clients:
        final_parts = server.get_global_model_parts(client.client_id)
        client.set_global_model(copy.deepcopy(final_parts))
        mae, rmse = client.evaluate(save_dir=os.path.normpath(exp_dir))
        cluster_id = server.client_clusters.get(client.client_id, 0)
        all_metrics.append({'client_id': client.client_id, 'cluster_id': cluster_id, 'MAE': mae, 'RMSE': rmse})

    avg_mae = np.mean([m['MAE'] for m in all_metrics])
    avg_rmse = np.mean([m['RMSE'] for m in all_metrics])

    sigma_val = 0.0
    if privacy_setting['enabled']:
        sigma_val = privacy_setting['sigma_dict']['trend']

    return {
        'Dataset': dataset_group_name,
        'Cluster_Method': clus_tag,
        'Privacy_Mode': noise_level_name,
        'Sigma_Trend': sigma_val,
        'MAE': avg_mae,
        'RMSE': avg_rmse
    }


def main():
    base_config = load_config('config/config.yaml')

    experiment_plans = [
        {
            'name': 'XJTU_Group',
            'files': ['data/batch-1.xlsx', 'data/batch-2.xlsx', 'data/batch-3.xlsx'],
            'cluster_config': {'method': 'spectral', 'num_clusters': 4}
        },
        # {
        #     'name': 'MIT_Group',
        #     'files': ['data/batch-4.xlsx', 'data/batch-5.xlsx'],
        #     'cluster_config': {'method': 'spectral', 'num_clusters': 3}
        # }
    ]

    # 定义隐私强度梯度
    privacy_levels = [
        # 1. 基准：无隐私噪声
        {'name': 'No_Noise', 'enabled': False, 'sigma_dict': None},
        #
        # # 2. 轻微噪声
        # {'name': 'Low_Noise', 'enabled': True, 'sigma_dict': {'seasonal': 0.001, 'trend': 0.005}},
        #
        # # 3. 中等噪声
        # {'name': 'Medium_Noise', 'enabled': True, 'sigma_dict': {'seasonal': 0.005, 'trend': 0.02}},
        #
        # # 4. 较高噪声
        # {'name': 'High_Noise', 'enabled': True, 'sigma_dict': {'seasonal': 0.01, 'trend': 0.05}},
        #
        # # 5. 强噪声
        # {'name': 'VeryHigh_Noise', 'enabled': True, 'sigma_dict': {'seasonal': 0.02, 'trend': 0.1}},
    ]

    timestamp = datetime.datetime.now().strftime("%m%d-%H%M")
    parent_dir = os.path.join(base_config['results']['save_dir_prefix'], f"exp_5_{timestamp}")
    parent_dir = os.path.normpath(parent_dir)
    os.makedirs(parent_dir, exist_ok=True)

    print(f"开始隐私权衡实验 (Exp 5 - No Plot)，结果保存至: {parent_dir}")
    print(f"计划: XJTU(Spectral-K4) & MIT(Spectral-K3)")

    results = []

    for plan in experiment_plans:
        valid_files = [f for f in plan['files'] if os.path.exists(f)]
        if not valid_files: continue

        for setting in privacy_levels:
            try:
                res = run_privacy_experiment(
                    dataset_group_name=plan['name'],
                    files=valid_files,
                    cluster_config=plan['cluster_config'],
                    privacy_setting=setting,
                    base_config=base_config,
                    parent_dir=parent_dir
                )
                if res: results.append(res)
            except Exception as e:
                print(f"实验出错 ({plan['name']} - {setting['name']}): {e}")
                import traceback
                traceback.print_exc()

    if results:
        df = pd.DataFrame(results)
        print("\n=== Exp 5 实验结果汇总 ===")
        print(df.to_string(index=False))
        csv_path = os.path.join(parent_dir, 'exp_5_summary.csv')
        df.to_csv(csv_path, index=False)
        print(f"汇总数据已保存: {csv_path}")


if __name__ == '__main__':
    main()
