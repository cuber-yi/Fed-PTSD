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
from utils.reporting_utils import save_summary_report
from client import Client
from server import Server
from src.cluster.utils import save_cluster_visualization


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_extended_clustering_experiment(dataset_group_name, files, cluster_config, model_name, window_size, pre_len,
                                       base_config, parent_dir):
    config = copy.deepcopy(base_config)

    # --- 1. 固定基础策略 ---
    config['model']['name'] = model_name
    config['model']['pfl_enabled'] = True
    config['aggregation'] = {'name': 'fedavgm', 'beta': 0.9, 'server_lr': 1.0}
    config['privacy']['enabled'] = False

    # --- 2. 注入聚类配置 ---
    if cluster_config['method'] is None:
        config['clustering']['enabled'] = False
        cluster_name = "No_Clustering"
    else:
        config['clustering']['enabled'] = True
        config['clustering'].update(cluster_config)
        k_val = cluster_config.get('num_clusters', 'Auto')
        cluster_name = f"{cluster_config['method']}_K{k_val}"

    # --- 3. 配置数据模式 ---
    config['data']['mode'] = 'multi_file_all_sheets'
    config['data']['files'] = files
    config['data']['window_size'] = window_size
    config['data']['pre_len'] = pre_len

    if 'config' not in config['model']: config['model']['config'] = {}
    config['model']['config']['enc_in'] = config['data']['enc_in']
    config['model']['config']['pred_len'] = config['data']['pre_len']
    config['model']['config']['seq_len'] = config['data']['window_size']

    model_config_path = Path('config') / f"{model_name}.yaml"
    if model_config_path.exists():
        with open(model_config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
            if cfg and 'config' in cfg:
                config['model']['config'].update(cfg['config'])

    device = torch.device(config['data']['device'])

    exp_sub_name = f"{dataset_group_name}_{cluster_name}"
    exp_dir = os.path.join(parent_dir, exp_sub_name)
    exp_dir = os.path.normpath(exp_dir)

    print(f"\n{'=' * 80}")
    print(f" >>> 执行实验: {dataset_group_name} | {cluster_name}")
    print(f"{'=' * 80}")

    exp_dir = os.path.join(parent_dir, exp_sub_name)
    os.makedirs(os.path.join(exp_dir, 'results'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'plots'), exist_ok=True)

    # --- 5. 初始化 ---
    set_seed(config.get('seed', 42))
    g = torch.Generator()
    g.manual_seed(config.get('seed', 42))

    client_dataloaders = setup_clients_multi_file_by_sheet(
        file_paths=files,
        window_size=window_size,
        pre_len=pre_len,
        batch_size=config['federation']['batch_size'],
        max_capacity=config['data']['max_capacity'],
        generator=g
    )

    if not client_dataloaders: return None
    num_clients = len(client_dataloaders)
    clients = [Client(client_id=i, dataloader=dl, config=config, device=device)
               for i, dl in enumerate(client_dataloaders)]
    server = Server(config=config, num_total_clients=num_clients, device=device)

    # --- 6. 训练循环 ---
    num_rounds = config['federation']['num_rounds']

    WARMUP_ROUNDS = config['clustering'].get('warmup_rounds', 9)
    RECLUSTER_INTERVAL = config['clustering'].get('recluster_every_n_rounds', 100)

    last_round_client_parts = {}

    for comm_round in range(num_rounds):
        if config['clustering']['enabled']:
            if comm_round == 0:
                print("  [Cluster] Round 0: 收集初始参数用于可视化...")
                init_parts = server.get_global_model_parts(0)
                temp_parts = {}
                for c in clients:
                    c.set_global_model(copy.deepcopy(init_parts))
                    c.local_train()
                    temp_parts[c.client_id] = c.get_local_parameters()

                save_cluster_visualization(exp_dir, 0, temp_parts, server.client_clusters, f"{cluster_name}_Init")

            if comm_round < WARMUP_ROUNDS:
                if comm_round == 0:
                    print(f"  [Cluster] 进入 Warmup 阶段 ({WARMUP_ROUNDS} 轮). 暂停聚类，仅运行 FedAvg.")

            else:
                is_first_clustering = (comm_round == WARMUP_ROUNDS)
                is_interval_clustering = ((comm_round - WARMUP_ROUNDS) % RECLUSTER_INTERVAL == 0)

                if is_first_clustering or is_interval_clustering:
                    if last_round_client_parts:  # 确保有上一轮的参数用于聚类
                        print(f"  [Cluster] Round {comm_round}: 执行聚类 (Warmup Done).")
                        server.recluster_clients(last_round_client_parts)
                        print(f"    -> Groups: {server.client_clusters}")

                        suffix = "WarmupEnd" if is_first_clustering else ""
                        save_cluster_visualization(exp_dir, comm_round, last_round_client_parts, server.client_clusters,
                                                   f"{cluster_name}_{suffix}")
                    else:
                        print("  [Cluster] Warning: No client parts available for clustering.")

        client_parts_dict = {}
        client_losses_dict = {}

        for client in clients:
            global_parts = server.get_global_model_parts(client.client_id)
            client.set_global_model(copy.deepcopy(global_parts))

            # 本地训练
            loss = client.local_train()
            local_parts = client.get_local_parameters()

            client_parts_dict[client.client_id] = local_parts
            client_losses_dict[client.client_id] = loss

        last_round_client_parts = copy.deepcopy(client_parts_dict)

        server.aggregate_parameters(client_parts_dict, client_losses_dict)

        if (comm_round + 1) % 10 == 0:
            print(f"  Round {comm_round + 1}/{num_rounds} - Avg Loss: {np.mean(list(client_losses_dict.values())):.4f}")

    print("正在评估...")
    all_metrics = []
    for client in clients:
        final_parts = server.get_global_model_parts(client.client_id)
        client.set_global_model(copy.deepcopy(final_parts))
        mae, rmse = client.evaluate(save_dir=exp_dir)
        cluster_id = server.client_clusters.get(client.client_id, 0)
        all_metrics.append({'client_id': client.client_id, 'cluster_id': cluster_id, 'MAE': mae, 'RMSE': rmse})

    # >>> 绘图：Final Round (Round 50)
    if config['clustering']['enabled'] and last_round_client_parts:
        save_cluster_visualization(exp_dir, num_rounds, last_round_client_parts, server.client_clusters,
                                   f"{cluster_name}_Final")

    avg_mae = np.mean([m['MAE'] for m in all_metrics])
    avg_rmse = np.mean([m['RMSE'] for m in all_metrics])
    std_mae = np.std([m['MAE'] for m in all_metrics])
    std_rmse = np.std([m['RMSE'] for m in all_metrics])

    save_summary_report(exp_dir, all_metrics, {'MAE': avg_mae, 'RMSE': avg_rmse})
    print(f"实验完成: {cluster_name} -> MAE={avg_mae:.4f}")

    return {
        'Dataset_Group': dataset_group_name,
        'Cluster_Method': cluster_name,
        'MAE': avg_mae,
        'RMSE': avg_rmse,
        'MAE_std': std_mae, 'RMSE_std': std_rmse
    }


def main():
    base_config = load_config('config/config.yaml')

    xjtu_plan = {
        'name': 'XJTU_Group',
        'files': ['data/batch-1.xlsx', 'data/batch-2.xlsx', 'data/batch-3.xlsx'],
        'win': 50, 'pre': 200
    }
    mit_plan = {
        'name': 'MIT_Group',
        'files': ['data/batch-4.xlsx', 'data/batch-5.xlsx'],
        'win': 100, 'pre': 500
    }

    experiment_plans = [xjtu_plan, mit_plan]
    target_model = 'xpatch'

    timestamp = datetime.datetime.now().strftime("%m%d-%H%M")
    parent_dir = os.path.join(base_config['results']['save_dir_prefix'], f"exp_4_{timestamp}")
    parent_dir = os.path.normpath(parent_dir)
    os.makedirs(parent_dir, exist_ok=True)

    summary_results = []
    print(f"开始执行 Exp 4 (Extended Clustering)，结果保存在: {parent_dir}")

    for plan in experiment_plans:
        valid_files = [f for f in plan['files'] if os.path.exists(f)]
        if not valid_files: continue

        current_strategies = []
        # current_strategies.append({'method': None, 'num_clusters': 1})

        # 灵活调整聚类数量
        if 'XJTU' in plan['name']:
            # cluster_nums = [3, 4, 5]  # XJTU
            cluster_nums = [4]
        else:
            cluster_nums = [2, 3]  # MIT

        # for k in cluster_nums:
        #     current_strategies.append({'method': 'kmeans', 'num_clusters': k})
        #     current_strategies.append({'method': 'gmm', 'num_clusters': k})
        #     current_strategies.append({'method': 'spectral', 'num_clusters': k})

        # current_strategies.append({'method': 'leiden', 'num_clusters': 'Auto', 'n_neighbors': 5})
        # current_strategies.append({'method': 'finch', 'num_clusters': 'Auto', 'finch_level': -1})
        current_strategies.append({'method': 'xleiden', 'num_clusters': 'Auto',
            'n_neighbors': 5, 'trend_weight': 0.8, 'seasonal_weight': 0.2})
        current_strategies.append({'method': 'xfinch', 'num_clusters': 'Auto',
            'finch_level': -1, 'trend_weight': 0.8, 'seasonal_weight': 0.2})

        for clus_conf in current_strategies:
            try:
                res = run_extended_clustering_experiment(
                    dataset_group_name=plan['name'],
                    files=valid_files,
                    cluster_config=clus_conf,
                    model_name=target_model,
                    window_size=plan['win'],
                    pre_len=plan['pre'],
                    base_config=base_config,
                    parent_dir=parent_dir
                )
                if res: summary_results.append(res)
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()

    if summary_results:
        df = pd.DataFrame(summary_results)
        print("\n=== Exp 4 结果汇总 ===")
        print(df.to_string(index=False))
        df.to_csv(os.path.join(parent_dir, 'exp_4_summary.csv'), index=False)


if __name__ == '__main__':
    main()
