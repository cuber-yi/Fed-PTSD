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
    """
    运行扩展聚类实验：支持多文件全Sheet模式，支持更多聚类算法
    """
    config = copy.deepcopy(base_config)

    # --- 1. 固定最佳聚合策略 ---
    config['model']['name'] = model_name
    config['model']['pfl_enabled'] = True  # 始终开启 PFL
    config['aggregation'] = {'name': 'fedavgm', 'beta': 0.9, 'server_lr': 1.0}
    # 禁用隐私以控制变量
    config['privacy']['enabled'] = False

    # --- 2. 注入聚类配置 ---
    if cluster_config['method'] is None:
        config['clustering']['enabled'] = False
        cluster_name = "No_Clustering"
    else:
        config['clustering']['enabled'] = True
        config['clustering'].update(cluster_config)
        cluster_name = f"{cluster_config['method']}_K{cluster_config['num_clusters']}"

    # --- 3. 配置数据模式为 Multi-File All-Sheets ---
    config['data']['mode'] = 'multi_file_all_sheets'
    config['data']['files'] = files
    config['data']['window_size'] = window_size
    config['data']['pre_len'] = pre_len

    # 模型维度注入
    if 'config' not in config['model']: config['model']['config'] = {}
    config['model']['config']['enc_in'] = config['data']['enc_in']
    config['model']['config']['pred_len'] = config['data']['pre_len']
    config['model']['config']['seq_len'] = config['data']['window_size']

    # 加载模型yaml
    model_config_path = Path('config') / f"{model_name}.yaml"
    if model_config_path.exists():
        with open(model_config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
            if cfg:
                if 'config' in cfg:
                    config['model']['config'].update(cfg['config'])
                else:
                    config['model']['config'].update(cfg)

    device = torch.device(config['data']['device'])

    # --- 4. 实验准备 ---
    exp_sub_name = f"{dataset_group_name}_{cluster_name}"

    print(f"\n{'=' * 80}")
    print(f" >>> 执行扩展聚类实验: {dataset_group_name} | {cluster_name}")
    print(f" >>> 文件列表: {files}")
    print(f" >>> 窗口设置: Win={window_size}, Pred={pre_len}")
    print(f"{'=' * 80}")

    exp_dir = os.path.join(parent_dir, exp_sub_name)
    os.makedirs(os.path.join(exp_dir, 'results'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'models'), exist_ok=True)

    # --- 5. 初始化 ---
    set_seed(config.get('seed', 42))
    g = torch.Generator()
    g.manual_seed(config.get('seed', 42))

    print("正在加载所有文件的所有Sheet作为独立客户端...")
    client_dataloaders = setup_clients_multi_file_by_sheet(
        file_paths=files,
        window_size=window_size,
        pre_len=pre_len,
        batch_size=config['federation']['batch_size'],
        max_capacity=config['data']['max_capacity'],
        generator=g
    )

    if not client_dataloaders:
        print("Error: 未能加载任何客户端数据。")
        return None

    num_clients = len(client_dataloaders)
    print(f"成功创建 {num_clients} 个客户端。")

    clients = [Client(client_id=i, dataloader=dl, config=config, device=device)
               for i, dl in enumerate(client_dataloaders)]
    server = Server(config=config, num_total_clients=num_clients, device=device)

    # --- 6. 训练循环 (带聚类逻辑) ---
    num_rounds = config['federation']['num_rounds']
    RECLUSTER_INTERVAL = 5

    last_round_client_parts = {}

    for comm_round in range(num_rounds):
        # 触发聚类
        if config['clustering']['enabled'] and (comm_round == 0 or comm_round % RECLUSTER_INTERVAL == 0):
            if comm_round == 0:
                # 第0轮预训练
                print("  [Cluster] Round 0: Pre-training for clustering initialization...")
                init_parts = server.get_global_model_parts(0)
                temp_parts = {}
                for c in clients:
                    c.set_global_model(copy.deepcopy(init_parts))
                    c.local_train()
                    temp_parts[c.client_id] = c.get_local_parameters()
                server.recluster_clients(temp_parts)
                print(f"  [Cluster] Groups: {server.client_clusters}")
            elif last_round_client_parts:
                server.recluster_clients(last_round_client_parts)
                print(f"  [Cluster] Re-clustered at Round {comm_round}. Groups: {server.client_clusters}")

        client_parts_dict = {}
        client_losses_dict = {}

        for client in clients:
            global_parts = server.get_global_model_parts(client.client_id)
            client.set_global_model(copy.deepcopy(global_parts))

            loss = client.local_train()
            local_parts = client.get_local_parameters()

            client_parts_dict[client.client_id] = local_parts
            client_losses_dict[client.client_id] = loss

        last_round_client_parts = copy.deepcopy(client_parts_dict)
        server.aggregate_parameters(client_parts_dict, client_losses_dict)

        if (comm_round + 1) % 10 == 0:
            print(f"  Round {comm_round + 1}/{num_rounds} - Avg Loss: {np.mean(list(client_losses_dict.values())):.4f}")

    # --- 7. 最终评估 ---
    print("正在评估...")
    all_metrics = []

    for client in clients:
        final_parts = server.get_global_model_parts(client.client_id)
        client.set_global_model(copy.deepcopy(final_parts))
        mae, rmse = client.evaluate(save_dir=exp_dir)

        cluster_id = server.client_clusters.get(client.client_id, 0)
        all_metrics.append({
            'client_id': client.client_id,
            'cluster_id': cluster_id,
            'MAE': mae,
            'RMSE': rmse
        })

    avg_mae = np.mean([m['MAE'] for m in all_metrics])
    avg_rmse = np.mean([m['RMSE'] for m in all_metrics])

    save_summary_report(exp_dir, all_metrics, {'MAE': avg_mae, 'RMSE': avg_rmse})
    print(f"实验完成: {dataset_group_name} - {cluster_name} -> MAE={avg_mae:.4f}")

    return {
        'Dataset_Group': dataset_group_name,
        'Cluster_Method': cluster_name,
        'Num_Clients': num_clients,
        'MAE': avg_mae,
        'RMSE': avg_rmse
    }


def main():
    base_config = load_config('config/config.yaml')

    # --- 1. 定义数据集分组计划 ---
    xjtu_plan = {
        'name': 'XJTU_Group',
        'files': ['data/batch-1.xlsx', 'data/batch-2.xlsx', 'data/batch-3.xlsx'],
        'win': 50,
        'pre': 200
    }
    mit_plan = {
        'name': 'MIT_Group',
        'files': ['data/batch-4.xlsx', 'data/batch-5.xlsx'],
        'win': 100,
        'pre': 500
    }

    # 将需要运行的组放入列表
    experiment_plans = [xjtu_plan, mit_plan]

    # --- 2. 定义聚类策略列表 ---
    cluster_strategies = [
        # 1. 基准：无聚类
        {'method': None, 'num_clusters': 1},

        # 2. K-Means
        {'method': 'kmeans', 'num_clusters': 3},
        {'method': 'kmeans', 'num_clusters': 5},

        # 3. GMM (软聚类)
        {'method': 'gmm', 'num_clusters': 3},

        # 4. Spectral (谱聚类)
        {'method': 'spectral', 'num_clusters': 3},
        {'method': 'spectral', 'num_clusters': 5},
    ]

    target_model = 'xpatch'

    timestamp = datetime.datetime.now().strftime("%m%d-%H%M")
    parent_dir = os.path.join(base_config['results']['save_dir_prefix'], f"exp_3_{timestamp}")
    os.makedirs(parent_dir, exist_ok=True)

    summary_results = []

    print(f"开始执行扩展聚类实验，结果保存在: {parent_dir}")

    for plan in experiment_plans:
        # 检查文件是否存在
        valid_files = [f for f in plan['files'] if os.path.exists(f)]
        if not valid_files:
            print(f"跳过组 {plan['name']}: 找不到任何文件。")
            continue

        for clus_conf in cluster_strategies:
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
                print(f"Error in {plan['name']} - {clus_conf['method']}: {e}")
                import traceback
                traceback.print_exc()

    if summary_results:
        df = pd.DataFrame(summary_results)
        print("\n=== 扩展聚类实验结果汇总 ===")
        print(df.to_string(index=False))
        df.to_csv(os.path.join(parent_dir, 'cluster_extended_summary.csv'), index=False)


if __name__ == '__main__':
    main()
