import os
import torch
import numpy as np
import random
import copy
import datetime
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from pathlib import Path
from utils.config_utils import load_config
from utils.data_loader import setup_clients_multi_file_by_sheet
from client import Client
from server import Server
from src.cluster.utils import vectorize_client_params


plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial']


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_detailed_source_labels(file_paths):
    """
    辅助函数：生成更详细的标签（用于可视化）
    返回: list of strings, e.g., ['batch-1', 'batch-1', ..., 'batch-2']
    """
    labels = []
    for fp in file_paths:
        try:
            # 提取文件名作为标签 (例如 'batch-1')
            file_name = os.path.basename(fp).split('.')[0]
            xls = pd.ExcelFile(fp)
            sheet_names = xls.sheet_names
            labels.extend([file_name] * len(sheet_names))
        except:
            pass
    return labels


def visualize_clustering(server, client_parts_dict, round_num, save_dir, client_source_labels, cluster_on):
    """
    执行 t-SNE 可视化
    """
    client_ids, X = vectorize_client_params(client_parts_dict, cluster_on=cluster_on)

    if len(client_ids) < 2: return

    # t-SNE 降维
    perp = min(30, len(client_ids) - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perp, init='pca', learning_rate='auto')
    X_embedded = tsne.fit_transform(X)

    # 获取聚类结果
    cluster_assignments = [server.client_clusters.get(cid, 0) for cid in client_ids]
    sources = [client_source_labels[cid] for cid in client_ids]

    plt.figure(figsize=(10, 8))
    unique_sources = sorted(list(set(sources)))
    # 使用 tab10 或 tab20 颜色板
    source_to_color = {s: plt.cm.tab10(i % 10) for i, s in enumerate(unique_sources)}

    for i, cid in enumerate(client_ids):
        x, y = X_embedded[i]
        s = sources[i]
        c_id = cluster_assignments[i]

        # 标记: 数字表示被算法分到了哪个 Cluster
        marker = f"${c_id}$"

        plt.scatter(x, y, c=[source_to_color[s]], s=180, marker=marker, alpha=0.8,
                    label=s if s not in plt.gca().get_legend_handles_labels()[1] else "")

    plt.title(f"Round {round_num} | Cluster On: {cluster_on.upper()}\nColor=Batch File, Number=Cluster ID")
    plt.legend(title="Batch Source", loc='best')
    plt.grid(True, alpha=0.3)

    plt.savefig(os.path.join(save_dir, f"tsne_round_{round_num:03d}.png"), dpi=150)
    plt.close()


def run_group_experiment(group_name, files, win, pre, n_clusters, base_config, parent_dir):
    """
    运行一组数据的实验
    :param n_clusters: 该组数据指定的聚类数量
    """
    # 1. 准备标签
    source_labels = get_detailed_source_labels(files)

    # 2. 定义要对比的策略
    strategies = [
        {'name': 'No_Clustering', 'clustering': False, 'cluster_on': 'trend'},
        {'name': 'Cluster_Trend', 'clustering': True, 'cluster_on': 'trend'},
    ]

    results = []
    print(f"\n{'#' * 60}")
    print(f" >>> Group Experiment: {group_name}")
    print(f" >>> Files: {files}")
    print(f" >>> Param: Win={win}, Pred={pre}, NumClusters={n_clusters}")
    print(f"{'#' * 60}")

    for strat in strategies:
        exp_name = f"{group_name}_{strat['name']}"

        # --- 配置构建 ---
        config = copy.deepcopy(base_config)
        config['model']['name'] = 'xpatch'
        config['model']['config'] = {}

        # 加载 xPatch 参数
        model_config_path = Path('config/xpatch.yaml')
        if model_config_path.exists():
            with open(model_config_path, 'r', encoding='utf-8') as f:
                model_cfg = yaml.safe_load(f)
                if 'config' in model_cfg:
                    config['model']['config'].update(model_cfg['config'])
                else:
                    config['model']['config'].update(model_cfg)

        # 策略参数
        config['clustering']['enabled'] = strat['clustering']
        if strat['clustering']:
            config['clustering']['cluster_on'] = strat['cluster_on']
            # 【修改点】使用传入的 n_clusters
            config['clustering']['num_clusters'] = n_clusters
            config['clustering']['method'] = 'kmeans'
            config['clustering']['recluster_every_n_rounds'] = 5

        # 数据参数
        config['data']['mode'] = 'multi_file_all_sheets'
        config['data']['files'] = files
        config['data']['window_size'] = win
        config['data']['pre_len'] = pre
        config['federation']['num_rounds'] = 50
        config['privacy']['enabled'] = False

        # 维度注入
        config['model']['config']['enc_in'] = config['data']['enc_in']
        config['model']['config']['pred_len'] = pre
        config['model']['config']['seq_len'] = win

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # --- 目录准备 ---
        exp_dir = os.path.join(parent_dir, exp_name)
        viz_dir = os.path.join(exp_dir, 'viz')
        os.makedirs(os.path.join(exp_dir, 'results'), exist_ok=True)
        os.makedirs(os.path.join(exp_dir, 'models'), exist_ok=True)
        os.makedirs(viz_dir, exist_ok=True)

        print(f"   Running Strategy: {strat['name']} (K={n_clusters if strat['clustering'] else 'N/A'})...")

        # --- 初始化 ---
        set_seed(42)
        g = torch.Generator()
        g.manual_seed(42)

        try:
            client_dataloaders = setup_clients_multi_file_by_sheet(
                file_paths=files, window_size=win, pre_len=pre,
                batch_size=32, max_capacity=2.0, generator=g
            )
        except Exception as e:
            print(f"   Skipping {group_name} due to data error: {e}")
            continue

        if not client_dataloaders: continue

        num_clients = len(client_dataloaders)
        clients = [Client(i, dl, config, device) for i, dl in enumerate(client_dataloaders)]
        server = Server(config, num_clients, device)

        current_labels = source_labels[:num_clients]

        # --- 训练 ---
        for comm_round in range(config['federation']['num_rounds']):
            if comm_round == 0 and config['clustering']['enabled']:
                init_parts = server.get_global_model_parts(0)
                tmp_parts = {}
                for c in clients:
                    c.set_global_model(copy.deepcopy(init_parts))
                    c.local_train()
                    tmp_parts[c.client_id] = c.get_local_parameters()
                server.recluster_clients(tmp_parts)

            client_parts_dict = {}
            client_losses_dict = {}

            # 1. 本地训练
            for client in clients:
                global_parts = server.get_global_model_parts(client.client_id)
                client.set_global_model(copy.deepcopy(global_parts))
                loss = client.local_train()
                client_parts_dict[client.client_id] = client.get_local_parameters()
                client_losses_dict[client.client_id] = loss

            # 2. 可视化
            viz_rounds = [0, 5, 20, 49]
            if comm_round in viz_rounds and client_parts_dict:
                visualize_clustering(
                    server, client_parts_dict, comm_round, viz_dir, current_labels,
                    cluster_on=strat['cluster_on']
                )

            # 3. 聚合 & 重聚类
            server.aggregate_parameters(client_parts_dict, client_losses_dict)
            if config['clustering']['enabled']:
                server.recluster_clients(client_parts_dict)

            if (comm_round + 1) % 10 == 0:
                print(f"     Round {comm_round + 1} Loss: {np.mean(list(client_losses_dict.values())):.4f}")

        # --- 评估 ---
        all_metrics = []
        for client in clients:
            final_parts = server.get_global_model_parts(client.client_id)
            client.set_global_model(copy.deepcopy(final_parts))
            mae, rmse = client.evaluate(save_dir=exp_dir)
            all_metrics.append({
                'client_id': client.client_id,
                'source': current_labels[client.client_id],
                'cluster_id': server.client_clusters.get(client.client_id, 0),
                'MAE': mae, 'RMSE': rmse
            })

        avg_mae = np.mean([m['MAE'] for m in all_metrics])
        avg_rmse = np.mean([m['RMSE'] for m in all_metrics])

        results.append({'Group': group_name, 'Strategy': strat['name'], 'MAE': avg_mae, 'RMSE': avg_rmse})

        # 保存详细信息
        pd.DataFrame(all_metrics).to_csv(os.path.join(exp_dir, 'details.csv'), index=False)

    return results


def main():
    base_config = load_config('config/config.yaml')

    experiment_groups = [
        {
            'name': 'XJTU_Internal',
            'files': ['data/batch-1.xlsx', 'data/batch-2.xlsx', 'data/batch-3.xlsx'],
            'win': 50,
            'pre': 200,
            'n_clusters': 6
        },
        {
            'name': 'MIT_Internal',
            'files': ['data/batch-4.xlsx', 'data/batch-5.xlsx'],
            'win': 100,
            'pre': 500,
            'n_clusters': 3
        }
    ]

    timestamp = datetime.datetime.now().strftime("%m%d-%H%M")
    parent_dir = os.path.join(base_config['results']['save_dir_prefix'], f"exp_3_{timestamp}")
    os.makedirs(parent_dir, exist_ok=True)
    print(f"Results saved to: {parent_dir}")

    all_results = []
    for group in experiment_groups:
        valid_files = [f for f in group['files'] if os.path.exists(f)]
        if not valid_files:
            print(f"Skipping {group['name']}, no files found.")
            continue

        # 传入 group['n_clusters']
        group_res = run_group_experiment(
            group['name'], valid_files, group['win'], group['pre'],
            group['n_clusters'], base_config, parent_dir
        )
        all_results.extend(group_res)

    if all_results:
        df = pd.DataFrame(all_results)
        print("\n" + "=" * 60)
        print("Experiment 3 Summary (Separate Groups)")
        print("=" * 60)
        print(df.to_string(index=False))
        df.to_csv(os.path.join(parent_dir, 'exp_3_summary.csv'), index=False)


if __name__ == '__main__':
    main()
