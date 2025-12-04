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
    """
    labels = []
    for fp in file_paths:
        try:
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

    perp = min(30, len(client_ids) - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perp, init='pca', learning_rate='auto')
    try:
        X_embedded = tsne.fit_transform(X)
    except Exception as e:
        print(f"[Viz Error] t-SNE failed: {e}")
        return

    cluster_assignments = [server.client_clusters.get(cid, 0) for cid in client_ids]
    sources = [client_source_labels[cid] for cid in client_ids]

    plt.figure(figsize=(10, 8))
    unique_sources = sorted(list(set(sources)))
    source_to_color = {s: plt.cm.tab10(i % 10) for i, s in enumerate(unique_sources)}

    for i, cid in enumerate(client_ids):
        x, y = X_embedded[i]
        s = sources[i]
        c_id = cluster_assignments[i]
        marker = f"${c_id}$"
        plt.scatter(x, y, c=[source_to_color[s]], s=180, marker=marker, alpha=0.8,
                    label=s if s not in plt.gca().get_legend_handles_labels()[1] else "")

    plt.title(f"Round {round_num} | Cluster On: {cluster_on.upper()}\nColor=Source, Number=Cluster ID")
    plt.legend(title="Batch Source", loc='best')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, f"tsne_round_{round_num:03d}.png"), dpi=150)
    plt.close()


def run_group_experiment(group_name, files, win, pre, n_clusters, base_config, parent_dir):
    """
    运行一组数据的实验，对比多种聚类策略
    """
    source_labels = get_detailed_source_labels(files)

    strategies = [
        # {'name': 'No_Clustering', 'clustering': False, 'cluster_on': 'both'},
        # {'name': 'KMeans', 'clustering': True, 'method': 'kmeans', 'cluster_on': 'both'},
        # {'name': 'DBSCAN', 'clustering': True, 'method': 'dbscan', 'cluster_on': 'both'},
        {'name': 'GMM', 'clustering': True, 'method': 'gmm', 'cluster_on': 'both'},
        {'name': 'Hierarchical', 'clustering': True, 'method': 'hierarchical', 'cluster_on': 'both'},
        {'name': 'Birch', 'clustering': True, 'method': 'birch', 'cluster_on': 'both'},
        {'name': 'AffinityProp', 'clustering': True, 'method': 'affinity', 'cluster_on': 'both'},
    ]

    results = []
    print(f"\n{'#' * 80}")
    print(f" >>> Group Experiment: {group_name}")
    print(f" >>> Target Clusters: {n_clusters}")
    print(f"{'#' * 80}")

    for strat in strategies:
        exp_name = f"{group_name}_{strat['name']}"

        # --- 配置构建 ---
        config = copy.deepcopy(base_config)
        config['model']['name'] = 'xpatch'
        config['model']['config'] = {}

        model_config_path = Path('config/xpatch.yaml')
        if model_config_path.exists():
            with open(model_config_path, 'r', encoding='utf-8') as f:
                model_cfg = yaml.safe_load(f)
                config['model']['config'].update(model_cfg.get('config', model_cfg))

        # 聚类基础设置
        config['clustering']['enabled'] = strat['clustering']
        config['clustering']['cluster_on'] = strat['cluster_on']

        if strat['clustering']:
            method = strat['method']
            config['clustering']['method'] = method
            config['clustering']['recluster_every_n_rounds'] = 5

            # --- 为不同算法注入特定参数 ---
            if method in ['kmeans', 'gmm', 'spectral', 'hierarchical', 'birch']:
                config['clustering']['num_clusters'] = n_clusters

            if method == 'affinity':
                config['clustering']['damping'] = 0.8  # 0.5-1.0, 较高值可避免震荡

            if method == 'birch':
                config['clustering']['threshold'] = 0.5

            # 4. DBSCAN (eps, min_samples 使用 config.yaml 默认值或在此微调)
            # config['clustering']['eps'] = 0.5

        # 数据与训练设置
        config['data']['mode'] = 'multi_file_all_sheets'
        config['data']['files'] = files
        config['data']['window_size'] = win
        config['data']['pre_len'] = pre
        config['federation']['num_rounds'] = 40
        config['privacy']['enabled'] = False

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

        print(f"Running Strategy: {strat['name']}")

        # --- 初始化 ---
        set_seed(2024)
        g = torch.Generator()
        g.manual_seed(2024)

        client_dataloaders = setup_clients_multi_file_by_sheet(
            file_paths=files, window_size=win, pre_len=pre,
            batch_size=32, max_capacity=2.0, generator=g
        )

        num_clients = len(client_dataloaders)
        clients = [Client(i, dl, config, device) for i, dl in enumerate(client_dataloaders)]
        server = Server(config, num_clients, device)
        current_labels = source_labels[:num_clients]

        # --- 训练循环 ---
        for comm_round in range(config['federation']['num_rounds']):
            # 初始聚类 (Round 0)
            if comm_round == 0 and config['clustering']['enabled']:
                init_parts = server.get_global_model_parts(0)
                tmp_parts = {}
                for c in clients:
                    c.set_global_model(copy.deepcopy(init_parts))
                    c.local_train()
                    tmp_parts[c.client_id] = c.get_local_parameters()
                server.recluster_clients(tmp_parts)

            client_parts_dict = {}
            client_losses = {}

            # 本地训练
            for client in clients:
                global_parts = server.get_global_model_parts(client.client_id)
                client.set_global_model(copy.deepcopy(global_parts))
                loss = client.local_train()
                client_parts_dict[client.client_id] = client.get_local_parameters()
                client_losses[client.client_id] = loss

            # 可视化 (首、中、尾)
            viz_rounds = [0, 5, 20, config['federation']['num_rounds'] - 1]
            if comm_round in viz_rounds and client_parts_dict:
                visualize_clustering(
                    server, client_parts_dict, comm_round, viz_dir, current_labels,
                    cluster_on=strat['cluster_on']
                )

            # 聚合 & 重聚类
            server.aggregate_parameters(client_parts_dict, client_losses)
            if config['clustering']['enabled']:
                server.recluster_clients(client_parts_dict)

            if (comm_round + 1) % 10 == 0:
                print(f"     Round {comm_round + 1} Loss: {np.mean(list(client_losses.values())):.4f}")

        # --- 评估 ---
        all_metrics = []
        for client in clients:
            final_parts = server.get_global_model_parts(client.client_id)
            client.set_global_model(copy.deepcopy(final_parts))
            mae, rmse = client.evaluate(save_dir=exp_dir)
            all_metrics.append({
                'client_id': client.client_id,
                'MAE': mae, 'RMSE': rmse,
                'Cluster': server.client_clusters.get(client.client_id, 0)
            })

        avg_mae = np.mean([m['MAE'] for m in all_metrics])
        avg_rmse = np.mean([m['RMSE'] for m in all_metrics])

        results.append({
            'Group': group_name,
            'Strategy': strat['name'],
            'MAE': avg_mae,
            'RMSE': avg_rmse,
            'Final_K': server.num_clusters
        })

        # 保存该策略的详细指标
        pd.DataFrame(all_metrics).to_csv(os.path.join(exp_dir, 'client_metrics.csv'), index=False)

        # 清理显存
        del clients, server, client_dataloaders
        torch.cuda.empty_cache()

    return results


def main():
    base_config = load_config('config/config.yaml')

    # 定义实验组
    experiment_groups = [
        {
            'name': 'XJTU_All',
            'files': ['data/batch-1.xlsx', 'data/batch-2.xlsx', 'data/batch-3.xlsx'],
            'win': 50, 'pre': 200, 'n_clusters': 6
        },
        {
            'name': 'MIT_All',
            'files': ['data/batch-4.xlsx', 'data/batch-5.xlsx'],
            'win': 100, 'pre': 500, 'n_clusters': 3
        }
    ]

    timestamp = datetime.datetime.now().strftime("%m%d-%H%M")
    parent_dir = os.path.join(base_config['results']['save_dir_prefix'], f"exp_3_{timestamp}")
    os.makedirs(parent_dir, exist_ok=True)
    print(f"Results saved to: {parent_dir}")

    all_results = []
    for group in experiment_groups:
        valid_files = [f for f in group['files'] if os.path.exists(f)]
        if not valid_files: continue

        group_res = run_group_experiment(
            group['name'], valid_files, group['win'], group['pre'],
            group['n_clusters'], base_config, parent_dir
        )
        all_results.extend(group_res)

    if all_results:
        df = pd.DataFrame(all_results)
        print("\n" + "=" * 80)
        print("Experiment 3 Summary")
        print("=" * 80)
        print(df.to_string(index=False))
        df.to_csv(os.path.join(parent_dir, 'exp_3_summary.csv'), index=False)


if __name__ == '__main__':
    main()
