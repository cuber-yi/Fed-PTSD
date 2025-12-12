import os
import torch
import numpy as np
import random
import copy
import datetime
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# 导入原有项目模块
from utils.config_utils import load_config
from utils.data_loader import setup_clients_multi_file_by_sheet
from utils.reporting_utils import save_summary_report
from client import Client
from server import Server
from src.cluster.utils import vectorize_client_params


# --- 1. 升级版可视化函数: 支持 stage 标签 ---
def plot_clustering_snapshot(client_parts_dict, assignments, round_num, save_dir, stage, method='pca'):
    """
    绘制客户端分布图
    参数 stage: 用于标记阶段 (e.g., "Before_Cluster", "After_Cluster", "Round+10")
    """
    try:
        client_ids, X = vectorize_client_params(client_parts_dict, use_pca=False)
    except Exception as e:
        print(f"  [Vis Error] 向量化参数失败: {e}")
        return

    # 降维
    if X.shape[1] > 2:
        if method == 'tsne':
            perp = min(30, len(client_ids) - 1) if len(client_ids) > 1 else 1
            reducer = TSNE(n_components=2, random_state=42, perplexity=perp, init='pca', learning_rate='auto')
        else:
            reducer = PCA(n_components=2, random_state=42)
        X_2d = reducer.fit_transform(X)
    else:
        X_2d = X[:, :2] if X.shape[1] >= 2 else np.column_stack((X, np.zeros_like(X)))

    # 准备绘图数据
    # 注意：assignments 可能不包含所有 client_id (如果刚开始)，需用 get(cid, 0) 处理
    cluster_labels = [f'Cluster {assignments.get(cid, 0)}' for cid in client_ids]

    data = pd.DataFrame({
        'Component 1': X_2d[:, 0],
        'Component 2': X_2d[:, 1],
        'Cluster': cluster_labels,
        'Client ID': [f'C{cid}' for cid in client_ids]
    })

    plt.figure(figsize=(10, 8))
    sns.set_style("whitegrid")

    # 使用 tab10 调色板，保证颜色区分度
    sns.scatterplot(
        data=data, x='Component 1', y='Component 2',
        hue='Cluster', style='Cluster', palette='tab10', s=150, alpha=0.9
    )

    # 标注 Client ID
    for i in range(data.shape[0]):
        plt.text(data['Component 1'][i] + 0.02, data['Component 2'][i] + 0.02,
                 data['Client ID'][i], fontsize=9, weight='bold')

    # 标题包含阶段信息
    plt.title(f'Round {round_num} - {stage} ({method.upper()})', fontsize=15)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # 文件名包含阶段信息，防止覆盖
    save_path_dir = os.path.join(save_dir, 'cluster_plots')
    os.makedirs(save_path_dir, exist_ok=True)
    filename = f'R{round_num:03d}_{stage}.png'
    plt.savefig(os.path.join(save_path_dir, filename), dpi=300)
    plt.close()
    print(f"  [Vis] 已保存绘图: {filename}")


# --- 2. 升级版 VizServer: 注入四个阶段的绘图逻辑 ---
class VizServer(Server):
    def set_viz_params(self, save_dir, current_round):
        self.viz_save_dir = save_dir
        self.viz_current_round = current_round

    def visualize_custom(self, client_parts_dict, stage_name):
        """
        手动调用绘图（用于 '10轮后' 和 '结束时'）
        """
        if hasattr(self, 'viz_save_dir'):
            plot_clustering_snapshot(
                client_parts_dict,
                self.client_clusters,  # 使用当前的聚类结果
                self.viz_current_round,
                self.viz_save_dir,
                stage=stage_name,
                method='pca'
            )

    def recluster_clients(self, client_parts_dict: dict):
        # -------------------------------------------------
        # 阶段 1: 聚类前 (Before Clustering)
        # -------------------------------------------------
        if hasattr(self, 'viz_save_dir') and self.clustering_enabled:
            print(f"  [Vis] 正在绘制: 聚类前 (Round {self.viz_current_round})")
            plot_clustering_snapshot(
                client_parts_dict,
                self.client_clusters,  # 此时还是旧的聚类标签
                self.viz_current_round,
                self.viz_save_dir,
                stage="1_Before_Clustering",
                method='pca'
            )

        # 执行原始聚类逻辑 (计算新标签)
        super().recluster_clients(client_parts_dict)

        # -------------------------------------------------
        # 阶段 2: 聚类后 (After Clustering)
        # -------------------------------------------------
        if hasattr(self, 'viz_save_dir') and self.clustering_enabled:
            print(f"  [Vis] 正在绘制: 聚类后 (Round {self.viz_current_round})")
            plot_clustering_snapshot(
                client_parts_dict,
                self.client_clusters,  # 此时已更新为新标签
                self.viz_current_round,
                self.viz_save_dir,
                stage="2_After_Clustering",
                method='pca'
            )


# --- 3. 实验主逻辑 ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


def run_viz_experiment(dataset_group_name, files, cluster_config, base_config, parent_dir):
    config = copy.deepcopy(base_config)

    # 强制配置
    config['model']['name'] = 'xpatch'
    config['model']['pfl_enabled'] = True
    config['aggregation'] = {'name': 'fedavgm', 'beta': 0.9, 'server_lr': 1.0}
    config['privacy']['enabled'] = False
    config['clustering']['enabled'] = True
    config['clustering'].update(cluster_config)
    config['data']['mode'] = 'multi_file_all_sheets'
    config['data']['files'] = files
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

    exp_name = f"VizStages_{dataset_group_name}"
    exp_dir = os.path.join(parent_dir, exp_name)

    # 创建目录
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'results'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'cluster_plots'), exist_ok=True)

    print(f"\n>>> 运行阶段可视化实验: {exp_name}")
    print(f">>> 结果将保存至: {exp_dir}/cluster_plots")

    set_seed(config.get('seed', 42))
    g = torch.Generator()
    g.manual_seed(config.get('seed', 42))

    client_dataloaders = setup_clients_multi_file_by_sheet(
        file_paths=files, window_size=50, pre_len=200,
        batch_size=config['federation']['batch_size'], max_capacity=config['data']['max_capacity'], generator=g
    )

    if not client_dataloaders: return None

    clients = [Client(i, dl, config, device) for i, dl in enumerate(client_dataloaders)]
    server = VizServer(config, len(clients), device)

    # 实验轮次设置
    num_rounds = 50
    WARMUP_ROUNDS = config['clustering'].get('warmup_rounds', 10)
    print(f">>> 策略: Warmup={WARMUP_ROUNDS} 轮，将在该轮次触发一次性聚类。")

    last_round_client_parts = {}
    clustering_happened_round = -1  # 记录聚类发生的具体轮次

    for comm_round in range(num_rounds):
        server.set_viz_params(exp_dir, comm_round)

        # === 聚类触发逻辑 ===
        if config['clustering']['enabled']:
            # 只在 Warmup 结束的那一轮触发一次聚类
            if comm_round == WARMUP_ROUNDS:
                print(f"  [Cluster] 触发一次性聚类 (Round {comm_round})...")
                # VizServer.recluster_clients 内部会自动画 "Before" 和 "After" 两张图
                if last_round_client_parts:
                    server.recluster_clients(last_round_client_parts)
                    clustering_happened_round = comm_round
                else:
                    print("  [Warning] 没有上一轮参数，跳过聚类。")

        # === 联邦训练循环 ===
        client_parts_dict = {}
        client_losses = {}

        for client in clients:
            global_parts = server.get_global_model_parts(client.client_id)
            client.set_global_model(copy.deepcopy(global_parts))
            loss = client.local_train()
            client_parts_dict[client.client_id] = client.get_local_parameters()
            client_losses[client.client_id] = loss

        last_round_client_parts = copy.deepcopy(client_parts_dict)

        # -------------------------------------------------
        # 阶段 3: 聚类10轮后 (10 Rounds Post-Cluster)
        # -------------------------------------------------
        if clustering_happened_round != -1 and comm_round == clustering_happened_round + 10:
            print(f"  [Vis] 正在绘制: 聚类10轮后 (Round {comm_round})")
            server.visualize_custom(client_parts_dict, "3_10_Rounds_Post_Cluster")

        # 聚合
        server.aggregate_parameters(client_parts_dict, client_losses)

        if (comm_round + 1) % 5 == 0:
            print(
                f"  Round {comm_round + 1}/{num_rounds} Complete. Avg Loss: {np.mean(list(client_losses.values())):.4f}")

    # -------------------------------------------------
    # 阶段 4: 训练结束时 (End of Training)
    # -------------------------------------------------
    print(f"  [Vis] 正在绘制: 训练结束时 (Round {num_rounds - 1})")
    # 使用最后一轮的参数 last_round_client_parts
    server.set_viz_params(exp_dir, num_rounds - 1)
    server.visualize_custom(last_round_client_parts, "4_End_Of_Training")

    # 最终评估
    print("正在评估...")
    all_metrics = []
    for client in clients:
        final_parts = server.get_global_model_parts(client.client_id)
        client.set_global_model(copy.deepcopy(final_parts))
        mae, rmse = client.evaluate(save_dir=exp_dir)
        all_metrics.append({'client_id': client.client_id, 'MAE': mae, 'RMSE': rmse})

    avg_mae = np.mean([m['MAE'] for m in all_metrics])
    save_summary_report(exp_dir, all_metrics, {'MAE': avg_mae, 'RMSE': 0.0})
    print(f"实验结束. MAE: {avg_mae:.4f}")


def main():
    base_config = load_config('config/config.yaml')
    timestamp = datetime.datetime.now().strftime("%m%d-%H%M")
    parent_dir = os.path.join("result", f"exp_5_{timestamp}")

    files = ['data/batch-1.xlsx', 'data/batch-2.xlsx', 'data/batch-3.xlsx']
    files = [f for f in files if os.path.exists(f)]

    if not files:
        print("未找到数据文件，请检查路径。")
        return

    cluster_config = {
        'method': 'kmeans',
        'num_clusters': 3
    }

    try:
        run_viz_experiment("XJTU", files, cluster_config, base_config, parent_dir)
    except Exception as e:
        print(f"运行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()