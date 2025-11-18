import numpy as np
import torch
import copy
import os
import datetime
import random
from utils.config_utils import load_config
from utils.data_loader import setup_clients_by_sheet, setup_clients_by_file, setup_clients_multi_file_by_sheet
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


def main():
    # --- 加载配置 ---
    config = load_config('config/config.yaml')
    seed = config.get('seed', 42)
    set_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    device = torch.device(config['data']['device'])
    aggregation_name = config.get('aggregation', {}).get('name', 'fedavg')
    print(f"加载模型: '{config['model']['name']}'. 运行设备: {device}")
    print(f"聚合策略: {aggregation_name}")

    # --- 创建结果保存目录 ---
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = f"{config['results']['save_dir_prefix']}{timestamp}"
    os.makedirs(os.path.join(save_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'results'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'plots'), exist_ok=True)
    print(f"结果保存地址: {save_dir}")

    # --- 准备数据和客户端 ---
    data_mode = config['data'].get('mode', 'multi_file')

    if data_mode == 'single_file_multi_client':
        print("从单文件配置客户端")
        single_file_path = config['data'].get('single_file', None)
        client_dataloaders = setup_clients_by_sheet(
            file_path=single_file_path,
            window_size=config['data']['window_size'],
            pre_len=config['data']['pre_len'],
            batch_size=config['federation']['batch_size'],
            max_capacity=config['data']['max_capacity'],
            generator=g
        )
    elif data_mode == 'multi_file':
        print("从多文件配置客户端")
        client_dataloaders = setup_clients_by_file(
            file_paths=config['data']['files'],
            window_size=config['data']['window_size'],
            pre_len=config['data']['pre_len'],
            batch_size=config['federation']['batch_size'],
            max_capacity=config['data']['max_capacity'],
            generator=g
        )
    elif data_mode == 'multi_file_all_sheets':
        print("配置模式: 多文件 - 全Sheet独立客户端 (Multi-File Individual Sheets)")
        # 确保这里使用的是 files 列表
        client_dataloaders = setup_clients_multi_file_by_sheet(
            file_paths=config['data']['files'],
            window_size=config['data']['window_size'],
            pre_len=config['data']['pre_len'],
            batch_size=config['federation']['batch_size'],
            max_capacity=config['data']['max_capacity'],
            generator=g
        )

    num_total_clients = len(client_dataloaders)
    print(f"总共创建客户端数量: {num_total_clients}")

    # 初始化Client和Server实例
    clients = [Client(client_id=i, dataloader=dl, config=config, device=device) for
               i, dl in enumerate(client_dataloaders)]
    server = Server(config=config, num_total_clients=num_total_clients, device=device)

    # --- 联邦学习主循环 ---
    num_rounds = config['federation']['num_rounds']
    clustering_config = config.get('clustering', {})
    clustering_enabled = clustering_config.get('enabled', False)
    RECLUSTER_EVERY = clustering_config.get('recluster_every_n_rounds', 10)
    if clustering_enabled:
        print(f"聚类联邦学习已启用. K={clustering_config.get('num_clusters')}, 每 {RECLUSTER_EVERY} 轮重新聚类.")
    last_round_client_parts = {}

    for comm_round in range(num_rounds):
        print(f"\n{'=' * 20} 通信轮次： {comm_round + 1}/{num_rounds} {'=' * 20}")

        # --- 触发重新聚类 ---
        if clustering_enabled and (comm_round == 0 or comm_round % RECLUSTER_EVERY == 0):
            # 第0轮特殊处理：客户端需要先训练一次才能有参数
            if comm_round == 0:
                print("[Round 0] 进行初始训练以用于首次聚类...")
                # 让所有客户端使用 cluster 0 的模型训练一次
                initial_parts = server.get_global_model_parts(client_id=0)
                for client in clients:
                    client.set_global_model(copy.deepcopy(initial_parts))
                    client.local_train()  # 只训练，不聚合
                    last_round_client_parts[client.client_id] = client.get_local_parameters()

            # 使用上一轮的参数执行聚类
            if last_round_client_parts:
                server.recluster_clients(last_round_client_parts)
            else:
                print("[Server Warning] 没有客户端参数可用于聚类.")

        client_parts_dict = {}
        client_losses_dict = {}
        # --- 客户端本地训练 ---
        for client in clients:
            # 【修改】客户端根据自己的ID获取对应的 cluster 模型
            global_model_parts = server.get_global_model_parts(client.client_id)
            # 客户端加载拆分后的模型
            client.set_global_model(copy.deepcopy(global_model_parts))
            # 本地训练并获取损失
            train_loss = client.local_train()
            # 客户端返回拆分后的本地参数
            local_parts = client.get_local_parameters()
            client_parts_dict[client.client_id] = local_parts
            client_losses_dict[client.client_id] = train_loss

            cluster_id = server.client_clusters.get(client.client_id, 0)
            print(f"  Client {client.client_id} (Cluster {cluster_id}) 训练完成, 损失: {train_loss:.4f}")

        last_round_client_parts = copy.deepcopy(client_parts_dict)
        # 服务器分别聚合所有部分（传递损失信息）
        server.aggregate_parameters(client_parts_dict, client_losses_dict)

        # 打印聚合器信息（如果支持）
        agg_info = server.get_aggregator_info()
        if agg_info is not None:
            print(f"  聚合权重 - Max: {agg_info['max_weight']:.4f}, "
                  f"Min: {agg_info['min_weight']:.4f}, "
                  f"Std: {agg_info['std_weight']:.4f}")

    # --- 最终评估 ---
    print(f"\n{'=' * 20} 开始最终评估 {'=' * 20}")
    all_metrics = []
    for client in clients:
        # 获取最终的、特定于其 cluster 的全局模型
        final_global_model_parts = server.get_global_model_parts(client.client_id)
        client.set_global_model(copy.deepcopy(final_global_model_parts))
        mae, rmse = client.evaluate(save_dir=save_dir)
        # 在指标中记录客户端ID和其最终所属的 cluster_id
        cluster_id = server.client_clusters.get(client.client_id, 0)
        all_metrics.append({'client_id': client.client_id, 'cluster_id': cluster_id, 'MAE': mae, 'RMSE': rmse})

    # --- 打印最终结果摘要 ---
    print(f"\n{'=' * 20} 评估结果 {'=' * 20}")
    avg_mae = np.mean([m['MAE'] for m in all_metrics])
    avg_rmse = np.mean([m['RMSE'] for m in all_metrics])
    for metrics in all_metrics:
        print(f"Client {metrics['client_id']}: MAE = {metrics['MAE']:.4f}, RMSE = {metrics['RMSE']:.4f}")
    print(f"\n平均评估结果: MAE = {avg_mae:.4f}, RMSE = {avg_rmse:.4f}")

    # --- 保存摘要 ---
    avg_metrics = {'MAE': avg_mae, 'RMSE': avg_rmse}
    save_summary_report(save_dir=save_dir, all_metrics=all_metrics, avg_metrics=avg_metrics)


if __name__ == '__main__':
    main()
