import os
import torch
import numpy as np
import copy
import pandas as pd
import datetime
import yaml
import time
from pathlib import Path
from utils.config_utils import load_config
from utils.data_loader import setup_clients_by_sheet
from utils.reporting_utils import save_summary_report
from client import Client
from server import Server


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def calculate_comm_size_mb(params_dict):
    """计算参数字典中所有张量的总大小 (MB)"""
    total_elements = 0
    for part_name, params in params_dict.items():
        for key, tensor in params.items():
            total_elements += tensor.numel()
    # 假设 float32 (4 bytes)
    size_in_mb = total_elements * 4 / (1024 * 1024)
    return size_in_mb


def run_single_pfl_experiment(file_path, strategy_name, window_size, pre_len, base_config, parent_dir):
    config = copy.deepcopy(base_config)

    config['model']['name'] = 'xpatch'
    config['model']['pfl_enabled'] = True
    config['aggregation'] = {'name': 'fedavgm', 'beta': 0.9, 'server_lr': 1.0}
    config['privacy']['enabled'] = False
    config['clustering']['enabled'] = False

    config['data']['window_size'] = window_size
    config['data']['pre_len'] = pre_len
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
    file_name = os.path.basename(file_path)

    exp_sub_name = f"{file_name}_{strategy_name}"
    exp_dir = os.path.join(parent_dir, exp_sub_name)
    os.makedirs(os.path.join(exp_dir, 'results'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'models'), exist_ok=True)

    print(f"\n{'=' * 80}")
    print(f" >>> 执行 pFL 实验: Dataset=[{file_name}] | Strategy=[{strategy_name.upper()}]")
    print(f"{'=' * 80}")

    set_seed(config.get('seed', 42))
    g = torch.Generator()
    g.manual_seed(config.get('seed', 42))

    # (单文件模式)
    client_dataloaders = setup_clients_by_sheet(
        file_path=file_path,
        window_size=window_size,
        pre_len=pre_len,
        batch_size=config['federation']['batch_size'],
        max_capacity=config['data']['max_capacity'],
        generator=g
    )

    if not client_dataloaders:
        print(f"Error: {file_name} 未能加载数据。")
        return None

    num_clients = len(client_dataloaders)
    clients = [Client(i, dl, config, device) for i, dl in enumerate(client_dataloaders)]
    server = Server(config, num_clients, device)

    num_rounds = config['federation']['num_rounds']
    metrics_history = {'train_time': [], 'comm_size': [] }

    for comm_round in range(num_rounds):
        client_parts_dict = {}
        client_losses = {}
        round_client_times = []
        round_client_sizes = []

        for client in clients:
            global_parts = server.get_global_model_parts(client.client_id)

            if strategy_name == 'fedbn':
                if 'full_model_no_bn' in global_parts:
                    current_dict = client.model.state_dict()
                    current_dict.update(global_parts['full_model_no_bn'])
                    client.model.load_state_dict(current_dict)
                elif 'full_model' in global_parts:  # 首轮
                    # 过滤掉 BN
                    to_load = {k: v for k, v in global_parts['full_model'].items()
                               if 'bn' not in k and 'running' not in k}
                    client.model.load_state_dict(to_load, strict=False)
            else:
                # xPatch_pFL 和 FedRep 都使用部分参数加载逻辑 (set_global_model 内部处理)
                client.set_global_model(copy.deepcopy(global_parts))

            start_time = time.time()

            if strategy_name == 'fedrep':
                # FedRep: 先训 Head 再训 Body
                loss = client.local_train_fedrep(head_epochs=5)
            else:
                # FedBN 和 xPatch 使用标准训练
                loss = client.local_train()

            end_time = time.time()
            round_client_times.append(end_time - start_time)

            local_parts = client.get_parameters_by_strategy(strategy_name)

            comm_size = calculate_comm_size_mb(local_parts)
            round_client_sizes.append(comm_size)

            client_parts_dict[client.client_id] = local_parts
            client_losses[client.client_id] = loss

        avg_time = np.mean(round_client_times)
        avg_size = np.mean(round_client_sizes)
        metrics_history['train_time'].append(avg_time)
        metrics_history['comm_size'].append(avg_size)

        server.aggregate_parameters(client_parts_dict, client_losses)

        if (comm_round + 1) % 10 == 0:
            avg_loss = np.mean(list(client_losses.values()))
            print(f"  Round {comm_round + 1}/{num_rounds} - Avg Loss: {avg_loss:.4f} | "
                  f"Time: {avg_time:.2f}s | Size: {avg_size:.2f}MB")

    print(f"正在评估 {strategy_name} ...")
    all_metrics = []
    for client in clients:
        # 获取最终全局参数
        global_parts = server.get_global_model_parts(client.client_id)

        # 加载参数用于评估
        if strategy_name != 'fedbn':
            # FedBN 保留本地最优 BN，不需要覆盖；其他策略需要加载最新的全局 Body
            client.set_global_model(copy.deepcopy(global_parts))

        mae, rmse = client.evaluate(save_dir=exp_dir)
        all_metrics.append({'client_id': client.client_id, 'MAE': mae, 'RMSE': rmse})

    avg_mae = np.mean([m['MAE'] for m in all_metrics])
    avg_rmse = np.mean([m['RMSE'] for m in all_metrics])
    std_mae = np.std([m['MAE'] for m in all_metrics])
    std_rmse = np.std([m['RMSE'] for m in all_metrics])

    final_avg_time = np.mean(metrics_history['train_time'])
    final_avg_size = np.mean(metrics_history['comm_size'])

    save_summary_report(exp_dir, all_metrics, {'MAE': avg_mae, 'RMSE': avg_rmse})
    print(f"实验完成: {file_name} | {strategy_name} -> MAE={avg_mae:.4f}")
    print(f"  > 平均训练耗时: {final_avg_time:.4f} s/round")
    print(f"  > 平均通信大小: {final_avg_size:.4f} MB/round")

    return {
        'Dataset': file_name,
        'Strategy': strategy_name,
        'MAE': avg_mae,
        'RMSE': avg_rmse, 'MAE_std': std_mae, 'RMSE_std': std_rmse,
        'Avg_Time_Sec': final_avg_time,
        'Comm_Size_MB': final_avg_size
    }


def main():
    base_config = load_config('config/config.yaml')

    files_plan = [
        {'path': 'data/batch-1.xlsx', 'win': 50, 'pre': 200},
        {'path': 'data/batch-2.xlsx', 'win': 50, 'pre': 200},
        {'path': 'data/batch-3.xlsx', 'win': 50, 'pre': 200},
        {'path': 'data/batch-4.xlsx', 'win': 100, 'pre': 500},
        {'path': 'data/batch-5.xlsx', 'win': 100, 'pre': 500},
    ]

    strategies = [
        'xpatch_pfl',  # 本文提出的方法 (Trend/Seasonal 分离)
        'fedrep',  # 基线: FedRep (共享 Body，本地 Head)
        'fedbn'  # 基线: FedBN (不上传 BN 层)
    ]

    timestamp = datetime.datetime.now().strftime("%m%d-%H%M")
    base_save_dir = base_config['results']['save_dir_prefix']
    parent_dir = os.path.join(base_save_dir, f"exp_3_{timestamp}")
    os.makedirs(parent_dir, exist_ok=True)

    print(f"开始 pFL 对比实验，结果保存在: {parent_dir}")

    summary_results = []

    for plan in files_plan:
        file_path = plan['path']
        if not os.path.exists(file_path):
            print(f"跳过: 文件 {file_path} 不存在")
            continue

        for strategy in strategies:
            try:
                result = run_single_pfl_experiment(
                    file_path=file_path,
                    strategy_name=strategy,
                    window_size=plan['win'],
                    pre_len=plan['pre'],
                    base_config=base_config,
                    parent_dir=parent_dir
                )
                if result:
                    summary_results.append(result)
            except Exception as e:
                print(f"实验出错 ({file_path} - {strategy}): {e}")
                import traceback
                traceback.print_exc()

    if summary_results:
        df = pd.DataFrame(summary_results)
        print("\n" + "#" * 60)
        print("pFL 算法对比汇总报告")
        print("#" * 60)
        print(df.to_string(index=False))

        csv_path = os.path.join(parent_dir, 'pfl_comparison_summary.csv')
        df.to_csv(csv_path, index=False)
        print(f"汇总表格已保存至: {csv_path}")


if __name__ == '__main__':
    main()
