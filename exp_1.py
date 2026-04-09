import os
import torch
import numpy as np
import random
import copy
import datetime
import yaml
import pandas as pd
from pathlib import Path

# 项目内部模块导入
from utils.config_utils import load_config
from utils.data_loader import setup_clients_by_sheet
from utils.reporting_utils import save_summary_report
from client import Client
from server import Server


def set_seed(seed):
    """设置随机种子以保证结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_single_experiment(file_path, model_name, window_size, pre_len, base_config, parent_dir):
    """
    运行单个具体的实验任务：特定的数据集 + 特定的模型
    """
    # --- 1. 动态构建配置 ---
    config = copy.deepcopy(base_config)

    # 设置模型名称
    config['model']['name'] = model_name
    config['model']['config'] = {}

    # 加载特定模型的参数配置文件
    model_config_path = Path('config') / f"{model_name}.yaml"
    if model_config_path.exists():
        with open(model_config_path, 'r', encoding='utf-8') as f:
            model_specific_config = yaml.safe_load(f)
            if model_specific_config:
                if 'config' in model_specific_config:
                    config['model']['config'].update(model_specific_config['config'])
                else:
                    config['model']['config'].update(model_specific_config)
    else:
        print(f"[Warning] 未找到模型配置文件: {model_config_path}，将使用默认参数。")

    # --- 2. 强制设定实验环境（控制变量） ---
    # 禁用高级特性，仅对比基础模型性能
    config['clustering']['enabled'] = False
    config['model']['pfl_enabled'] = False
    config['privacy']['enabled'] = False

    if 'aggregation' not in config:
        config['aggregation'] = {}
    config['aggregation']['name'] = 'fedavg'

    # 设置数据参数
    config['data']['window_size'] = window_size
    config['data']['pre_len'] = pre_len

    # --- 3. 关键：注入维度信息到模型配置 ---
    # 许多模型需要知道 seq_len (输入长度) 和 pred_len (输出长度)
    config['model']['config']['enc_in'] = config['data']['enc_in']
    config['model']['config']['pred_len'] = config['data']['pre_len']
    config['model']['config']['seq_len'] = config['data']['window_size']

    # 获取设备
    device_str = config['data']['device']
    device = torch.device(device_str if device_str != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu'))

    # --- 4. 准备实验记录 ---
    file_name = os.path.basename(file_path)
    print(f"\n{'=' * 80}")
    print(f" >>> 执行实验: Dataset=[{file_name}] | Model=[{model_name.upper()}]")
    print(f" >>> 设置: Window={window_size}, Pred={pre_len}")
    print(f"{'=' * 80}")

    exp_dir = os.path.join(parent_dir, f"{file_name}_{model_name}")
    os.makedirs(os.path.join(exp_dir, 'results'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'models'), exist_ok=True)

    # --- 5. 初始化环境与数据 ---
    seed = config.get('seed', 42)
    set_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)

    print(f"正在从 {file_path} 加载数据 (Sheet模式)...")
    # 使用 setup_clients_by_sheet 将每个 Sheet 视为一个 Client
    client_dataloaders = setup_clients_by_sheet(
        file_path=file_path,
        window_size=window_size,
        pre_len=pre_len,
        batch_size=config['federation']['batch_size'],
        max_capacity=config['data']['max_capacity'],
        generator=g
    )

    num_clients = len(client_dataloaders)
    if num_clients == 0:
        print(f"Error: 未能从 {file_path} 创建任何客户端，跳过此实验。")
        return None
    print(f"成功加载 {num_clients} 个客户端。")

    # --- 6. 初始化联邦系统 ---
    clients = [Client(client_id=i, dataloader=dl, config=config, device=device)
               for i, dl in enumerate(client_dataloaders)]
    server = Server(config=config, num_total_clients=num_clients, device=device)

    # --- 7. 训练循环 (Standard FedAvg) ---
    num_rounds = config['federation']['num_rounds']

    for comm_round in range(num_rounds):
        client_parts_dict = {}
        client_losses_dict = {}

        for client in clients:
            # 1. Server 下发全局模型
            global_parts = server.get_global_model_parts(client.client_id)
            client.set_global_model(copy.deepcopy(global_parts))

            # 2. Client 本地训练
            loss = client.local_train()
            local_parts = client.get_local_parameters()

            # 3. 收集参数和损失
            client_parts_dict[client.client_id] = local_parts
            client_losses_dict[client.client_id] = loss

        # 4. Server 聚合
        server.aggregate_parameters(client_parts_dict, client_losses_dict)

        # 进度打印
        if (comm_round + 1) % 5 == 0 or comm_round == 0:
            print(f"  Round {comm_round + 1}/{num_rounds} - Last Client Loss: {loss:.4f}")

    # --- 8. 最终评估 ---
    print("正在评估所有客户端...")
    all_metrics = []
    for client in clients:
        # 获取最终全局模型进行评估
        final_parts = server.get_global_model_parts(client.client_id)
        client.set_global_model(copy.deepcopy(final_parts))
        mae, rmse = client.evaluate(save_dir=exp_dir)
        all_metrics.append({'client_id': client.client_id, 'MAE': mae, 'RMSE': rmse})

    # 计算平均值
    avg_mae = np.mean([m['MAE'] for m in all_metrics])
    avg_rmse = np.mean([m['RMSE'] for m in all_metrics])
    std_mae = np.std([m['MAE'] for m in all_metrics])
    std_rmse = np.std([m['RMSE'] for m in all_metrics])


    save_summary_report(exp_dir, all_metrics,
                        {'MAE': avg_mae, 'RMSE': avg_rmse, 'MAE_std': std_mae, 'RMSE_std': std_rmse})
    print(f"实验完成。 Avg MAE: {avg_mae:.4f} (±{std_mae:.4f}), Avg RMSE: {avg_rmse:.4f} (±{std_rmse:.4f})")

    return {'Dataset': file_name, 'Model': model_name, 'MAE': avg_mae, 'RMSE': avg_rmse,
            'MAE_std': std_mae, 'RMSE_std': std_rmse}


def main():
    # 读取基础配置
    base_config = load_config('config/config.yaml')

    # 1. 数据集与参数计划
    files_plan = [
        {'path': 'data/batch-1.xlsx', 'win': 50, 'pre': 200},
        {'path': 'data/batch-2.xlsx', 'win': 50, 'pre': 200},
        {'path': 'data/batch-3.xlsx', 'win': 50, 'pre': 200},
        {'path': 'data/batch-4.xlsx', 'win': 100, 'pre': 500},
        {'path': 'data/batch-5.xlsx', 'win': 100, 'pre': 500},
    ]

    # 2. 待测试模型列表
    models_to_test = [
        'xpatch',
        'patchtst',
        'timesnet',
        'fedformer',
        'pyraformer',
        'dlinear',
        'rnn',
        'lstm',
        'gru',
        'mlp'
    ]

    # --- 准备总目录 ---
    timestamp = datetime.datetime.now().strftime("%m%d-%H%M")
    base_save_dir = base_config['results']['save_dir_prefix']
    parent_dir_name = f"exp_1_{timestamp}"
    parent_dir = os.path.join(base_save_dir, parent_dir_name)
    os.makedirs(parent_dir, exist_ok=True)
    print(f"本次所有实验结果将保存在: {parent_dir}")

    summary_results = []
    total_experiments = len(files_plan) * len(models_to_test)
    current_idx = 0

    print(f"计划执行 {total_experiments} 个实验任务...")

    # --- 主循环执行实验 ---
    for plan in files_plan:
        file_path = plan['path']

        if not os.path.exists(file_path):
            print(f"跳过: 文件 {file_path} 不存在")
            current_idx += len(models_to_test)
            continue

        for model_name in models_to_test:
            current_idx += 1

            try:
                result = run_single_experiment(
                    file_path=file_path,
                    model_name=model_name,
                    window_size=plan['win'],
                    pre_len=plan['pre'],
                    base_config=base_config,
                    parent_dir=parent_dir
                )

                if result:
                    summary_results.append(result)

            except KeyboardInterrupt:
                print("\n用户中断实验。正在输出已完成的结果...")
                # 中断时直接跳出所有循环
                summary_results_df = pd.DataFrame(summary_results)
                summary_path = os.path.join(parent_dir, 'exp_1.csv')
                summary_results_df.to_csv(summary_path, index=False)
                return
            except Exception as e:
                print(f"\n!!! 实验出错 ({file_path} - {model_name}): {e}")
                import traceback
                traceback.print_exc()

    # --- 输出最终汇总表 ---
    if summary_results:
        df = pd.DataFrame(summary_results)
        print("\n" + "#" * 60)
        print("模型对比汇总报告 (Model Comparison Summary)")
        print("#" * 60)
        print(df.to_string(index=False))
        print("#" * 60)

        # 保存汇总CSV
        summary_path = os.path.join(parent_dir, 'exp_1.csv')
        df.to_csv(summary_path, index=False)
        print(f"汇总表格已保存至: {summary_path}")
    else:
        print("没有完成任何实验。")


if __name__ == '__main__':
    main()
