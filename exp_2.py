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


def run_fl_experiment(file_path, agg_config, model_name, window_size, pre_len, base_config, parent_dir):
    # --- 1. 动态构建配置 ---
    config = copy.deepcopy(base_config)
    # 固定使用的模型
    config['model']['name'] = model_name
    # 强制启用 PFL (个性化层分离)
    config['model']['pfl_enabled'] = True
    if 'pfl_enabled' in agg_config:
        config['model']['pfl_enabled'] = agg_config['pfl_enabled']

    agg_config_clean = {k: v for k, v in agg_config.items() if k not in ['pfl_enabled', 'display_name']}
    config['aggregation'] = agg_config_clean
    agg_name = agg_config['name']
    exp_tag = agg_config.get('display_name', agg_name)
    # 暂时禁用隐私和聚类，控制变量，专注于对比“聚合算法”
    config['privacy']['enabled'] = False
    config['clustering']['enabled'] = False
    # 设置数据参数
    config['data']['window_size'] = window_size
    config['data']['pre_len'] = pre_len
    # 注入维度信息到模型配置
    if 'config' not in config['model']:
        config['model']['config'] = {}
    config['model']['config']['enc_in'] = config['data']['enc_in']
    config['model']['config']['pred_len'] = config['data']['pre_len']
    config['model']['config']['seq_len'] = config['data']['window_size']
    # 加载模型特定配置
    model_config_path = Path('config') / f"{model_name}.yaml"
    if model_config_path.exists():
        with open(model_config_path, 'r', encoding='utf-8') as f:
            model_specific = yaml.safe_load(f)
            if model_specific:
                if 'config' in model_specific:
                    config['model']['config'].update(model_specific['config'])
                else:
                    config['model']['config'].update(model_specific)

    # 获取设备
    device_str = config['data']['device']
    device = torch.device(device_str if device_str != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu'))

    # --- 2. 准备实验记录 ---
    file_name = os.path.basename(file_path)
    # 实验命名规则: 数据集_模型_算法
    exp_sub_name = f"{file_name}_{model_name}_{agg_name}"

    print(f"\n{'=' * 80}")
    print(f" >>> 执行实验: Dataset=[{file_name}] | Alg=[{agg_name}]")
    print(f" >>> 聚合参数: {agg_config}")
    print(f"{'=' * 80}")

    exp_dir = os.path.join(parent_dir, exp_sub_name)
    os.makedirs(os.path.join(exp_dir, 'results'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'models'), exist_ok=True)

    # --- 3. 初始化环境与数据 ---
    seed = config.get('seed', 42)
    set_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)

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

    # --- 4. 初始化联邦系统 ---
    clients = [Client(client_id=i, dataloader=dl, config=config, device=device)
               for i, dl in enumerate(client_dataloaders)]
    server = Server(config=config, num_total_clients=num_clients, device=device)

    # --- 5. 训练循环 ---
    num_rounds = config['federation']['num_rounds']

    # 用于记录 Loss 曲线
    loss_history = []

    for comm_round in range(num_rounds):
        client_parts_dict = {}
        client_losses_dict = {}
        round_losses = []

        for client in clients:
            # 1. Server 下发
            global_parts = server.get_global_model_parts(client.client_id)
            client.set_global_model(copy.deepcopy(global_parts))

            # 2. Client 本地训练
            loss = client.local_train()
            local_parts = client.get_local_parameters()

            # 3. 收集
            client_parts_dict[client.client_id] = local_parts
            client_losses_dict[client.client_id] = loss
            round_losses.append(loss)

        # 4. Server 聚合
        server.aggregate_parameters(client_parts_dict, client_losses_dict)

        avg_loss = np.mean(round_losses)
        loss_history.append(avg_loss)

        # 进度打印
        if (comm_round + 1) % 10 == 0 or comm_round == 0:
            print(f"  Round {comm_round + 1}/{num_rounds} - Avg Train Loss: {avg_loss:.4f}")

    # --- 6. 最终评估 ---
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

    # 保存Loss历史
    pd.DataFrame(loss_history, columns=['loss']).to_csv(
        os.path.join(exp_dir, 'results', 'loss_history.csv'), index_label='round'
    )

    # 保存摘要文本
    save_summary_report(exp_dir, all_metrics,
                        {'MAE': avg_mae, 'RMSE': avg_rmse, 'MAE_std': std_mae, 'RMSE_std': std_rmse})
    print(f"实验完成。 Avg MAE: {avg_mae:.4f} (±{std_mae:.4f}), Avg RMSE: {avg_rmse:.4f} (±{std_rmse:.4f})")

    return {
        'Dataset': file_name,
        'Algorithm': agg_name,
        'Params': str(agg_config),
        'MAE': avg_mae,
        'RMSE': avg_rmse,
        'MAE_std': std_mae, 'RMSE_std': std_rmse
    }


def main():
    # 读取基础配置
    base_config = load_config('config/config.yaml')

    files_plan = [
        {'path': 'data/batch-1.xlsx', 'win': 50, 'pre': 200},
        {'path': 'data/batch-2.xlsx', 'win': 50, 'pre': 200},
        {'path': 'data/batch-3.xlsx', 'win': 50, 'pre': 200},
        {'path': 'data/batch-4.xlsx', 'win': 100, 'pre': 500},
        {'path': 'data/batch-5.xlsx', 'win': 100, 'pre': 500},
    ]

    agg_strategies = [
        { 'name': 'fedavg', 'pfl_enabled': True, 'display_name': 'PFL_FedAvg'},
        # 普通联邦聚合
        { 'name': 'fedavg', 'pfl_enabled': False, 'display_name': 'Global_FedAvg'},
        # 解决Non-IID
        {'name': 'fedprox', 'mu': 0.01},
        # 动量加速
        {'name': 'fedavgm', 'beta': 0.9, 'server_lr': 1.0},
        # 自适应优化
        {'name': 'fedadam', 'beta1': 0.9, 'beta2': 0.99, 'server_lr': 0.01},
        {'name': 'fedyogi', 'beta1': 0.9, 'beta2': 0.99, 'server_lr': 0.01},
        {'name': 'fedadagrad', 'beta1': 0.9, 'server_lr': 0.01},
        # 鲁棒聚合
        {'name': 'fedmedian'},
    ]

    target_model = 'xpatch'

    timestamp = datetime.datetime.now().strftime("%m%d-%H%M")
    base_save_dir = base_config['results']['save_dir_prefix']

    parent_dir_name = f"exp_2_{timestamp}"
    parent_dir = os.path.join(base_save_dir, parent_dir_name)
    os.makedirs(parent_dir, exist_ok=True)
    print(f"本次所有实验结果将保存在: {parent_dir}")

    summary_results = []
    total_experiments = len(files_plan) * len(agg_strategies)
    current_idx = 0

    print(f"计划执行 {total_experiments} 个实验任务...")

    for plan in files_plan:
        file_path = plan['path']

        if not os.path.exists(file_path):
            print(f"跳过: 文件 {file_path} 不存在")
            continue

        for agg_config in agg_strategies:
            current_idx += 1
            try:
                result = run_fl_experiment(
                    file_path=file_path,
                    agg_config=agg_config,
                    model_name=target_model,
                    window_size=plan['win'],
                    pre_len=plan['pre'],
                    base_config=base_config,
                    parent_dir=parent_dir
                )

                if result:
                    summary_results.append(result)

            except KeyboardInterrupt:
                print("\n用户中断实验。正在输出已完成的结果...")
                # 保存已跑完的结果
                if summary_results:
                    df = pd.DataFrame(summary_results)
                    summary_path = os.path.join(parent_dir, 'exp_2_summary.csv')
                    df.to_csv(summary_path, index=False)
                return
            except Exception as e:
                print(f"\n!!! 实验出错 ({file_path} - {agg_config['name']}): {e}")
                import traceback
                traceback.print_exc()

    # --- 输出最终汇总表 ---
    if summary_results:
        df = pd.DataFrame(summary_results)
        print("\n" + "#" * 60)
        print("联邦算法对比汇总报告")
        print("#" * 60)
        print(df.to_string(index=False))
        print("#" * 60)

        # 保存汇总CSV
        summary_path = os.path.join(parent_dir, 'exp_2_summary.csv')
        df.to_csv(summary_path, index=False)
        print(f"汇总表格已保存至: {summary_path}")
    else:
        print("没有完成任何实验。")


if __name__ == '__main__':
    main()
