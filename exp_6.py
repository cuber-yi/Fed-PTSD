import os
import torch
import numpy as np
import copy
import pandas as pd
import datetime
import yaml
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


def run_single_pfl_experiment(file_path, strategy_name, window_size, pre_len, base_config, parent_dir):
    """
    运行单个 pFL 实验：特定数据集 + 特定 pFL 策略
    """
    # --- 1. 动态构建配置 ---
    config = copy.deepcopy(base_config)

    # 强制设置模型为 xpatch (作为基座)
    config['model']['name'] = 'xpatch'
    config['model']['pfl_enabled'] = True  # 默认开启，用于 xPatch 自身的逻辑

    # 聚合策略固定为 FedAvgM (或其他你常用的)，因为 pFL 的核心在于 Client 端传什么
    config['aggregation'] = {'name': 'fedavgm', 'beta': 0.9, 'server_lr': 1.0}

    # 关闭隐私和聚类以控制变量
    config['privacy']['enabled'] = False
    config['clustering']['enabled'] = False

    # 设置数据维度
    config['data']['window_size'] = window_size
    config['data']['pre_len'] = pre_len
    if 'config' not in config['model']: config['model']['config'] = {}
    config['model']['config']['enc_in'] = config['data']['enc_in']
    config['model']['config']['pred_len'] = config['data']['pre_len']
    config['model']['config']['seq_len'] = config['data']['window_size']

    # 加载 xpatch.yaml 参数
    model_config_path = Path('config') / "xpatch.yaml"
    if model_config_path.exists():
        with open(model_config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
            if cfg and 'config' in cfg:
                config['model']['config'].update(cfg['config'])

    device = torch.device(config['data']['device'])
    file_name = os.path.basename(file_path)

    # 实验命名
    exp_sub_name = f"{file_name}_{strategy_name}"
    exp_dir = os.path.join(parent_dir, exp_sub_name)
    os.makedirs(os.path.join(exp_dir, 'results'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'models'), exist_ok=True)

    print(f"\n{'=' * 80}")
    print(f" >>> 执行 pFL 实验: Dataset=[{file_name}] | Strategy=[{strategy_name.upper()}]")
    print(f"{'=' * 80}")

    # --- 2. 初始化环境与数据 (参考 exp_2) ---
    set_seed(config.get('seed', 42))
    g = torch.Generator()
    g.manual_seed(config.get('seed', 42))

    # 使用 setup_clients_by_sheet (单文件模式)
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

    # --- 3. 初始化 Server & Client ---
    num_clients = len(client_dataloaders)
    clients = [Client(i, dl, config, device) for i, dl in enumerate(client_dataloaders)]
    server = Server(config, num_clients, device)

    # --- 4. 训练循环 ---
    num_rounds = config['federation']['num_rounds']

    for comm_round in range(num_rounds):
        client_parts_dict = {}
        client_losses = {}

        for client in clients:
            # A. 下发参数
            global_parts = server.get_global_model_parts(client.client_id)

            # 特殊处理 FedBN: 仅加载非 BN 参数
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

            # B. 本地训练
            if strategy_name == 'fedrep':
                # FedRep: 先训 Head 再训 Body
                loss = client.local_train_fedrep(head_epochs=5)
            else:
                # FedBN 和 xPatch 使用标准训练
                loss = client.local_train()

            # C. 获取上传参数
            # 注意：需确保 client.py 中已实现 get_parameters_by_strategy
            local_parts = client.get_parameters_by_strategy(strategy_name)

            client_parts_dict[client.client_id] = local_parts
            client_losses[client.client_id] = loss

        # D. 聚合
        server.aggregate_parameters(client_parts_dict, client_losses)

        if (comm_round + 1) % 10 == 0:
            avg_loss = np.mean(list(client_losses.values()))
            print(f"  Round {comm_round + 1}/{num_rounds} - Avg Train Loss: {avg_loss:.4f}")

    # --- 5. 评估 ---
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

    # 保存摘要
    save_summary_report(exp_dir, all_metrics, {'MAE': avg_mae, 'RMSE': avg_rmse})
    print(f"实验完成: {file_name} | {strategy_name} -> MAE={avg_mae:.4f}")

    return {
        'Dataset': file_name,
        'Strategy': strategy_name,
        'MAE': avg_mae,
        'RMSE': avg_rmse
    }


def main():
    base_config = load_config('config/config.yaml')

    # --- 1. 定义实验计划 (与 exp_2.py 保持一致) ---
    files_plan = [
        # {'path': 'data/batch-1.xlsx', 'win': 50, 'pre': 200},
        # {'path': 'data/batch-2.xlsx', 'win': 50, 'pre': 200},
        # {'path': 'data/batch-3.xlsx', 'win': 50, 'pre': 200},
        # 根据需要可以取消注释
        {'path': 'data/batch-4.xlsx', 'win': 100, 'pre': 500},
        {'path': 'data/batch-5.xlsx', 'win': 100, 'pre': 500},
    ]

    # --- 2. 定义对比策略 ---
    strategies = [
        'xpatch_pfl',  # 本文提出的方法 (Trend/Seasonal 分离)
        'fedrep',  # 基线: FedRep (共享 Body，本地 Head)
        'fedbn'  # 基线: FedBN (不上传 BN 层)
    ]

    # --- 3. 结果保存 ---
    timestamp = datetime.datetime.now().strftime("%m%d-%H%M")
    base_save_dir = base_config['results']['save_dir_prefix']
    parent_dir = os.path.join(base_save_dir, f"exp_6_{timestamp}")
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

    # --- 4. 汇总输出 ---
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
