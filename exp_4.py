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


def run_privacy_experiment(dataset_group_name, files, privacy_setting, base_config, parent_dir):
    config = copy.deepcopy(base_config)


    config['model']['name'] = 'xpatch'
    config['model']['pfl_enabled'] = True
    config['aggregation'] = {'name': 'fedavgm', 'beta': 0.9, 'server_lr': 1.0}
    config['clustering']['enabled'] = True
    config['clustering']['method'] = 'kmeans'
    config['clustering']['num_clusters'] = 3
    WARMUP_ROUNDS = config['clustering'].get('warmup_rounds', 10)
    RECLUSTER_INTERVAL = config['clustering'].get('recluster_every_n_rounds', 5)

    noise_level_name = privacy_setting['name']
    if privacy_setting['enabled']:
        config['privacy']['enabled'] = True
        config['privacy']['clipping_norm'] = 1.5
        config['privacy']['noise_sigma'] = privacy_setting['sigma_dict']
    else:
        config['privacy']['enabled'] = False

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

    exp_sub_name = f"{dataset_group_name}_{noise_level_name}"
    print(f"\n{'=' * 80}")
    print(f" >>> 执行隐私实验: {dataset_group_name} | {noise_level_name}")
    print(f" >>> 策略: Warmup={WARMUP_ROUNDS} Rounds, Re-cluster Interval={RECLUSTER_INTERVAL}")
    print(f"{'=' * 80}")

    exp_dir = os.path.join(parent_dir, exp_sub_name)
    os.makedirs(os.path.join(exp_dir, 'results'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'plots'), exist_ok=True)

    set_seed(config.get('seed', 42))
    g = torch.Generator()
    g.manual_seed(config.get('seed', 42))

    client_dataloaders = setup_clients_multi_file_by_sheet(
        file_paths=files,
        window_size=config['data']['window_size'],
        pre_len=config['data']['pre_len'],
        batch_size=config['federation']['batch_size'],
        max_capacity=config['data']['max_capacity'],
        generator=g
    )

    if not client_dataloaders: return None

    num_clients = len(client_dataloaders)
    clients = [Client(client_id=i, dataloader=dl, config=config, device=device)
               for i, dl in enumerate(client_dataloaders)]
    server = Server(config=config, num_total_clients=num_clients, device=device)

    num_rounds = config['federation']['num_rounds']

    last_round_client_parts = {}

    for comm_round in range(num_rounds):
        is_warmup_done = (comm_round >= WARMUP_ROUNDS)
        is_recluster_time = ((comm_round - WARMUP_ROUNDS) % RECLUSTER_INTERVAL == 0)

        if config['clustering']['enabled']:
            if is_warmup_done and is_recluster_time:
                print(f"  [Cluster] Round {comm_round}: Triggering Clustering (Warmup Done)...")

                if last_round_client_parts:
                    server.recluster_clients(last_round_client_parts)
                    print(f"  [Cluster] Result: {server.client_clusters}")
                else:
                    print("  [Cluster] Initializing clustering with Round 0 pre-trained weights...")
                    pass

            elif not is_warmup_done:
                if comm_round == 0:
                    print(f"  [Cluster] Warmup Phase Started ({WARMUP_ROUNDS} rounds)")

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
            print(f"  Round {comm_round + 1}/{num_rounds} Avg Loss: {np.mean(list(client_losses_dict.values())):.4f}")

    print("正在评估...")
    all_metrics = []
    for client in clients:
        final_parts = server.get_global_model_parts(client.client_id)
        client.set_global_model(copy.deepcopy(final_parts))
        mae, rmse = client.evaluate(save_dir=exp_dir)
        all_metrics.append({'client_id': client.client_id, 'MAE': mae, 'RMSE': rmse})

    avg_mae = np.mean([m['MAE'] for m in all_metrics])
    avg_rmse = np.mean([m['RMSE'] for m in all_metrics])

    sigma_val = 0.0
    if privacy_setting['enabled']:
        sigma_val = privacy_setting['sigma_dict']['trend']

    return {
        'Privacy_Mode': noise_level_name,
        'Sigma_Trend': sigma_val,
        'MAE': avg_mae,
        'RMSE': avg_rmse
    }


def main():
    base_config = load_config('config/config.yaml')

    target_files = ['data/batch-1.xlsx', 'data/batch-2.xlsx', 'data/batch-3.xlsx']

    privacy_levels = [
        {'name': '0.020_Noise', 'enabled': True, 'sigma_dict': {'seasonal': 0.004, 'trend': 0.02}},
        {'name': '0.030_Noise', 'enabled': True, 'sigma_dict': {'seasonal': 0.005, 'trend': 0.03}},
        {'name': '0.040_Noise', 'enabled': True, 'sigma_dict': {'seasonal': 0.008, 'trend': 0.04}},
        {'name': '0.080_Noise', 'enabled': True, 'sigma_dict': {'seasonal': 0.02, 'trend': 0.08}},
        {'name': '0.100_Noise', 'enabled': True, 'sigma_dict': {'seasonal': 0.02, 'trend': 0.1}},
    ]

    timestamp = datetime.datetime.now().strftime("%m%d-%H%M")
    parent_dir = os.path.join(base_config['results']['save_dir_prefix'], f"exp_4_{timestamp}")
    os.makedirs(parent_dir, exist_ok=True)


    print(f"开始隐私权衡实验，结果保存至: {parent_dir}")
    print(f"从配置文件读取聚类参数: Warmup={base_config['clustering'].get('warmup_rounds')}, "
          f"Interval={base_config['clustering'].get('recluster_every_n_rounds')}")

    results = []

    for setting in privacy_levels:
        try:
            res = run_privacy_experiment(
                dataset_group_name="XJTU",
                files=target_files,
                privacy_setting=setting,
                base_config=base_config,
                parent_dir=parent_dir
            )
            if res: results.append(res)
        except Exception as e:
            print(f"实验出错: {e}")
            import traceback
            traceback.print_exc()

    if results:
        df = pd.DataFrame(results)
        print("\n=== 隐私-性能权衡实验汇总 ===")
        print(df.to_string(index=False))
        csv_path = os.path.join(parent_dir, 'privacy_tradeoff_summary.csv')
        df.to_csv(csv_path, index=False)
        print(f"汇总数据已保存: {csv_path}")


if __name__ == '__main__':
    main()