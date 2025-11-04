import os
import matplotlib.pyplot as plt
import numpy as np

def save_client_results(save_dir: str, client_id: int, metrics: dict, y_true: np.ndarray, y_pred: np.ndarray):
    """
    保存单个客户端的评估指标和预测图。

    参数:
        save_dir (str): 结果保存的根目录。
        client_id (int): 客户端ID。
        metrics (dict): 包含 'MAE' 和 'RMSE' 的字典。
        y_true (np.ndarray): 真实值数组，用于绘图。
        y_pred (np.ndarray): 预测值数组，用于绘图。
    """
    # --- 1. 确保结果和绘图目录存在 ---
    results_path_dir = os.path.join(save_dir, 'results')
    plots_path_dir = os.path.join(save_dir, 'plots')
    os.makedirs(results_path_dir, exist_ok=True)
    os.makedirs(plots_path_dir, exist_ok=True)

    # --- 2. 保存评估指标到文件 ---
    results_file_path = os.path.join(results_path_dir, f'client_{client_id}_metrics.txt')
    with open(results_file_path, 'w') as f:
        f.write(f"MAE: {metrics['MAE']:.4f}\n")
        f.write(f"RMSE: {metrics['RMSE']:.4f}\n")

    # --- 3. 绘制并保存预测对比图 ---
    plt.figure(figsize=(10, 6))
    plt.plot(y_true, label='Ground Truth', marker='o')
    plt.plot(y_pred, label='Prediction', marker='x', linestyle='--')
    plt.title(f'Client {client_id} - Sample Prediction vs. Ground Truth')
    plt.xlabel('Prediction Horizon (Steps)')
    plt.ylabel('SOH')
    plt.legend()
    plt.grid(True)
    plot_file_path = os.path.join(plots_path_dir, f'client_{client_id}_prediction_plot.png')
    plt.savefig(plot_file_path)
    plt.close()


def save_summary_report(save_dir: str, all_metrics: list, avg_metrics: dict):
    """
    保存所有客户端的评估结果摘要。

    参数:
        save_dir (str): 结果保存的根目录。
        all_metrics (list): 包含所有客户端指标字典的列表。
        avg_metrics (dict): 包含平均 'MAE' 和 'RMSE' 的字典。
    """
    # --- 确保结果目录存在 ---
    results_path_dir = os.path.join(save_dir, 'results')
    os.makedirs(results_path_dir, exist_ok=True)

    # --- 写入摘要文件 ---
    summary_path = os.path.join(results_path_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        for metrics in all_metrics:
            f.write(f"Client {metrics['client_id']}: MAE = {metrics['MAE']:.4f}, RMSE = {metrics['RMSE']:.4f}\n")
        f.write(f"\nAverage: MAE = {avg_metrics['MAE']:.4f}, RMSE = {avg_metrics['RMSE']:.4f}\n")
    print(f"Summary saved to {summary_path}")
