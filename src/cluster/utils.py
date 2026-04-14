import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def vectorize_client_params(client_parts_dict, use_pca=True, n_components=0.95, trend_weight=1.0, seasonal_weight=1.0):
    client_ids = []
    client_vectors = []

    sorted_ids = sorted(client_parts_dict.keys())
    targets = ['seasonal', 'trend']

    for client_id in sorted_ids:
        parts = client_parts_dict[client_id]
        vector_parts = []

        for target in targets:
            if target not in parts:
                continue

            need_scaling = False
            scale_val = 1.0
            if target == 'trend' and trend_weight != 1.0:
                scale_val = np.sqrt(trend_weight)
                need_scaling = True
            elif target == 'seasonal' and seasonal_weight != 1.0:
                scale_val = np.sqrt(seasonal_weight)
                need_scaling = True

            # 将所有参数展平并拼接
            for param in parts[target].values():
                flat_param = param.data.view(-1)

                if need_scaling:
                    flat_param = flat_param * scale_val

                vector_parts.append(flat_param)

        if not vector_parts:
            continue

        full_vector = torch.cat(vector_parts).cpu().numpy()
        client_vectors.append(full_vector)
        client_ids.append(client_id)

    X = np.array(client_vectors)

    X_reduced = X
    if use_pca and len(client_ids) > 1:
        n_samples = X.shape[0]
        n_comp = min(n_samples - 1, 3)

        if n_samples > 5:
            pca = PCA(n_components=n_components)
        else:
            pca = PCA(n_components=n_comp)

        try:
            X_reduced = pca.fit_transform(X)
        except Exception as e:
            print(f"  [Cluster Utils] PCA 失败，使用原始维度: {e}")
            X_reduced = X

    return client_ids, X_reduced


def save_cluster_visualization(exp_dir, round_idx, client_parts_dict, client_clusters, method_name):
    client_ids, _ = vectorize_client_params(client_parts_dict, use_pca=False)

    vectors = []
    targets = ['seasonal', 'trend']
    for cid in client_ids:
        parts = client_parts_dict[cid]
        vp = []
        for t in targets:
            if t in parts:
                for p in parts[t].values():
                    vp.append(p.data.view(-1))
        vectors.append(torch.cat(vp).cpu().numpy())
    X = np.array(vectors)

    if len(client_ids) < 2:
        return

    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    labels = [client_clusters.get(cid, 0) for cid in client_ids]

    plot_data = pd.DataFrame({
        'client_id': client_ids,
        'pca_x': X_2d[:, 0],
        'pca_y': X_2d[:, 1],
        'cluster': labels
    })
    data_save_path = os.path.join(exp_dir, 'plots', f'cluster_data_round_{round_idx}.csv')
    plot_data.to_csv(data_save_path, index=False)

    plt.figure(figsize=(10, 8))
    unique_labels = sorted(list(set(labels)))

    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        mask = np.array(labels) == label
        plt.scatter(X_2d[mask, 0], X_2d[mask, 1],
                    label=f'Cluster {label}',
                    s=100, alpha=0.7, edgecolors='k')

        for x, y, cid in zip(X_2d[mask, 0], X_2d[mask, 1], np.array(client_ids)[mask]):
            plt.text(x, y, str(cid), fontsize=9, ha='right', va='bottom')

    plt.title(f'Cluster Visualization (Round {round_idx}) - {method_name}')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    img_save_path = os.path.join(exp_dir, 'plots', f'cluster_view_round_{round_idx}.png')
    plt.savefig(img_save_path)
    plt.close()

    print(f"  [Vis] 聚类可视化已保存: {img_save_path}")
