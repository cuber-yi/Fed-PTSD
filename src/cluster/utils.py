import torch
import numpy as np
from sklearn.decomposition import PCA


def vectorize_client_params(client_parts_dict, use_pca=True, n_components=0.95):
    """
    将客户端参数转化为向量，并可选地应用 PCA 降维
    """
    client_ids = []
    client_vectors = []

    targets = ['seasonal', 'trend']

    for client_id, parts in client_parts_dict.items():
        client_ids.append(client_id)
        vector_parts = []

        for target in targets:
            if target not in parts:
                continue
            for param in parts[target].values():
                vector_parts.append(param.data.view(-1))

        if not vector_parts:
            continue

        # 拼接该客户端的所有参数
        full_vector = torch.cat(vector_parts).cpu().numpy()
        client_vectors.append(full_vector)

    X = np.array(client_vectors)

    if use_pca and len(client_ids) > 1:
        n_samples = X.shape[0]
        n_comp = min(n_samples - 1, 3)

        if n_samples > 5:
            pca = PCA(n_components=n_components)
        else:
            pca = PCA(n_components=n_comp)

        try:
            X_pca = pca.fit_transform(X)
            print(f"  [Cluster Utils] PCA 降维: {X.shape[1]} -> {X_pca.shape[1]} dims")
            return client_ids, X_pca
        except Exception as e:
            print(f"  [Cluster Utils] PCA 失败，使用原始维度: {e}")
            return client_ids, X

    return client_ids, X
