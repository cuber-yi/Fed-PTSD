import torch
import numpy as np


def vectorize_client_params(client_parts_dict, cluster_on='trend'):
    """
    将客户端的参数字典转换为向量矩阵。
    返回: (client_ids 列表, client_vectors numpy矩阵)
    """
    client_ids = []
    client_vectors = []

    for client_id, parts in client_parts_dict.items():
        client_ids.append(client_id)
        vector_parts = []

        if cluster_on == 'both':
            # 合并 'seasonal' 和 'trend'
            targets = ['seasonal', 'trend']
        else:
            # 仅使用单一组件
            targets = [cluster_on]

        for target in targets:
            if target not in parts:
                continue
            for param in parts[target].values():
                vector_parts.append(param.data.view(-1))

        if not vector_parts:
            # 防止某些情况下为空
            continue

        # 合并所有张量为一个大向量
        full_vector = torch.cat(vector_parts).cpu().numpy()
        client_vectors.append(full_vector)

    return client_ids, np.array(client_vectors)
