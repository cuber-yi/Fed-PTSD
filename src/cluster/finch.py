from finch import FINCH
import numpy as np
from .utils import vectorize_client_params


class FinchClusteringStrategy:
    def __init__(self, config):
        self.config = config
        self.level = config.get('finch_level', -1)

    def run(self, client_parts_dict):
        client_ids, X = vectorize_client_params(client_parts_dict)
        num_clients = len(client_ids)

        # 客户端过少时无需聚类
        if num_clients < 2:
            print(f"[Cluster] FINCH: 客户端数量不足 ({num_clients})，回退到单簇。")
            return {cid: 0 for cid in client_ids}, 1, None

        # 运行 FINCH 算法
        c, num_clust, req_c = FINCH(X)

        # 确保 level 索引不越界
        max_level = c.shape[1] - 1
        level_idx = min(abs(self.level), max_level) if self.level >= 0 else max(c.shape[1] + self.level, 0)

        labels = c[:, level_idx]
        n_clusters = num_clust[level_idx]

        assignments = {cid: int(label) for cid, label in zip(client_ids, labels)}
        print(f"  [FINCH] 自动发现 {n_clusters} 个簇 (使用层级 {level_idx}/{max_level})")

        return assignments, n_clusters, None
