from finch import FINCH
import numpy as np
from .utils import vectorize_client_params


class XFinchClustering:
    def __init__(self, config):
        # -1 取最宏观划分层级，0 取最细粒度层级
        self.level = config.get('finch_level', -1)
        # 解耦参数的先验权重
        self.trend_weight = config.get('trend_weight', 0.8)
        self.seasonal_weight = config.get('seasonal_weight', 0.2)

    def run(self, client_parts_dict):
        # 1. 在“分解感知”加权空间中获取特征矩阵
        client_ids, X = vectorize_client_params(
            client_parts_dict,
            trend_weight=self.trend_weight,
            seasonal_weight=self.seasonal_weight
        )
        num_clients = len(client_ids)

        if num_clients < 2:
            print(f"[Cluster] xFINCH: 客户端数量不足 ({num_clients})，回退到单簇。")
            return {cid: 0 for cid in client_ids}, 1, None

        # 2. 运行无参数层次聚类 (FINCH)
        # c: 各层聚类分配, num_clust: 各层簇数量
        try:
            c, num_clust, req_c = FINCH(X)
        except Exception as e:
            print(f"[Cluster Error] xFINCH 聚类失败: {e}")
            return {cid: 0 for cid in client_ids}, 1, None

        # 3. 确定自适应层级
        max_level = c.shape[1] - 1
        level_idx = min(abs(self.level), max_level) if self.level >= 0 else max(c.shape[1] + self.level, 0)

        labels = c[:, level_idx]
        n_clusters = num_clust[level_idx]

        assignments = {cid: int(label) for cid, label in zip(client_ids, labels)}
        print(
            f"  [Zero-Prior xFINCH] 第一近邻层次划分完成 -> 自动提取 {n_clusters} 个簇 (Depth: {level_idx}/{max_level})")

        return assignments, n_clusters, None
