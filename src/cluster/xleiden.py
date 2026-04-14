import numpy as np
from sklearn.neighbors import kneighbors_graph
import igraph as ig
import leidenalg
from .utils import vectorize_client_params


class XLeidenClustering:
    def __init__(self, config):
        # KNN 构图的邻居数量（决定拓扑图稀疏度）
        self.n_neighbors = config.get('n_neighbors', 5)
        # 解耦参数的先验权重 (默认赋予 Trend 更高权重)
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

        # 客户端数量过少则无需聚类
        if num_clients < 2:
            print(f"[Cluster] xLeiden: 客户端数量不足 ({num_clients})，回退到单簇。")
            return {cid: 0 for cid in client_ids}, 1, None

        # 2. 动态调整 KNN 参数并构建稀疏参数拓扑图
        effective_neighbors = min(self.n_neighbors, num_clients - 1)
        effective_neighbors = max(1, effective_neighbors)

        A = kneighbors_graph(X, n_neighbors=effective_neighbors, mode='distance', include_self=False)

        # 3. 将距离转换为亲和度权重 (距离越近，权重越大)
        sources, targets = A.nonzero()
        distances = A.data
        weights = 1.0 / (1.0 + distances)

        # 4. 构建 igraph 无向图
        g = ig.Graph(directed=False)
        g.add_vertices(num_clients)
        g.add_edges(list(zip(sources, targets)))
        g.es['weight'] = weights

        # 5. 执行 Leiden 算法最大化模块度，自适应发现群体
        try:
            partition = leidenalg.find_partition(
                g,
                leidenalg.ModularityVertexPartition,
                weights='weight'
            )
        except Exception as e:
            print(f"[Cluster Error] xLeiden 图社区发现失败: {e}")
            return {cid: 0 for cid in client_ids}, 1, None

        # 6. 解析图分区结果
        assignments = {}
        for cluster_id, node_indices in enumerate(partition):
            for node_idx in node_indices:
                assignments[client_ids[node_idx]] = cluster_id

        num_clusters = len(partition)
        print(f"  [FedGraph-xLeiden] 模块度优化完成 -> 自适应涌现 {num_clusters} 个客户端社区")

        return assignments, num_clusters, None
