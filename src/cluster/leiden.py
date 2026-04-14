from sklearn.neighbors import kneighbors_graph
import igraph as ig
import leidenalg
from .utils import vectorize_client_params


class LeidenClusteringStrategy:
    def __init__(self, config):
        # K-近邻构图时的邻居数量 (决定图的稀疏程度)
        self.n_neighbors = config.get('n_neighbors', 5)

    def run(self, client_parts_dict):
        client_ids, X = vectorize_client_params(client_parts_dict)
        num_clients = len(client_ids)

        if num_clients < 2:
            return {cid: 0 for cid in client_ids}, 1, None

        # 1. 动态调整邻居数量，防止 N 过小时报错
        effective_neighbors = min(self.n_neighbors, num_clients - 1)
        effective_neighbors = max(1, effective_neighbors)

        # 2. 构建稀疏 KNN 图 (获取距离矩阵)
        A = kneighbors_graph(X, n_neighbors=effective_neighbors, mode='distance', include_self=False)

        # 3. 将距离转化为相似度权重 (距离越小，权重越大)
        sources, targets = A.nonzero()
        distances = A.data
        weights = 1.0 / (1.0 + distances)

        # 4. 构建 igraph 图结构
        g = ig.Graph(directed=False)
        g.add_vertices(num_clients)
        g.add_edges(list(zip(sources, targets)))
        g.es['weight'] = weights

        # 5. 执行 Leiden 算法最大化模块度
        try:
            partition = leidenalg.find_partition(
                g,
                leidenalg.ModularityVertexPartition,
                weights='weight'
            )
        except Exception as e:
            print(f"[Cluster Error] Leiden 社区发现失败: {e}")
            return {cid: 0 for cid in client_ids}, 1, None

        # 6. 解析聚类结果
        assignments = {}
        for cluster_id, node_indices in enumerate(partition):
            for node_idx in node_indices:
                assignments[client_ids[node_idx]] = cluster_id

        num_clusters = len(partition)
        print(f"  [Leiden] 图社区自动发现 {num_clusters} 个簇 (KNN={effective_neighbors})")

        return assignments, num_clusters, None
