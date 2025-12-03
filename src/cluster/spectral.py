from sklearn.cluster import SpectralClustering
import numpy as np
from .utils import vectorize_client_params


class SpectralClusteringStrategy:
    def __init__(self, config):
        self.num_clusters = config.get('num_clusters', 2)
        self.seed = config.get('seed', 42)

        # 谱聚类特有参数
        # 'affinity': 构建相似度矩阵的方式，推荐 'nearest_neighbors' (基于KNN) 或 'rbf' (径向基函数)
        self.affinity = config.get('affinity', 'nearest_neighbors')
        # 'n_neighbors': 当 affinity='nearest_neighbors' 时使用
        self.n_neighbors = config.get('n_neighbors', 5)
        # 'gamma': 当 affinity='rbf' 时使用
        self.gamma = config.get('gamma', 1.0)

    def run(self, client_parts_dict, cluster_on):
        """
        执行谱聚类 (Spectral Clustering)
        """
        # 1. 向量化参数
        client_ids, X = vectorize_client_params(client_parts_dict, cluster_on)
        num_clients = len(client_ids)

        # 2. 边界检查：客户端数量过少
        if num_clients < self.num_clusters:
            print(f"[Cluster] 警告: 客户端数量 ({num_clients}) 少于簇数量 ({self.num_clusters})，回退到单簇。")
            return {cid: 0 for cid in client_ids}, 1, None

        # 3. 动态调整 n_neighbors (防止 n_neighbors >= num_clients)
        effective_n_neighbors = self.n_neighbors
        if self.affinity == 'nearest_neighbors' and self.n_neighbors >= num_clients:
            effective_n_neighbors = max(1, num_clients - 1)
            print(f"[Cluster] 调整 n_neighbors 为 {effective_n_neighbors} (因样本数限制)")

        # 4. 初始化并拟合模型
        # assign_labels='discretize' 通常比 'kmeans' 在初始化敏感度上更稳定
        spectral = SpectralClustering(
            n_clusters=self.num_clusters,
            eigen_solver=None,
            random_state=self.seed,
            n_init=10,
            gamma=self.gamma,
            affinity=self.affinity,
            n_neighbors=effective_n_neighbors,
            assign_labels='discretize',
            n_jobs=-1
        )

        try:
            labels = spectral.fit_predict(X)
        except Exception as e:
            print(f"[Cluster Error] 谱聚类失败: {e}。回退到默认 Cluster 0。")
            return {cid: 0 for cid in client_ids}, 1, None

        # 5. 格式化输出
        assignments = {cid: int(label) for cid, label in zip(client_ids, labels)}

        # 谱聚类是硬聚类，无概率权重
        return assignments, self.num_clusters, None
