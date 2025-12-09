from sklearn.cluster import SpectralClustering
import numpy as np
from .utils import vectorize_client_params

class SpectralClusteringStrategy:
    def __init__(self, config):
        self.num_clusters = config.get('num_clusters', 2)
        self.seed = config.get('seed', 42)
        self.affinity = config.get('affinity', 'nearest_neighbors')
        self.n_neighbors = config.get('n_neighbors', 5)
        self.gamma = config.get('gamma', 1.0)

    def run(self, client_parts_dict):
        client_ids, X = vectorize_client_params(client_parts_dict)
        num_clients = len(client_ids)

        if num_clients < self.num_clusters:
            print(f"[Cluster] 警告: 客户端数量 ({num_clients}) 少于簇数量 ({self.num_clusters})，回退到单簇。")
            return {cid: 0 for cid in client_ids}, 1, None


        effective_n_neighbors = self.n_neighbors

        if self.affinity == 'nearest_neighbors':
            recommended_neighbors = max(int(num_clients * 0.6), 2)

            if effective_n_neighbors < recommended_neighbors:
                effective_n_neighbors = recommended_neighbors
                effective_n_neighbors = min(effective_n_neighbors, num_clients - 1)

            if effective_n_neighbors >= num_clients:
                effective_n_neighbors = max(1, num_clients - 1)

        spectral = SpectralClustering(
            n_clusters=self.num_clusters,
            eigen_solver=None,
            random_state=self.seed,
            n_init=20,
            gamma=self.gamma,
            affinity=self.affinity,
            n_neighbors=effective_n_neighbors,
            assign_labels='discretize',
            n_jobs=-1
        )

        try:
            labels = spectral.fit_predict(X)
        except Exception as e:
            print(f"[Cluster Error] 谱聚类失败: {e}。")
            return {cid: 0 for cid in client_ids}, 1, None

        assignments = {cid: int(label) for cid, label in zip(client_ids, labels)}
        return assignments, self.num_clusters, None
