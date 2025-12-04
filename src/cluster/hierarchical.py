from sklearn.cluster import AgglomerativeClustering
from .utils import vectorize_client_params


class HierarchicalClusteringStrategy:
    def __init__(self, config):
        self.num_clusters = config.get('num_clusters', 2)
        # linkage 可选: 'ward', 'complete', 'average', 'single'
        self.linkage = config.get('linkage', 'ward')

    def run(self, client_parts_dict, cluster_on):
        client_ids, X = vectorize_client_params(client_parts_dict, cluster_on)

        if len(client_ids) < self.num_clusters:
            print(f"[Cluster] Warning: Not enough clients for Hierarchical Clustering. Fallback to cluster 0.")
            return {cid: 0 for cid in client_ids}, 1, None

        # 凝聚层次聚类
        hc = AgglomerativeClustering(n_clusters=self.num_clusters, linkage=self.linkage)
        labels = hc.fit_predict(X)

        assignments = {cid: int(label) for cid, label in zip(client_ids, labels)}
        return assignments, self.num_clusters, None
