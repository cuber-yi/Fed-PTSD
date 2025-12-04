from sklearn.cluster import KMeans
from .utils import vectorize_client_params

class KMeansClustering:
    def __init__(self, config):
        self.num_clusters = config.get('num_clusters', 2)
        self.seed = config.get('seed', 42)

    def run(self, client_parts_dict):
        client_ids, X = vectorize_client_params(client_parts_dict)

        if len(client_ids) < self.num_clusters:
            print(f"[Cluster] 警告: 客户端数量 ({len(client_ids)}) 少于簇数量 ({self.num_clusters})")
            return {cid: 0 for cid in client_ids}, 1, None

        kmeans = KMeans(n_clusters=self.num_clusters, random_state=self.seed, n_init=10)
        labels = kmeans.fit_predict(X)

        assignments = {cid: int(label) for cid, label in zip(client_ids, labels)}
        return assignments, self.num_clusters, None
