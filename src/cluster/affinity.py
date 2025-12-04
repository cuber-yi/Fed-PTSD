from sklearn.cluster import AffinityPropagation
from .utils import vectorize_client_params


class AffinityPropagationClustering:
    def __init__(self, config):
        # damping: 阻尼系数 [0.5, 1.0)，用于避免数值震荡
        self.damping = config.get('damping', 0.5)
        self.seed = config.get('seed', 42)
        self.preference = config.get('preference', None)  # None 表示使用中位数

    def run(self, client_parts_dict, cluster_on):
        client_ids, X = vectorize_client_params(client_parts_dict, cluster_on)

        # AP 聚类自动确定簇数量
        ap = AffinityPropagation(damping=self.damping, random_state=self.seed, preference=self.preference)
        try:
            labels = ap.fit_predict(X)
        except Exception as e:
            print(f"[Cluster Error] Affinity Propagation failed: {e}. Fallback to Cluster 0.")
            return {cid: 0 for cid in client_ids}, 1, None

        num_clusters = len(set(labels))
        assignments = {cid: int(label) for cid, label in zip(client_ids, labels)}

        return assignments, num_clusters, None
