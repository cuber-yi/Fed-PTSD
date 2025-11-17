from sklearn.mixture import GaussianMixture
from .utils import vectorize_client_params


class GMMClustering:
    def __init__(self, config):
        self.num_clusters = config.get('num_clusters', 2)
        self.seed = config.get('seed', 42)

    def run(self, client_parts_dict, cluster_on):
        """
        执行 GMM 软聚类
        返回:
            assignments: {client_id: main_cluster_id} (用于推断时选择主模型)
            num_clusters: 簇数量
            weights: {client_id: [prob_0, prob_1, ...]} (用于聚合时加权)
        """
        client_ids, X = vectorize_client_params(client_parts_dict, cluster_on)

        if len(client_ids) < self.num_clusters:
            return {cid: 0 for cid in client_ids}, 1, None

        gmm = GaussianMixture(n_components=self.num_clusters, random_state=self.seed)
        gmm.fit(X)

        # 获取软概率权重
        probs = gmm.predict_proba(X)  # Shape: [n_clients, n_clusters]
        labels = gmm.predict(X)  # 硬分类标签（概率最大的那个）

        assignments = {cid: int(label) for cid, label in zip(client_ids, labels)}

        weights = {}
        for idx, cid in enumerate(client_ids):
            weights[cid] = probs[idx]

        return assignments, self.num_clusters, weights
