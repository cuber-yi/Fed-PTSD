from sklearn.cluster import AgglomerativeClustering
from .utils import vectorize_client_params

class AgglomerativeClusteringStrategy:
    def __init__(self, config):
        self.num_clusters = config.get('num_clusters', 2)
        self.linkage = config.get('linkage', 'ward')

    def run(self, client_parts_dict):
        client_ids, X = vectorize_client_params(client_parts_dict)

        if len(client_ids) < self.num_clusters:
            return {cid: 0 for cid in client_ids}, 1, None

        # Agglomerative Clustering
        agg = AgglomerativeClustering(
            n_clusters=self.num_clusters,
            linkage=self.linkage
        )
        labels = agg.fit_predict(X)

        assignments = {cid: int(label) for cid, label in zip(client_ids, labels)}
        return assignments, self.num_clusters, None
