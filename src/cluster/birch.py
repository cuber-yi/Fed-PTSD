from sklearn.cluster import Birch
from .utils import vectorize_client_params


class BirchClustering:
    def __init__(self, config):
        self.num_clusters = config.get('num_clusters', 2)
        # threshold:以此半径新建子簇，越小分得越细
        self.threshold = config.get('threshold', 0.5)

    def run(self, client_parts_dict, cluster_on):
        client_ids, X = vectorize_client_params(client_parts_dict, cluster_on)

        if len(client_ids) < self.num_clusters:
            return {cid: 0 for cid in client_ids}, 1, None

        brc = Birch(n_clusters=self.num_clusters, threshold=self.threshold)
        labels = brc.fit_predict(X)

        assignments = {cid: int(label) for cid, label in zip(client_ids, labels)}
        return assignments, self.num_clusters, None
