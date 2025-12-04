from sklearn.mixture import GaussianMixture
from .utils import vectorize_client_params

class GMMClustering:
    def __init__(self, config):
        self.num_clusters = config.get('num_clusters', 2)
        self.seed = config.get('seed', 42)
        self.covariance_type = config.get('covariance_type', 'diag')

    def run(self, client_parts_dict):
        client_ids, X = vectorize_client_params(client_parts_dict)

        if len(client_ids) < self.num_clusters:
            return {cid: 0 for cid in client_ids}, 1, None

        gmm = GaussianMixture(
            n_components=self.num_clusters,
            random_state=self.seed,
            covariance_type=self.covariance_type
        )
        gmm.fit(X)

        probs = gmm.predict_proba(X)
        labels = gmm.predict(X)

        assignments = {cid: int(label) for cid, label in zip(client_ids, labels)}

        weights = {}
        for idx, cid in enumerate(client_ids):
            weights[cid] = probs[idx]

        return assignments, self.num_clusters, weights