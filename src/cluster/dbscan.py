from sklearn.cluster import DBSCAN
import numpy as np
from .utils import vectorize_client_params


class DBSCANClustering:
    def __init__(self, config):
        # DBSCAN 特有参数，建议放入 config.yaml 的 clustering 下
        # eps: 两个样本被视为邻居的最大距离
        # min_samples: 一个核心点所需的最小邻居数
        self.eps = config.get('eps', 0.5)
        self.min_samples = config.get('min_samples', 2)

    def run(self, client_parts_dict, cluster_on):
        client_ids, X = vectorize_client_params(client_parts_dict, cluster_on)

        # 使用余弦距离通常对高维参数向量效果更好，也可以用 'euclidean'
        db = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric='cosine')
        labels = db.fit_predict(X)

        # DBSCAN 会产生 -1 标签表示噪声点(outliers)
        # 策略：将噪声点单独作为一个簇，或者分配给最近的簇
        # 这里简单处理：将 -1 映射为最大的 label ID + 1
        if -1 in labels:
            max_label = max(labels)
            outlier_label = max_label + 1
            labels = [x if x != -1 else outlier_label for x in labels]
            print(f"[Cluster] DBSCAN 发现 {list(labels).count(outlier_label)} 个离群点，已归入 Cluster {outlier_label}")

        num_clusters = len(set(labels))
        assignments = {cid: int(label) for cid, label in zip(client_ids, labels)}

        print(f"[Cluster] DBSCAN 自动识别出 {num_clusters} 个簇")
        return assignments, num_clusters, None
