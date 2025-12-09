from .kmeans import KMeansClustering
from .gmm import GMMClustering
from .spectral import SpectralClusteringStrategy
from .agglomerative import AgglomerativeClusteringStrategy

# 注册表
CLUSTERING_STRATEGIES = {
    'kmeans': KMeansClustering,
    'gmm': GMMClustering,
    'spectral': SpectralClusteringStrategy,
    'agglomerative': AgglomerativeClusteringStrategy
}


def get_clustering_strategy(method_name, config):
    """
    根据名称获取聚类策略实例
    """
    strategy_class = CLUSTERING_STRATEGIES.get(method_name.lower())
    if not strategy_class:
        raise ValueError(f"Unknown clustering method: {method_name}. Available: {list(CLUSTERING_STRATEGIES.keys())}")

    return strategy_class(config)
