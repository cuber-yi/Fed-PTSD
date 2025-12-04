from .kmeans import KMeansClustering
from .dbscan import DBSCANClustering
from .gmm import GMMClustering
from .spectral import SpectralClusteringStrategy
from .hierarchical import HierarchicalClusteringStrategy
from .affinity import AffinityPropagationClustering
from .birch import BirchClustering

# 注册表
CLUSTERING_STRATEGIES = {
    'kmeans': KMeansClustering,
    'dbscan': DBSCANClustering,
    'gmm': GMMClustering,
    'spectral': SpectralClusteringStrategy,
    'hierarchical': HierarchicalClusteringStrategy,
    'affinity': AffinityPropagationClustering,
    'birch': BirchClustering
}


def get_clustering_strategy(method_name, config):
    """
    根据名称获取聚类策略实例
    """
    strategy_class = CLUSTERING_STRATEGIES.get(method_name.lower())
    if not strategy_class:
        raise ValueError(f"Unknown clustering method: {method_name}. Available: {list(CLUSTERING_STRATEGIES.keys())}")

    return strategy_class(config)
