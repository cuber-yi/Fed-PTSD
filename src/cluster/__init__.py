from .kmeans import KMeansClustering
from .gmm import GMMClustering
from .spectral import SpectralClusteringStrategy
from .finch import FinchClusteringStrategy
from .leiden import LeidenClusteringStrategy
from .xleiden import XLeidenClustering
from .xfinch import XFinchClustering

# 注册表
CLUSTERING_STRATEGIES = {
    'kmeans': KMeansClustering,
    'gmm': GMMClustering,
    'spectral': SpectralClusteringStrategy,
    'finch': FinchClusteringStrategy,
    'leiden': LeidenClusteringStrategy,
    'xleiden': XLeidenClustering,  # 注册 xLeiden
    'xfinch': XFinchClustering     # 注册 xFINCH
}


def get_clustering_strategy(method_name, config):
    """
    根据名称获取聚类策略实例
    """
    strategy_class = CLUSTERING_STRATEGIES.get(method_name.lower())
    if not strategy_class:
        raise ValueError(f"Unknown clustering method: {method_name}. Available: {list(CLUSTERING_STRATEGIES.keys())}")

    return strategy_class(config)
