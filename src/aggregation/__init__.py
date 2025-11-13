"""
联邦学习聚合策略模块
"""

from .fed_avg import FedAvg
from .fed_prox import FedProx

__all__ = [
    'FedAvg',
    'FedProx',
]

