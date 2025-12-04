import torch
from collections import OrderedDict, defaultdict
from utils.model_utils import get_model_class
from client import _classify_xpatch_param
from src.aggregation.fed_avg import FedAvg
from src.aggregation.fed_prox import FedProx
from src.aggregation.fed_avgm import FedAvgM
from src.aggregation.fed_adam import FedAdam
from src.cluster import get_clustering_strategy


class Server:
    def __init__(self, config: dict, num_total_clients: int, device: torch.device):
        self.config = config
        self.device = device

        self.model_name = self.config['model']['name']
        self.model_params = self.config['model']['config']

        # --- 聚类相关 ---
        self.clustering_config = self.config.get('clustering', {})
        self.clustering_enabled = self.clustering_config.get('enabled', False)
        self.clustering_method = self.clustering_config.get('method', 'kmeans')

        if self.clustering_enabled:
            if get_clustering_strategy:
                self.cluster_strategy = get_clustering_strategy(self.clustering_method, self.clustering_config)
            else:
                raise ImportError("启用聚类失败：找不到 src.cluster 模块")
            self.num_clusters = self.clustering_config.get('num_clusters', 1)
        else:
            self.num_clusters = 1

        # 初始化时，所有客户端都在 cluster 0
        self.client_clusters = {i: 0 for i in range(num_total_clients)}
        # 软聚类权重缓存 (用于 GMM 等软聚类)
        self.client_weights = None

        # 初始化时，只有 cluster 0 的一个模型
        self.cluster_models = {}
        self.cluster_models[0] = self._create_new_model()

        # --- 聚合策略初始化 ---
        aggregator_name = config.get('aggregation', {}).get('name', 'fedavg').lower()
        self.aggregator_class = self._get_aggregator_class(aggregator_name)
        self.cluster_aggregators = {}

    def _create_new_model(self):
        """辅助函数：创建一个新的模型实例"""
        ModelClass = get_model_class(self.model_name)
        return ModelClass(**self.model_params).to(self.device)

    def _get_aggregator_class(self, name: str):
        """根据配置返回聚合策略类"""
        aggregators = {
            'fedavg': FedAvg,
            'fedprox': FedProx,
            'fedavgm': FedAvgM,
            'fedadam': FedAdam,
        }

        if name not in aggregators:
            print(f"[Warning] 未知聚合策略 '{name}', 使用默认 FedAvg")
            return FedAvg

        return aggregators[name]

    def _get_cluster_aggregator(self, cluster_id: int):
        """获取或创建指定 Cluster 的聚合器实例"""
        if cluster_id not in self.cluster_aggregators:
            # 实例化一个新的聚合器
            self.cluster_aggregators[cluster_id] = self.aggregator_class(self.config)
        return self.cluster_aggregators[cluster_id]

    def get_global_model_parts(self, client_id: int) -> dict:
        # 1. 找到该客户端所属的 cluster
        cluster_id = self.client_clusters.get(client_id, 0)
        # 2. 获取该 cluster 对应的模型
        model_to_use = self.cluster_models.get(cluster_id)
        if model_to_use is None:
            print(f"[Server Warning] Client {client_id} 的 cluster {cluster_id} 没有模型, 使用 cluster 0")
            model_to_use = self.cluster_models.get(0, self._create_new_model())

        # --- 拆分模型 ---
        pfl_enabled = self.config['model'].get('pfl_enabled', True)
        is_xpatch_pFL = (self.model_name.lower() == 'xpatch') and pfl_enabled

        if is_xpatch_pFL:
            parts = {'seasonal': OrderedDict(), 'trend': OrderedDict()}
            for name, param in model_to_use.named_parameters():
                part_name = _classify_xpatch_param(name)
                if part_name != 'personal':
                    parts[part_name][name] = param.data.clone()
            return parts
        else:
            # 如果 PFL 被禁用，下发完整模型参数
            return {'full_model': model_to_use.state_dict()}

    def aggregate_parameters(self, client_parts_dict: dict, client_losses_dict: dict):
        # --- 未启用聚类或硬聚类 ---
        if not self.clustering_enabled or self.client_weights is None:
            cluster_groups = defaultdict(list)
            cluster_losses = defaultdict(list)

            for client_id, parts in client_parts_dict.items():
                cluster_id = self.client_clusters.get(client_id, 0)
                cluster_groups[cluster_id].append(parts)
                cluster_losses[cluster_id].append(client_losses_dict.get(client_id, 0.0))

            # 对每个 Cluster 单独聚合
            for cluster_id, parts_list in cluster_groups.items():
                if not parts_list:
                    continue

                aggregator = self._get_cluster_aggregator(cluster_id)
                aggregated_parts = aggregator.aggregate(parts_list, self.device)
                self.update_global_model(aggregated_parts, cluster_id)
            return

        # --- 软聚类 ---
        for cluster_k in range(self.num_clusters):
            weighted_sum_parts = defaultdict(lambda: defaultdict(float))
            total_weight_k = 0.0
            count_contributors = 0
            # 遍历所有客户端进行加权
            for client_id, parts in client_parts_dict.items():
                # 获取 Client i 对 Cluster k 的权重
                w_ik = self.client_weights[client_id][cluster_k]

                if w_ik < 1e-6:
                    continue

                total_weight_k += w_ik
                count_contributors += 1

                for part_name in parts:
                    for key, param in parts[part_name].items():
                        val = param.data * w_ik
                        if isinstance(weighted_sum_parts[part_name][key], float):
                            weighted_sum_parts[part_name][key] = val
                        else:
                            weighted_sum_parts[part_name][key] += val

            if total_weight_k > 0:
                final_parts = OrderedDict()
                for part_name in weighted_sum_parts:
                    final_parts[part_name] = OrderedDict()
                    for key, val_tensor in weighted_sum_parts[part_name].items():
                        # 归一化
                        final_parts[part_name][key] = (val_tensor / total_weight_k).to(self.device)

                # 更新对应的 cluster 模型
                self.update_global_model(final_parts, cluster_k)
            else:
                print(f"  [Server Warning] Cluster {cluster_k} has 0 total weight. Skipping update.")

    def update_global_model(self, aggregated_parts: dict, cluster_id: int):
        if cluster_id not in self.cluster_models:
            print(f"[Server] Creating new model for Cluster {cluster_id}")
            self.cluster_models[cluster_id] = self._create_new_model()

        model_to_update = self.cluster_models[cluster_id]
        current_state_dict = model_to_update.state_dict()

        # 将聚合后的部分参数更新到完整模型状态中
        for part_name, params_dict in aggregated_parts.items():
            current_state_dict.update(params_dict)

        model_to_update.load_state_dict(current_state_dict)

    def recluster_clients(self, client_parts_dict: dict):
        if not self.clustering_enabled:
            return

        # --- 调用策略执行聚类 ---
        new_assignments, num_clusters_found, weights = self.cluster_strategy.run(client_parts_dict)
        self.client_clusters = new_assignments
        self.num_clusters = num_clusters_found
        self.client_weights = weights
        self._initialize_cluster_models(client_parts_dict)

    def _initialize_cluster_models(self, client_parts_dict):
        new_cluster_models = {}

        cluster_groups = defaultdict(list)
        for client_id, parts in client_parts_dict.items():
            cluster_id = self.client_clusters.get(client_id, 0)
            cluster_groups[cluster_id].append(parts)

        for cluster_id, parts_list in cluster_groups.items():
            if not parts_list:
                continue

            aggregator = self._get_cluster_aggregator(cluster_id)
            init_parts = aggregator.aggregate(parts_list, self.device)

            new_model = self._create_new_model()
            current_state_dict = new_model.state_dict()
            for part_name, params_dict in init_parts.items():
                current_state_dict.update(params_dict)
            new_model.load_state_dict(current_state_dict)

            new_cluster_models[cluster_id] = new_model

        # 填补空缺的 Cluster 模型
        for k in range(self.num_clusters):
            if k not in new_cluster_models:
                if k in self.cluster_models:
                    new_cluster_models[k] = self.cluster_models[k]
                else:
                    new_cluster_models[k] = self._create_new_model()

        self.cluster_models = new_cluster_models

    def get_aggregator_info(self):
        """获取聚合器信息"""
        target_agg = None
        if 0 in self.cluster_aggregators:
            target_agg = self.cluster_aggregators[0]
        elif self.cluster_aggregators:
            target_agg = list(self.cluster_aggregators.values())[0]

        if target_agg and hasattr(target_agg, 'get_weights_info'):
            return target_agg.get_weights_info()
        return None
