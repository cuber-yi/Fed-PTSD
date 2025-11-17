import torch
from collections import OrderedDict, defaultdict
from utils.model_utils import get_model_class
from client import _classify_xpatch_param
import numpy as np
from sklearn.cluster import KMeans
import copy
from src.aggregation.fed_avg import FedAvg
from src.aggregation.fed_prox import FedProx


class Server:
    def __init__(self, config: dict, num_total_clients: int, device: torch.device):
        self.config = config
        self.device = device

        self.model_name = self.config['model']['name']
        self.model_params = self.config['model']['config']

        # --- 聚类相关 ---
        self.clustering_config = self.config.get('clustering', {})
        self.clustering_enabled = self.clustering_config.get('enabled', False)
        self.num_clusters = self.clustering_config.get('num_clusters', 1) if self.clustering_enabled else 1
        #    初始化时，所有客户端都在 cluster 0
        self.client_clusters = {i: 0 for i in range(num_total_clients)}
        #    初始化时，只有 cluster 0 的一个模型
        self.cluster_models = {}
        self.cluster_models[0] = self._create_new_model()

        # 初始化聚合策略
        aggregation_name = config.get('aggregation', {}).get('name', 'fedavg').lower()
        self.aggregator = self._get_aggregator(aggregation_name)

    def _create_new_model(self):
        """辅助函数：创建一个新的模型实例"""
        ModelClass = get_model_class(self.model_name)
        return ModelClass(**self.model_params).to(self.device)

    def _get_aggregator(self, name: str):
        """根据配置选择聚合策略"""
        aggregators = {
            'fedavg': FedAvg,
            'fedprox': FedProx,
        }

        if name not in aggregators:
            print(f"[Warning] 未知聚合策略 '{name}', 使用默认 FedAvg")
            name = 'fedavg'

        return aggregators[name](self.config)

    def get_global_model_parts(self, client_id: int) -> dict:
        # 1. 找到该客户端所属的 cluster
        cluster_id = self.client_clusters.get(client_id, 0)
        # 2. 获取该 cluster 对应的模型
        model_to_use = self.cluster_models.get(cluster_id)
        if model_to_use is None:
            print(f"[Server Warning] Client {client_id} 的 cluster {cluster_id} 没有模型, 使用 cluster 0")
            model_to_use = self.cluster_models[0]
        # --- 拆分模型 ---
        is_xpatch_pFL = self.model_name.lower() == 'xpatch'
        if is_xpatch_pFL:
            parts = {'seasonal': OrderedDict(), 'trend': OrderedDict()}
            for name, param in model_to_use.named_parameters():
                part_name = _classify_xpatch_param(name)
                if part_name != 'personal':
                    parts[part_name][name] = param.data.clone()
            return parts
        else:
            return {'full_model': model_to_use.state_dict()}

    def aggregate_parameters(self, client_parts_dict: dict, client_losses_dict: dict):
        """
        不再是聚合所有客户端，而是按 cluster 分组聚合。
        参数:
            client_parts_dict: {client_id: parts} 的字典
            client_losses_dict: {client_id: loss} 的字典
        """
        if not self.clustering_enabled:
            # --- 传统聚合 ---
            parts_list = list(client_parts_dict.values())
            losses_list = list(client_losses_dict.values())
            aggregated_parts = self.aggregator.aggregate(parts_list, self.device)
            self.update_global_model(aggregated_parts, cluster_id=0)  # 只更新 0 号模型
            return

        # --- 按 Cluster 分组 ---
        cluster_groups = defaultdict(list)
        cluster_losses = defaultdict(list)

        for client_id, parts in client_parts_dict.items():
            cluster_id = self.client_clusters[client_id]
            cluster_groups[cluster_id].append(parts)
            cluster_losses[cluster_id].append(client_losses_dict[client_id])

        # --- 对每个 Cluster 单独聚合 ---
        for cluster_id, parts_list in cluster_groups.items():
            if not parts_list:
                continue

            losses_list = cluster_losses[cluster_id]
            print(f"  Aggregating for Cluster {cluster_id} with {len(parts_list)} clients...")
            aggregated_parts = self.aggregator.aggregate(parts_list, self.device)
            self.update_global_model(aggregated_parts, cluster_id)

    def update_global_model(self, aggregated_parts: dict, cluster_id: int):
        if cluster_id not in self.cluster_models:
            print(f"[Server Error] 尝试更新不存在的 cluster {cluster_id} 模型")
            return

        model_to_update = self.cluster_models[cluster_id]
        current_state_dict = model_to_update.state_dict()

        for part_name, params_dict in aggregated_parts.items():
            current_state_dict.update(params_dict)

        model_to_update.load_state_dict(current_state_dict)

    def recluster_clients(self, client_parts_dict: dict):
        """
        根据客户端上传的参数，重新计算聚类。
        """
        if not self.clustering_enabled or self.num_clusters <= 1:
            return

        cluster_on = self.clustering_config.get('cluster_on', 'trend')
        print(f"\n[Server] Re-clustering {len(client_parts_dict)} clients based on '{cluster_on}' component...")

        client_ids = []
        client_vectors = []

        # 1. 将参数 "向量化" (Vectorize)
        for client_id, parts in client_parts_dict.items():
            client_ids.append(client_id)
            vector_parts = []

            if cluster_on == 'both':
                # 方案1: 合并 'seasonal' 和 'trend'
                for part_name in ['seasonal', 'trend']:
                    for param in parts[part_name].values():
                        vector_parts.append(param.data.view(-1))
            else:
                # 方案2: 仅使用 'trend' 或 'seasonal'
                if cluster_on not in parts:
                    print(f"[Server Warning] '{cluster_on}' not in client parts. Defaulting to 'trend'.")
                    cluster_on = 'trend'
                for param in parts[cluster_on].values():
                    vector_parts.append(param.data.view(-1))

            # 合并所有张量为一个大向量
            full_vector = torch.cat(vector_parts).cpu().numpy()
            client_vectors.append(full_vector)

        if not client_vectors:
            print("[Server] No client vectors to cluster.")
            return

        # 2. 执行 K-Means 聚类
        X = np.array(client_vectors)
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=self.config.get('seed', 42), n_init=10)
        new_labels = kmeans.fit_predict(X)

        # 3. 更新客户端的 cluster 分配
        self.client_clusters = {cid: label for cid, label in zip(client_ids, new_labels)}
        print(f"[Server] Clustering complete. New assignments: {self.client_clusters}")

        # 4. 初始化/重置 Cluster 模型
        self._initialize_cluster_models(client_parts_dict)

    def _initialize_cluster_models(self, client_parts_dict):
        print("[Server] Initializing/Resetting cluster models...")
        new_cluster_models = {}

        cluster_groups = defaultdict(list)
        for client_id, parts in client_parts_dict.items():
            cluster_id = self.client_clusters[client_id]
            cluster_groups[cluster_id].append(parts)

        for cluster_id, parts_list in cluster_groups.items():
            if not parts_list:
                continue

            init_parts = self.aggregator.aggregate(parts_list, self.device)
            new_model = self._create_new_model()

            current_state_dict = new_model.state_dict()
            for part_name, params_dict in init_parts.items():
                current_state_dict.update(params_dict)
            new_model.load_state_dict(current_state_dict)

            new_cluster_models[cluster_id] = new_model

        self.cluster_models = new_cluster_models
        print(f"[Server] {len(self.cluster_models)} cluster models are ready.")

    def get_aggregator_info(self):
        """获取聚合器信息"""
        if hasattr(self.aggregator, 'get_weights_info'):
            return self.aggregator.get_weights_info()
        return None
