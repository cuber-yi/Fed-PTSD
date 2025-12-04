import torch
import numpy as np


def vectorize_client_params(client_parts_dict):
    client_ids = []
    client_vectors = []

    targets = ['seasonal', 'trend']

    for client_id, parts in client_parts_dict.items():
        client_ids.append(client_id)
        vector_parts = []

        for target in targets:
            if target not in parts:
                continue
            for param in parts[target].values():
                vector_parts.append(param.data.view(-1))

        if not vector_parts:
            continue

        full_vector = torch.cat(vector_parts).cpu().numpy()
        client_vectors.append(full_vector)

    return client_ids, np.array(client_vectors)
