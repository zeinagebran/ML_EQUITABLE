from torch_geometric.data import Data
import torch

def build_graphs_from_subject(coherence, wpli):
    """
    Transforme les matrices [9, C, C] en liste de graphes PyG (un par bande de fréquence).

    Args:
        coherence (np.ndarray): Matrices de cohérence [9, C, C] → features des noeuds
        wpli (np.ndarray): Matrices wPLI [9, C, C] → features des arêtes

    Returns:
        List[Data]: Liste de 9 graphes PyG
    """
    graphs = []
    num_bands, num_nodes, _ = coherence.shape

    for b in range(num_bands):
        # Feature de nœud : moyenne des connexions pour chaque nœud
        node_features = coherence[b].mean(axis=1, keepdims=True)  # [C, 1]

        # Construction du graphe
        edge_index = []
        edge_attr = []

        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    edge_index.append([i, j])
                    edge_attr.append([wpli[b, i, j]])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # [2, E]
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32)  # [E, 1]
        x = torch.tensor(node_features, dtype=torch.float32)  # [C, 1]

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data.batch = torch.zeros(x.size(0), dtype=torch.long)  # tous les nœuds dans un seul graphe
        graphs.append(data)

    return graphs
