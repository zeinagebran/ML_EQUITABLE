import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool

class ConnectomeTokenizer(nn.Module):
    def __init__(self, in_channels=32, hidden_dim=64, out_dim=128):
        super(ConnectomeTokenizer, self).__init__()

        nn1 = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
        self.gnn = GINEConv(nn1, edge_dim=1)
        self.out_dim = out_dim

    def forward(self, list_of_graph_lists):
        """
        list_of_graph_lists: List of size [B], each element is a list of 9 PyG graphs (one per band)
        Returns: token_tensor of shape [B, 9, out_dim]
        """
        batch_size = len(list_of_graph_lists)
        all_tokens = []

        for graphs_per_sample in list_of_graph_lists:
            sample_tokens = []
            for g in graphs_per_sample:
                # Make batch dimension to use PyG
                g.batch = torch.zeros(g.x.size(0), dtype=torch.long)  # 1 sample = 1 graph
                x = self.gnn(g.x, g.edge_index, g.edge_attr) ## [N, out_dim] , N is the number of nodes in the graph.
                token = global_mean_pool(x, g.batch)  # Shape: [1, out_dim]
                sample_tokens.append(token)

            # Shape: [9, out_dim]
            sample_tokens = torch.cat(sample_tokens, dim=0).unsqueeze(0)  # Add batch dim
            all_tokens.append(sample_tokens)

        # Shape: [B, 9, out_dim]
        token_tensor = torch.cat(all_tokens, dim=0)
        return token_tensor
