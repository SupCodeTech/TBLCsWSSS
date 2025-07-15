import torch
import torch.nn as nn
import torch.nn.functional as F

class SGCSA_GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(SGCSA_GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, A_hat, H):
        H = torch.matmul(A_hat, H)
        H = self.linear(H)
        H = F.relu(H)
        return H


class SGCSA_Module(nn.Module):
    def __init__(self, num_layers=2, in_features=1, hidden_features=64):
        super(SGCSA_Module, self).__init__()
        layers = []
        for i in range(num_layers):
            in_dim = in_features if i == 0 else hidden_features
            layers.append(SGCSA_GCNLayer(in_dim, hidden_features))
        self.gcn_layers = nn.ModuleList(layers)

    def forward(self, attn_matrix):
        S = attn_matrix.mean(dim=-1)  # [hwd, hwd]
        S_sym = S + S.T
        A = S_sym
        I = torch.eye(A.size(0), device=A.device)
        A_hat = A + I
        D_hat = torch.diag(torch.sum(A_hat, dim=1))
        D_hat_inv_sqrt = torch.linalg.inv(torch.sqrt(D_hat))
        A_norm = D_hat_inv_sqrt @ A_hat @ D_hat_inv_sqrt
        H = S_sym.unsqueeze(-1)
        for gcn in self.gcn_layers:
            H = gcn(A_norm, H)
        W_aff = H.squeeze(-1)
        return W_aff

