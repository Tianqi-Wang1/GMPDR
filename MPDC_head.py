import torch.nn as nn
import torch
from torch.nn.functional import normalize
from torch.nn.functional import softmax
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self, input_dim, pen_dim, feature_dim, class_num):
        super(Network, self).__init__()

        self.input_dim = input_dim
        self.pen_dim = pen_dim
        self.feature_dim = feature_dim
        self.cluster_num = class_num
        self.instance_projector = nn.Sequential(
            nn.Linear(self.input_dim, self.pen_dim),
            nn.ReLU(),
            nn.Linear(self.pen_dim, self.feature_dim),
        )

        # Multi-head: one head per cluster
        self.instance_heads = nn.ModuleList([
            nn.Linear(self.input_dim, self.feature_dim)
            for _ in range(class_num)
        ])

        self.cluster_projector = nn.Sequential(
            nn.Linear(self.input_dim, self.pen_dim),
            nn.ReLU(),
            nn.Linear(self.pen_dim, self.cluster_num),
            nn.Softmax(dim=1)
        )

        self.mu = nn.Parameter(torch.randn(class_num, feature_dim))

    def forward(self, x_i, x_j, task_id, device):

        proj_feat1 = normalize(self.instance_projector(x_i), dim=1)
        z_i = [normalize(head(x_i), dim=1) for head in self.instance_heads]
        z_i = torch.stack(z_i, dim=1)  # shape: [B, K, D]

        proj_feat2 = normalize(self.instance_projector(x_j), dim=1)
        z_j = [normalize(head(x_j), dim=1) for head in self.instance_heads]
        z_j = torch.stack(z_j, dim=1)  # shape: [B, K, D]

        c_i = self.cluster_projector(x_i)
        c_j = self.cluster_projector(x_j)

        return proj_feat1, proj_feat2, z_i, z_j, c_i, c_j, self.mu

    def forward_cluster(self, x, task_id, device):
        c = self.cluster_projector(x)
        c = torch.argmax(c, dim=1)

        z = [normalize(head(x), dim=1) for head in self.instance_heads]
        z = torch.stack(z, dim=1)  # shape: [B, K, D]

        b_indices = torch.arange(z.size(0), device=z.device)
        z_selected = z[b_indices, c]  # shape: [B, D]

        return z_selected, c, self.mu

    def forward_score(self, x, task_id, prototype, covariance, device):
        mean_vector = normalize(self.mu, dim=1)

        c = self.cluster_projector(x)
        c = torch.argmax(c, dim=1)

        z = [normalize(head(x), dim=1) for head in self.instance_heads]
        z = torch.stack(z, dim=1)  # shape: [B, K, D]

        b_indices = torch.arange(z.size(0), device=z.device)
        z_selected = z[b_indices, c]  # shape: [B, D]

        cosine_sim = F.cosine_similarity(z_selected, mean_vector[c], dim=1)

        return cosine_sim, c

    def forward_threshold(self, x, task_id, threshold, device):
        mean_vector = normalize(self.mu, dim=1)

        c = self.cluster_projector(x)
        c = torch.argmax(c, dim=1)

        z = [normalize(head(x), dim=1) for head in self.instance_heads]
        z = torch.stack(z, dim=1)  # shape: [B, K, D]

        b_indices = torch.arange(z.size(0), device=z.device)
        z_selected = z[b_indices, c]  # shape: [B, D]

        cosine_sim = F.cosine_similarity(z_selected, mean_vector[c], dim=1)

        return cosine_sim, c