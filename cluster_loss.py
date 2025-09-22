import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class ProtoContrastLoss(nn.Module):
    def __init__(self, batch_size, temperature):
        super().__init__()
        self.B  = batch_size
        self.t  = 0.2
        self.temperature = 0.5

        self.gumbel_tau = 0.8
        self.ce_sum = nn.CrossEntropyLoss(reduction="sum")
        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    @staticmethod
    def _sample_one_hot(probs, tau):
        return F.gumbel_softmax((probs + 1e-12).log(), tau=tau, hard=True)

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, proj_feat1, proj_feat2, z_i, z_j, w_i, prototypes, loss_epoch):

        B, K, D = z_i.shape
        N = 2 * B
        self.K = K

        # Auxiliary per-instance loss
        z = torch.cat((proj_feat1, proj_feat2), dim=0)

        sim = torch.matmul(z, z.T) / self.temperature
        sim_i_j = torch.diag(sim, self.B)
        sim_j_i = torch.diag(sim, -self.B)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        prototypes = F.normalize(prototypes, dim=1)

        # Cluster-prior hyperspherical embedding
        idx = self._sample_one_hot(w_i, self.gumbel_tau).argmax(1)   # (B,)

        zi = z_i[torch.arange(B), idx]              # (B,D)
        zj = z_j[torch.arange(B), idx]              # (B,D)
        anchors      = torch.cat([zi, zj], 0)                        # (N,D)
        counterparts = torch.cat([zj, zi], 0)                        # (N,D)

        proto_sel   = prototypes[idx].repeat(2, 1)                   # (N,D)
        pos_proto   = (anchors * proto_sel).sum(1, keepdim=True) / self.t
        pos_view    = (anchors * counterparts).sum(1, keepdim=True) / self.t

        neg_bank = [[] for _ in range(K)]
        mask_notk = idx.unsqueeze(1).ne(torch.arange(K).to(positive_samples.device))    # (B,K)
        for k in range(K):
            valid = mask_notk[:, k]                     # (B,)
            neg_bank[k].append(z_i[valid, k])
            neg_bank[k].append(z_j[valid, k])
            neg_bank[k] = torch.cat(neg_bank[k], 0) if len(neg_bank[k]) else None

        neg_logits_list = []
        for a in range(N):
            k_a  = idx[a % B].item()
            negs = neg_bank[k_a]
            if negs is None:
                neg_logits_list.append(anchors.new_empty(0))
                continue
            neg_logits = anchors[a] @ negs.T / self.t  # (M_k,)
            neg_logits_list.append(neg_logits)

        max_neg = max(l.size(0) for l in neg_logits_list)
        logits  = anchors.new_full((N, 2 + max_neg), -1e9)  # (N, 2+max_neg)
        for a, neg in enumerate(neg_logits_list):
            logits[a, 0] = pos_proto[a]
            logits[a, 1] = pos_view[a]
            if neg.numel():
                logits[a, 2 : 2 + neg.size(0)] = neg

        log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)  # (N,L)
        loss_per_anchor = -torch.logsumexp(log_prob[:, :2], dim=1)

        loss += loss_per_anchor.mean()

        return loss

class ClusterLoss(nn.Module):
    def __init__(self, class_num, temperature):
        super(ClusterLoss, self).__init__()
        self.class_num = class_num
        self.temperature = temperature

        self.mask = self.mask_correlated_clusters(class_num)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_clusters(self, class_num):
        N = 2 * class_num
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(class_num):
            mask[i, class_num + i] = 0
            mask[class_num + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, c_i, c_j):
        p_i = c_i.sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
        p_j = c_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        ne_loss = ne_i + ne_j

        c_i = c_i.t()
        c_j = c_j.t()
        N = 2 * self.class_num
        c = torch.cat((c_i, c_j), dim=0)

        sim = self.similarity_f(c.unsqueeze(1), c.unsqueeze(0)) / self.temperature
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_clusters = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss + ne_loss


def compute_sorted_cosine_similarities(data, labels, mu):
    """
    For each category:
        1. Extract features from all samples in that category and compute the mean vector
        2. Calculate the cosine similarity between each sample and the mean vector
        3. Return the similarity scores sorted from highest to lowest along with their corresponding indices (within that category)
    """
    unique_labels = torch.unique(labels)
    results = {}

    for label in unique_labels:
        mask = (labels == label)
        class_features = data[mask]

        mean_vector = mu[label,:]

        cos_sim = F.cosine_similarity(class_features, mean_vector, dim=1)

        sorted_sim, sorted_indices = torch.sort(cos_sim, descending=True)

        original_indices = torch.nonzero(mask, as_tuple=True)[0]
        sorted_original_indices = original_indices[sorted_indices]

        results[int(label.item())] = {
            'sorted_similarities': sorted_sim.detach().cpu(),  # Sorted similarity
            'sorted_indices_within_class': sorted_indices.detach().cpu(),  # Index within this category
            'sorted_original_indices': sorted_original_indices.detach().cpu()  # Indexing in raw data
        }

    return results