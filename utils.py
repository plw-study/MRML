import torch
import numpy as np


def euclidean_dist(x, y):
    dist = x-y
    dist = torch.pow(dist, 2).sum(1, keepdim=True)
    dist = dist.sqrt()
    return dist.cuda()


def inner_conduct(x):
    temp = torch.zeros(x.shape[0])
    for i in range(x.shape[0]):
        temp[i] = torch.dot(x[i],x[i])
    return temp.cuda()


def pairwise_loss_with_label(score, optim, labels):
    size_v = score.size(0)
    size_t = score.size(1)
    diagonal = score.diag().view(size_v, 1)
    d1 = diagonal.expand_as(score)
    d2 = diagonal.t().expand_as(score)

    # compare every diagonal score to scores in its column
    cost_s = (optim['pair_margin'] + score - d1).clamp(min=0)
    # compare every diagonal score to scores in its row
    cost_im = (optim['pair_margin'] + score - d2).clamp(min=0)

    mask = torch.eye(score.size(0)) > .5
    if torch.cuda.is_available():
        mask = mask.cuda()
    cost_s = cost_s.masked_fill_(mask, 0)
    cost_im = cost_im.masked_fill_(mask, 0)

    labels_expand = labels[:, 0].expand(size_v, size_v)
    label_mask = labels_expand ^ labels_expand.T
    label_mask = label_mask < 1

    cost_s = cost_s.masked_fill_(label_mask, 0)
    cost_im = cost_im.masked_fill_(label_mask, 0)

    cost_s = cost_s.max(1)[0]
    cost_im = cost_im.max(0)[0]

    return (cost_s.sum() + cost_im.sum()) * optim['pair_loss_lambda']


def triplet_loss_with_label(direct_embs, target_embs, optim, target_dist, labels):
    batch_size = labels.size(0)
    cosine_matrix = torch.nn.functional.cosine_similarity(direct_embs[:, None, :], direct_embs[None, :, :], dim=2)
    mask = torch.eye(cosine_matrix.size(0)) > .5
    if torch.cuda.is_available():
        mask = mask.cuda()
    cosine_matrix = cosine_matrix.masked_fill_(mask, 0)
    labels_expand = labels[:, 0].expand(batch_size, batch_size)
    label_mask = labels_expand ^ labels_expand.T
    label_mask = label_mask > 0
    idx_p = cosine_matrix.masked_fill(label_mask, 0).max(dim=1)[1]
    idx_n = cosine_matrix.masked_fill(~label_mask, 0).max(dim=1)[1]

    with torch.no_grad():
        dist1_te = euclidean_dist(direct_embs, direct_embs[idx_p])
        dist2_te = euclidean_dist(direct_embs, direct_embs[idx_n])

        b = (dist1_te <= dist2_te)
        delta_single = torch.ones_like(b)
        delta_single[b] = -1
        delta_single = delta_single.float()
        delta_single = torch.squeeze(delta_single)

    d_i = inner_conduct(target_dist - target_dist[idx_p])
    d_j = inner_conduct(target_dist - target_dist[idx_n])
    diff_cap = d_i - d_j
    diff_cap_norm = torch.sigmoid(diff_cap.clamp(min=-5.0, max=5.0))
    dist_loss1 = (optim['tri_margin'] + delta_single * diff_cap_norm).clamp(min=0)
    return torch.sum(dist_loss1) * optim['tri_loss_beta']
