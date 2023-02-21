import torch
import numpy as np


def params_count(model):
    count = 0
    for p in model.parameters():
        c = 1
        for i in range(p.dim()):
            c *= p.size(i)
        count += c
    return count


def euclidean_dist(x, y):
    dist = x-y
    dist = torch.pow(dist, 2).sum(1, keepdim=True)
    dist = dist.sqrt()
    return dist.cuda()


def l2norm_r(X):
    norm = torch.pow(X, 2).sum(dim=0, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X


def inner_conduct(x):
    temp = torch.zeros(x.shape[0])
    for i in range(x.shape[0]):
        temp[i] = torch.dot(x[i],x[i])
    return temp.cuda()


def cos_sim(x, y, mask):
    sim = torch.nn.functional.cosine_similarity(x[:, None, :], y[None, :, :], dim=2)
    sim = sim.masked_fill_(mask, 0)
    im_neg = sim.max(1)[1]
    te_neg = sim.max(0)[1]
    return im_neg, te_neg


def e_sim(x, y):
    im_neg = []
    te_neg = []
    for i in range(len(x)):
        im_scores = torch.nn.functional.pairwise_distance(x[i], y)
        im_scores[i] = 0
        im_neg.append(im_scores.max(0)[1])
    for j in range(len(y)):
        te_scores = torch.nn.functional.pairwise_distance(x, y[j])
        te_scores[j] = 0
        te_neg.append(te_scores.max(0)[1])
    return torch.tensor(im_neg), torch.tensor(te_neg)


def calcul_pairwise_loss(score, optim, image_embs, text_embs):
    size_v = score.size(0)
    size_t = score.size(1)
    diagonal = score.diag().view(size_v, 1)
    d1 = diagonal.expand_as(score)
    d2 = diagonal.t().expand_as(score)

    # compare every diagonal score to scores in its column
    cost_s = (optim['pairwise_margin'] + score - d1).clamp(min=0)
    # compare every diagonal score to scores in its row
    cost_im = (optim['pairwise_margin'] + score - d2).clamp(min=0)

    mask = torch.eye(score.size(0)) > .5
    if torch.cuda.is_available():
        mask = mask.cuda()
    cost_s = cost_s.masked_fill_(mask, 0)
    cost_im = cost_im.masked_fill_(mask, 0)

    cost_s = cost_s.max(1)[0]
    cost_im = cost_im.max(0)[0]
    return (cost_s.sum()+cost_im.sum()) * optim['pairwise_loss_lambda']


def calcul_triplet_loss(direct_embs, target_embs, optim, target_dist):
    cosine_matrix = torch.nn.functional.cosine_similarity(direct_embs[:, None, :], direct_embs[None, :, :], dim=2)
    mask = torch.eye(cosine_matrix.size(0)) > .5
    if torch.cuda.is_available():
        mask = mask.cuda()
    cosine_matrix = cosine_matrix.masked_fill_(mask, 0)
    idx_1 = cosine_matrix.max(dim=1)[1]
    idx_2 = cosine_matrix.min(dim=1)[1]

    with torch.no_grad():
        dist1_te = euclidean_dist(direct_embs, direct_embs[idx_1])  # image_embs is h^A, image_embs[idx_1] is h^A_j
        dist2_te = euclidean_dist(direct_embs, direct_embs[idx_2])  # image_embs[idx_2] is h^A_i , yes

        b = (dist1_te <= dist2_te)
        delta_single = torch.ones_like(b)
        delta_single[b] = -1
        delta_single = delta_single.float()  # [batch_size, 1]
        delta_single = torch.squeeze(delta_single)  # [batch_size]

    d_i = inner_conduct(target_dist - target_dist[idx_1])
    d_j = inner_conduct(target_dist - target_dist[idx_2])
    diff_cap = d_i - d_j
    diff_cap_norm = torch.sigmoid(diff_cap.clamp(min=-5.0, max=5.0))
    dist_loss1 = (optim['triplet_margin'] + delta_single * diff_cap_norm).clamp(min=0)
    return torch.sum(dist_loss1) * optim['triplet_loss_lambda']

