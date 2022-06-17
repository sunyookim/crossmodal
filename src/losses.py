import torch
import torch.nn.functional as F

def cos_loss(out1, out2, ys=None, margin=None):
    return 1 - F.cosine_similarity(out1, out2).mean()


def tri_loss(out1, out2, ys, feat1, feat2, feat_penalties, sharing = False, margin=0.1):
    '''
    Args:
        out1: (batch_size, emb_dim)
        out2: (batch_size, emb_dim)
        feat1: list of intermediate outputs 1 (length: num of layers)
        feat2: list of intermediate outputs 2 (length: num of layers)
    '''

    out1 = out1/(torch.norm(out1,2, dim=1)[:, None])
    out2 = out2/(torch.norm(out2,2, dim=1)[:, None])
    y_counts = {}
    o1s = {}
    o2s = {}
    for o1, o2, y in zip(out1, out2, ys):
        y = y.cpu().item()
        if y in y_counts:
            y_counts[y] += 1
            # o1s[y] += o1
            # o2s[y] += o2
            o1s[y] = o1s[y] + o1
            o2s[y] = o2s[y] + o2
        else:
            y_counts[y] = 1
            o1s[y] = o1
            o2s[y] = o2
    avg_1s = []
    avg_2s = []
    for y in o1s:
        o1 = o1s[y] / y_counts[y]
        o2 = o2s[y] / y_counts[y]
        avg_1s.append(o1)
        avg_2s.append(o2)
    avg_1s = torch.stack(avg_1s) # (5, 128)
    avg_2s = torch.stack(avg_2s) # (5, 128)
    mat = torch.einsum('ij,kj->ik', avg_1s, avg_2s) # (5, 5)
    diag = torch.diag(mat) # (5)
    tri_loss = torch.sum(F.relu(margin + mat - diag.repeat(len(diag), 1)))/len(out1)
    
    sharing_loss = 0.0
    if sharing:
        for i in range(len(feat1)):
            f1 = feat1[i].reshape(feat1[i].shape[0], -1)
            f2 = feat2[i].reshape(feat2[i].shape[0], -1)

            sharing_loss += feat_penalties[i] * torch.sum((f1-f2).norm(dim=1, p=2)/f1.shape[1])

        sharing_loss = sharing_loss/len(out1)
    return tri_loss + sharing_loss

def euclidean_dist(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    
    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)

def prototypical_loss(input, target, n_support):
    '''
    Inspired by https://github.com/jakesnell/prototypical-networks/blob/master/protonets/models/few_shot.py
    Compute the barycentres by averaging the features of n_support
    samples for each class in target, computes then the distances from each
    samples' features to each one of the barycentres, computes the
    log_probability for each n_query samples for each one of the current
    classes, of appartaining to a class c, loss and accuracy are then computed
    and returned
    Args:
    - input: the model output for a batch of samples
    - target: ground truth for the above batch of samples
    - n_support: number of samples to keep in account when computing
      barycentres, for each one of the current classes
    '''
    target_cpu = target.to('cpu') # (75)
    input_cpu = input.to('cpu') # (75, 128)

    def supp_idxs(c):
        # FIXME when torch will support where as np
        return target_cpu.eq(c).nonzero()[:n_support].squeeze(1)

    # FIXME when torch.unique will be available on cuda too
    classes = torch.unique(target_cpu)
    n_classes = len(classes) # 5
    # FIXME when torch will support where as np
    # assuming n_query, n_target constants
    n_query = target_cpu.eq(classes[0].item()).sum().item() - n_support # 10

    support_idxs = list(map(supp_idxs, classes)) # 5

    prototypes = torch.stack([input_cpu[idx_list].mean(0) for idx_list in support_idxs]) # (5, 128)
    # FIXME when torch will support where as np
    query_idxs = torch.stack(list(map(lambda c: target_cpu.eq(c).nonzero()[n_support:], classes))).view(-1) # (50)

    query_samples = input.to('cpu')[query_idxs] # (50, 128)
    dists = euclidean_dist(query_samples, prototypes) # (50, 5)

    log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1) # (5, 10, 5)

    target_inds = torch.arange(0, n_classes)
    target_inds = target_inds.view(n_classes, 1, 1)
    target_inds = target_inds.expand(n_classes, n_query, 1).long()

    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    _, y_hat = log_p_y.max(2)
    acc_val = y_hat.eq(target_inds.squeeze(2)).float().mean()

    return loss_val,  acc_val