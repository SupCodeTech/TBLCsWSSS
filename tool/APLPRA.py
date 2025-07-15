import torch
import torch.nn.functional as F

def compute_affinity_cosine(feature_map):
    """
    feature_map: [C, H, W, D] (feature vector at each voxel)
    return: Affinity Matrix [N, N]  where N = H*W*D
    """
    C, H, W, D = feature_map.shape
    N = H * W * D
    feat = feature_map.view(C, -1).transpose(0, 1)  # [N, C]

    # Normalize each vector
    feat_norm = F.normalize(feat, p=2, dim=1)  # [N, C]

    # Cosine similarity matrix
    affinity = torch.matmul(feat_norm, feat_norm.transpose(0, 1))  # [N, N]

    # Row-wise normalization (sum=1)
    affinity = affinity / (affinity.sum(dim=1, keepdim=True) + 1e-8)
    return affinity  # [N, N]

def focal_affinity_loss(W_aff_pred, Y_aff):
    """
    W_aff_pred: predicted affinity matrix [N, N]
    Y_aff: ground truth affinity label [N, N]  (0/1)
    """

    pos_mask = (Y_aff == 1).float()
    neg_mask = (Y_aff == 0).float()

    num_pos = pos_mask.sum() + 1e-6
    num_neg = neg_mask.sum() + 1e-6

    pos_loss = torch.log(1 + torch.exp(-W_aff_pred)) * pos_mask
    neg_loss = torch.log(1 + torch.exp(W_aff_pred)) * neg_mask

    loss = (pos_loss.sum() / num_pos) + (neg_loss.sum() / num_neg)
    return loss

def region_wise_affinity_propagation(M, W_aff, regions, eta=0.4):
    """
    M: Activation map [H, W, D, C]
    W_aff: Affinity matrix [H, W, D, H, W, D]  (or a function to extract local)
    regions: list of dict with keys {'i':, 'j':, 'k':, 'h':, 'w':, 'd':}
    eta: propagation strength
    return: Refined Activation Map [H, W, D, C]
    """
    H, W, D, C = M.shape
    M_out = torch.zeros_like(M)
    weight_sum = torch.zeros(H, W, D, device=M.device)

    for region in regions:
        i0, j0, k0 = region['i'], region['j'], region['k']
        h_r, w_r, d_r = region['h'], region['w'], region['d']

        M_r = M[i0:i0+h_r, j0:j0+w_r, k0:k0+d_r, :]  # [h_r, w_r, d_r, C]
        W_r = W_aff[i0:i0+h_r, j0:j0+w_r, k0:k0+d_r,
                    i0:i0+h_r, j0:j0+w_r, k0:k0+d_r]  # [N_r, N_r]

        N_r = h_r * w_r * d_r
        M_r_flat = M_r.reshape(N_r, C)

        D_r = torch.diag(W_r.sum(dim=1) + 1e-6)
        T_r = torch.linalg.inv(D_r) @ (W_r ** eta)  # [N_r, N_r]

        M_r_prop = torch.matmul(T_r, M_r_flat).reshape(h_r, w_r, d_r, C)
        M_out[i0:i0+h_r, j0:j0+w_r, k0:k0+d_r, :] += M_r_prop
        weight_sum[i0:i0+h_r, j0:j0+w_r, k0:k0+d_r] += 1.0

    # Normalize overlapping regions
    M_out = M_out / (weight_sum.unsqueeze(-1) + 1e-6)
    return M_out

def split_regions(H, W, D, num_regions=15, overlap=5):
    # 简单划分示例，可以根据实际需求调整
    step_h = H // 3
    step_w = W // 3
    step_d = D // 2
    regions = []
    for i in range(0, H, step_h):
        for j in range(0, W, step_w):
            for k in range(0, D, step_d):
                h_r = min(step_h + overlap, H - i)
                w_r = min(step_w + overlap, W - j)
                d_r = min(step_d + overlap, D - k)
                regions.append({'i': i, 'j': j, 'k': k, 'h': h_r, 'w': w_r, 'd': d_r})
                if len(regions) >= num_regions:
                    return regions
    return regions
