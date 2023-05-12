import torch
import torch.nn as nn
from math import log

# This loss is just for us to test, not training loss.

def TotalLoss(preds, ground_truth, projections, lambda_value, temperature=0.7, weight=None,
                convex=None, device='cuda'):
    # add weight if data is imbalanced, use ratio.
    if weight is not None:
        cross_entropy = torch.nn.CrossEntropyLoss(weight=weight)
    else:
        cross_entropy = torch.nn.CrossEntropyLoss()
    ce_loss = cross_entropy(preds, ground_truth)
    supcon_loss = SupCon(projections, ground_truth)

    # if using a convex linear combination
    if convex is not None:
      if lambda_value >= 0 and lambda_value <=1:
        loss = torch.tensor(1 - lambda_value, device=device) * ce_loss + torch.tensor(lambda_value,
                                                                                      device=device) * supcon_loss
    else:
        loss = torch.tensor(lambda_value, device=device) * supcon_loss + ce_loss
    return loss


class SupCon(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupCon, self).__init__()
        self.temperature = temperature

    def forward(self, projections, targets):
        device = torch.device("cuda") if projections.is_cuda else torch.device("cpu")

        dot_product_tempered = torch.mm(projections, projections.T) / self.temperature
        # Minus max for numerical stability with exponential. Same done in cross entropy. Epsilon added to avoid log(0)
        exp_dot_tempered = (
                torch.exp(dot_product_tempered - torch.max(dot_product_tempered, dim=1, keepdim=True)[0]) + 1e-5
        )

        # mask pos to ensure label contrastive
        mask_similar_class = (targets.unsqueeze(1).repeat(1, targets.shape[0]) == targets).to(device)
        mask_anchor_out = (1 - torch.eye(exp_dot_tempered.shape[0])).to(device)
        mask_combined = mask_similar_class * mask_anchor_out
        cardinality_per_samples = torch.sum(mask_combined, dim=1)

        # compute probability
        log_prob = -torch.log(exp_dot_tempered / (torch.sum(exp_dot_tempered * mask_anchor_out, dim=1, keepdim=True)))
        loss = torch.sum(log_prob * mask_combined, dim=1) / cardinality_per_samples
        loss_mean = torch.mean(loss)

        return loss_mean
