import torch
import torch.nn as nn

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        # Normalize features
        features = nn.functional.normalize(features, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Extract positive pairs
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float()
        mask = mask.fill_diagonal_(0)  # Exclude self-similarity
        
        # Calculate the loss
        exp_sim = torch.exp(similarity_matrix) * (1 - mask)
        pos_sim = torch.exp(similarity_matrix) * mask
        
        # Compute loss for each sample
        loss = -torch.log(pos_sim.sum(dim=1) / exp_sim.sum(dim=1) + 1e-8)
        
        return loss.mean()


