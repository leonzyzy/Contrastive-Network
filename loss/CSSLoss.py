import torch 
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning import losses

# cross-subject-similarity loss/ supervised contrastive loss
class CSSLoss(nn.Module):
    def __init__(self, temperature):
        super(CSSLoss, self).__init__()
        self.temperature = temperature

    def forward(self, feature_vectors, labels):
        # Normalize feature vectors
        feature_vectors_normalized = F.normalize(feature_vectors, p=2, dim=1)
        # Compute logits
        logits = torch.div(
            torch.matmul(
                feature_vectors_normalized, torch.transpose(feature_vectors_normalized, 0, 1)
            ),
            self.temperature,
        )
        return losses.NTXentLoss(temperature=self.temperature)(logits, torch.squeeze(labels))
    
