import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Define a custom dataset for multimodal inputs
class MultiModalDataset(Dataset):
    def __init__(self, modality1_data, modality2_data, modality3_data, labels, transform=None):
        self.modality1_data = modality1_data
        self.modality2_data = modality2_data
        self.modality3_data = modality3_data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample1 = self.modality1_data[idx]
        sample2 = self.modality2_data[idx]
        sample3 = self.modality3_data[idx]
        label = self.labels[idx]

        if self.transform:
            sample1 = self.transform(sample1)
            sample2 = self.transform(sample2)
            sample3 = self.transform(sample3)

        return (sample1, sample2, sample3), label

# Define a network for modality-invariant contrastive learning
class ModalityInvariantNetwork(nn.Module):
    def __init__(self):
        super(ModalityInvariantNetwork, self).__init__()
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

    def forward(self, modality1, modality2, modality3):
        embedding1 = self.feature_extractor(modality1)
        embedding2 = self.feature_extractor(modality2)
        embedding3 = self.feature_extractor(modality3)
        return embedding1, embedding2, embedding3

# Define the contrastive loss function for modality-invariant learning
class ModalityInvariantContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ModalityInvariantContrastiveLoss, self).__init__()
        self.margin = margin
        self.pairwise_distance = nn.PairwiseDistance(p=2)

    def forward(self, embedding1, embedding2, embedding3, label):
        # Compute pairwise distances
        dist12 = self.pairwise_distance(embedding1, embedding2)
        dist13 = self.pairwise_distance(embedding1, embedding3)
        dist23 = self.pairwise_distance(embedding2, embedding3)
        
        # Contrastive loss
        loss = torch.mean(
            torch.clamp(dist12 - dist13 + self.margin, min=0.0) +
            torch.clamp(dist12 - dist23 + self.margin, min=0.0) +
            torch.clamp(dist13 - dist23 + self.margin, min=0.0)
        )
        return loss

# Example training function
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    
    for (modality1, modality2, modality3), labels in dataloader:
        modality1, modality2, modality3, labels = modality1.to(device), modality2.to(device), modality3.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        embeddings = model(modality1, modality2, modality3)
        embedding1, embedding2, embedding3 = embeddings
        
        loss = criterion(embedding1, embedding2, embedding3, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * modality1.size(0)
    
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


