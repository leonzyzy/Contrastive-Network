import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Define a custom dataset for triplets with multimodal inputs
class MultiModalTripletDataset(Dataset):
    def __init__(self, anchor_data1, anchor_data2, anchor_data3,
                       positive_data1, positive_data2, positive_data3,
                       negative_data1, negative_data2, negative_data3,
                       transform=None):
        self.anchor_data1 = anchor_data1
        self.anchor_data2 = anchor_data2
        self.anchor_data3 = anchor_data3
        self.positive_data1 = positive_data1
        self.positive_data2 = positive_data2
        self.positive_data3 = positive_data3
        self.negative_data1 = negative_data1
        self.negative_data2 = negative_data2
        self.negative_data3 = negative_data3
        self.transform = transform

    def __len__(self):
        return len(self.anchor_data1)

    def __getitem__(self, idx):
        anchor1 = self.anchor_data1[idx]
        anchor2 = self.anchor_data2[idx]
        anchor3 = self.anchor_data3[idx]
        positive1 = self.positive_data1[idx]
        positive2 = self.positive_data2[idx]
        positive3 = self.positive_data3[idx]
        negative1 = self.negative_data1[idx]
        negative2 = self.negative_data2[idx]
        negative3 = self.negative_data3[idx]

        if self.transform:
            anchor1 = self.transform(anchor1)
            anchor2 = self.transform(anchor2)
            anchor3 = self.transform(anchor3)
            positive1 = self.transform(positive1)
            positive2 = self.transform(positive2)
            positive3 = self.transform(positive3)
            negative1 = self.transform(negative1)
            negative2 = self.transform(negative2)
            negative3 = self.transform(negative3)

        return (anchor1, anchor2, anchor3), (positive1, positive2, positive3), (negative1, negative2, negative3)

# Define the Triplet Network model for multimodal inputs
class MultiModalTripletNetwork(nn.Module):
    def __init__(self):
        super(MultiModalTripletNetwork, self).__init__()
        
        self.feature_extractor1 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        self.feature_extractor2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        self.feature_extractor3 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

    def forward(self, anchor1, anchor2, anchor3,
                     positive1, positive2, positive3,
                     negative1, negative2, negative3):
        anchor_out1 = self.feature_extractor1(anchor1)
        anchor_out2 = self.feature_extractor2(anchor2)
        anchor_out3 = self.feature_extractor3(anchor3)
        
        positive_out1 = self.feature_extractor1(positive1)
        positive_out2 = self.feature_extractor2(positive2)
        positive_out3 = self.feature_extractor3(positive3)
        
        negative_out1 = self.feature_extractor1(negative1)
        negative_out2 = self.feature_extractor2(negative2)
        negative_out3 = self.feature_extractor3(negative3)
        
        return (anchor_out1, anchor_out2, anchor_out3), (positive_out1, positive_out2, positive_out3), (negative_out1, negative_out2, negative_out3)

# Define the Triplet Loss function
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.pairwise_distance = nn.PairwiseDistance(p=2)

    def forward(self, anchor_out, positive_out, negative_out):
        anchor_out1, anchor_out2, anchor_out3 = anchor_out
        positive_out1, positive_out2, positive_out3 = positive_out
        negative_out1, negative_out2, negative_out3 = negative_out
        
        positive_distance1 = self.pairwise_distance(anchor_out1, positive_out1)
        negative_distance1 = self.pairwise_distance(anchor_out1, negative_out1)
        
        positive_distance2 = self.pairwise_distance(anchor_out2, positive_out2)
        negative_distance2 = self.pairwise_distance(anchor_out2, negative_out2)
        
        positive_distance3 = self.pairwise_distance(anchor_out3, positive_out3)
        negative_distance3 = self.pairwise_distance(anchor_out3, negative_out3)
        
        losses1 = torch.clamp(positive_distance1 - negative_distance1 + self.margin, min=0.0)
        losses2 = torch.clamp(positive_distance2 - negative_distance2 + self.margin, min=0.0)
        losses3 = torch.clamp(positive_distance3 - negative_distance3 + self.margin, min=0.0)
        
        return torch.mean(losses1 + losses2 + losses3)

# Example training function
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    
    for (anchor1, anchor2, anchor3), (positive1, positive2, positive3), (negative1, negative2, negative3) in dataloader:
        anchor1, anchor2, anchor3 = anchor1.to(device), anchor2.to(device), anchor3.to(device)
        positive1, positive2, positive3 = positive1.to(device), positive2.to(device), positive3.to(device)
        negative1, negative2, negative3 = negative1.to(device), negative2.to(device), negative3.to(device)
        
        optimizer.zero_grad()
        
        anchor_out, positive_out, negative_out = model(
            anchor1, anchor2, anchor3,
            positive1, positive2, positive3,
            negative1, negative2, negative3
        )
        
        loss = criterion(anchor_out, positive_out, negative_out)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * anchor1.size(0)
    
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss



