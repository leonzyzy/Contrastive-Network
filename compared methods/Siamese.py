import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

# Define a custom dataset for handling three modalities
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

# Define the Siamese Network with three modalities
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        
        self.shared_fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        self.fc1 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        self.fc3 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        self.distance = nn.PairwiseDistance(p=2)

    def forward(self, input1, input2, input3):
        output1 = self.fc1(input1)
        output2 = self.fc2(input2)
        output3 = self.fc3(input3)
        
        combined = torch.cat((output1, output2, output3), dim=1)
        combined = self.shared_fc(combined)
        
        return combined

# Define contrastive loss function
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.pairwise_distance = nn.PairwiseDistance(p=2)
    
    def forward(self, output1, output2, label):
        euclidean_distance = self.pairwise_distance(output1, output2)
        loss = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                          (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss

# Example training function
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    
    for (data1, data2, data3), labels in dataloader:
        data1, data2, data3, labels = data1.to(device), data2.to(device), data3.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        output1 = model(data1)
        output2 = model(data2)
        output3 = model(data3)
        
        loss = criterion(output1, output2, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * data1.size(0)
    
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

