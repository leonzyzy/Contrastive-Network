import torch
import torch.nn as nn
from torch.utils.data import Dataset

# define triplet dataset
class TripletDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        x1 = self.x[idx]
        y1 = self.y[idx]
        
        # randomly 
        y_temp = torch.where(torch.tensor(self.y) == torch.tensor(y1))[0]
        y_temp = y_temp[y_temp != idx]
        idx2 = torch.randperm(len(y_temp))[0]
        x2 = self.x[y_temp[idx2]]
        
        y_temp1 = torch.where(torch.tensor(self.y) != torch.tensor(y1))[0]
        y_temp1 = y_temp1[y_temp1 != idx]
        idx3 = torch.randperm(len(y_temp1))[0]
        x3 = self.x[y_temp1[idx3]]
        
        # change dtype to float32
        x1 = torch.tensor(x1, dtype=torch.float32)
        x2 = torch.tensor(x2, dtype=torch.float32)
        x3 = torch.tensor(x3, dtype=torch.float32)
        
        return x1, x2, x3, y1

# define network 
class TripletNet(nn.Module):
    def __init__(self):
        super(TripletNet, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(72, 32),
            nn.ReLU(),
            nn.Linear(32, 6)
        )
    def forward(self, x, pos, neg):
        x_vec = self.linear(x)
        x_pos = self.linear(pos)
        x_neg = self.linear(neg)
        
        return x_vec, x_pos, x_neg

# define triplet loss
class TripletLoss(nn.Module):
    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin
    def forward(self, x, pos, neg):
        pos_dist = torch.norm(x - pos, p=2, dim=1)
        neg_dist = torch.norm(x - neg, p=2, dim=1)
        loss = torch.mean(torch.relu(pos_dist - neg_dist + self.margin))
        
        return loss