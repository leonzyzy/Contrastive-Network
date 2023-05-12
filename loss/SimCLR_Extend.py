import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F

class SimCLR_Loss(nn.Module):
    def __init__(self, batch_size, temperature, n):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)
        self.num_modalities = n
        self.mask = self.mask_correlated_samples()

    def mask_correlated_samples(self):
        N = self.batch_size * self.num_modalities
        mask = torch.ones((N, N),dtype=bool)
        mask = mask.fill_diagonal_(0)
        
        for i in range(N):
            for j in range(N):
                if (j-i) % self.num_modalities == 0:
                    mask[i,j] = 0
                if (i-j) % self.num_modalities == 0:
                    mask[i,j] = 0
        return mask

    def forward(self, modalities):

        N = self.batch_size * self.num_modalities

        z = torch.cat(modalities, dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        # get all positive samples
        pos = []
        for i in range(1,N):
            if i*self.batch_size < N:
                sim_ij = torch.diag(sim, i*self.batch_size)
                sim_ji = torch.diag(sim, -i*self.batch_size)
                p = torch.cat((sim_ij, sim_ji), dim=0)
                pos.append(p)
        pos = torch.cat(pos, dim=0).reshape(N,-1)
        neg = sim[self.mask].reshape(N,-1)
        
        #SIMCLR
        labels = torch.from_numpy(np.array([0]*N)).reshape(-1).to(pos.device).long() #.float()
        
        logits = torch.cat((pos, neg), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        
        return loss
    
b = 50
cmc = SimCLR_Loss(b, 0.5, 3)
z1 = torch.randn(b,12)
z2 = torch.randn(b,12)
z3 = torch.randn(b,12)
modalities = [z1,z2,z3]
loss = cmc(modalities)
print(loss)
