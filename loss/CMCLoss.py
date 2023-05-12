import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F

def device_as(t1, t2):
   """
   Moves t1 to the device of t2
   """
   return t1.to(t2.device)

# cross-modal-complementary loss
class CMCLoss(nn.Module):
    def __init__(self, batch_size, temperature, n):
        super(CMCLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.num_modalities = n
       
        # define similarity function
        self.similarity_f = nn.CosineSimilarity(dim=2)
        
        # define mask matrix
        self.mask = self.mask_correlated_matrix()

    def mask_correlated_matrix(self):
        """
        Returns a mask to zero out correlated samples
        """
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
        representations = torch.cat(modalities, dim=0)
        sim = self.similarity_f(representations.unsqueeze(1), representations.unsqueeze(0)) / self.temperature

        # get all samples
        denominator = device_as(self.mask, sim) * torch.exp(sim/self.temperature)
        
        # get positive samples
        positive_index = sim[self.mask.fill_diagonal_(1)==0]
        positive_samples = positive_index.reshape(denominator.shape[0], -1)
        nominator = torch.sum(torch.exp(positive_samples / self.temperature), dim = 1)

        # compute loss
        all_losses = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(all_losses) / N            
        return loss

if __name__ == 'main':
    b = 6
    cmc = CMCLoss(6, 0.5, 3)
    z1 = torch.randn(b,12)
    z2 = torch.randn(b,12)
    z3 = torch.randn(b,12)
    z4 = torch.randn(b,12)
    z5 = torch.randn(b,12)
    modalities = [z1,z2,z3]
    loss = cmc(modalities)
    
    print(loss)
