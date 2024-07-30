import torch
import torch.nn as nn
import torch.optim as optim

# define the fusion block
class FusionNet(nn.Module):
    def __init__(self, input_size=8):
        super(FusionNet, self).__init__()
        self.fusion = nn.Linear(input_size*4, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, fused_features):
        # can customize ur own network there.....
        x = self.fusion(fused_features)
        return x

# define moco block as supervised setting...
class SupervisedMoCo(nn.Module):
    def __init__(self, encoder, moco_dim=128, moco_k=65536, moco_m=0.999):
        super(SupervisedMoCo, self).__init__()
        self.encoder_q = encoder
        self.encoder_k = encoder
        self.encoder_k.eval()  # Key encoder is fixed during training
        self.moco_dim = moco_dim
        self.moco_k = moco_k
        self.moco_m = moco_m
        
        self.queue = torch.randn(self.moco_dim, self.moco_k)
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer('queue', self.queue)
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))
    
    def forward(self, x_q, x_k):
        features_q, logits_q = self.encoder_q(x_q)
        features_q = nn.functional.normalize(features_q, dim=1)
        
        with torch.no_grad():
            features_k, _ = self.encoder_k(x_k)
            features_k = nn.functional.normalize(features_k, dim=1)
        
        return features_q, logits_q, features_k
    
    def update_queue(self, k):
        batch_size = k.shape[0]
        ptr = int(self.queue_ptr)
        
        self.queue[:, ptr:ptr + batch_size] = k.T
        ptr = (ptr + batch_size) % self.moco_k
        self.queue_ptr[0] = ptr
    
    def compute_contrastive_loss(self, q, k):
        pos = torch.exp(torch.mm(q, k.t()) / 0.07)
        neg = torch.mm(q, self.queue.clone().detach()) / 0.07
        loss = -torch.log(pos / (pos + neg)).mean()
        return loss

def train_supervised_moco(moco, data_loader, num_epochs=10, lr=1e-3):
    optimizer = optim.Adam(moco.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
      
    for epoch in range(num_epochs):
        moco.encoder_q.train()
        for (x1, x2, x3, y) in data_loader:
            optimizer.zero_grad()
              
            # Flatten the input tensors
            x1, x2, x3 = x1.view(x1.size(0), -1), x2.view(x2.size(0), -1), x3.view(x3.size(0), -1)
              
            # Forward pass through the model
            q1, logits_q1, k1 = moco(x1, x1)
            q2, logits_q2, k2 = moco(x2, x2)
            q3, logits_q3, k3 = moco(x3, x3)
              
            # Compute contrastive loss
            contrastive_loss1 = moco.compute_contrastive_loss(q1, k1)
            contrastive_loss2 = moco.compute_contrastive_loss(q2, k2)
            contrastive_loss3 = moco.compute_contrastive_loss(q3, k3)
              
            contrastive_loss = (contrastive_loss1 + contrastive_loss2 + contrastive_loss3) / 3
              
            # Compute classification loss
            class_loss1 = criterion(logits_q1, y)
            class_loss2 = criterion(logits_q2, y)
            class_loss3 = criterion(logits_q3, y)
              
            class_loss = (class_loss1 + class_loss2 + class_loss3) / 3
              
            # Total loss
            total_loss = contrastive_loss + class_loss
              
            # Backward pass and optimization
            total_loss.backward()
            optimizer.step()
              
            # Update queue
            moco.update_queue(k1)
            moco.update_queue(k2)
            moco.update_queue(k3)

if __name__=='main':  
    # Random modalities and labels
    x1 = torch.randn(100, 87 * 87)
    x2 = torch.randn(100, 87 * 87)
    x3 = torch.randn(100, 87 * 100)
    y = torch.randint(0, 2, (100,))
    
    # Create data loaders
    data = torch.utils.data.TensorDataset(x1, x2, x3, y)
    data_loader = torch.utils.data.DataLoader(data, batch_size=32, shuffle=True)
    
    # Initialize and train the model
    encoder = FusionNet(#customize....here...#)
    moco = SupervisedMoCo(encoder)
    train_supervised_moco(moco, data_loader)
  
