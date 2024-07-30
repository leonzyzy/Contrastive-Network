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

# define mocov2 supervised setting
class MoCoV2(nn.Module):
    def __init__(self, encoder, projection_dim, queue_size=65536, momentum=0.999):
        super(MoCoV2, self).__init__()
        self.encoder_q = encoder
        self.encoder_k = encoder
        self.encoder_k.eval()  # Key encoder is fixed during training
        
        self.projection_head_q = ProjectionHead(128, projection_dim)
        self.projection_head_k = ProjectionHead(128, projection_dim)
        
        self.queue_size = queue_size
        self.momentum = momentum
        
        self.queue = torch.randn(projection_dim, queue_size)
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer('queue', self.queue)
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))
    
    def forward(self, x_q, x_k):
        features_q, logits_q = self.encoder_q(x_q)
        features_q = nn.functional.normalize(features_q, dim=1)
        z_q = self.projection_head_q(features_q)
        
        with torch.no_grad():
            features_k, _ = self.encoder_k(x_k)
            features_k = nn.functional.normalize(features_k, dim=1)
            z_k = self.projection_head_k(features_k)
        
        return z_q, logits_q, z_k
    
    def update_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)
        for param_q, param_k in zip(self.projection_head_q.parameters(), self.projection_head_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)
    
    def update_queue(self, k):
        batch_size = k.shape[0]
        ptr = int(self.queue_ptr)
        
        self.queue[:, ptr:ptr + batch_size] = k.T
        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr
    
    def compute_contrastive_loss(self, z_q, z_k):
        # MoCo v2 loss function
        q = nn.functional.normalize(z_q, dim=1)
        k = nn.functional.normalize(z_k, dim=1)
        
        pos = torch.exp(torch.mm(q, k.t()) / 0.07)
        neg = torch.mm(q, self.queue.clone().detach()) / 0.07
        loss = -torch.log(pos / (pos + neg)).mean()
        return loss

def train_moco_v2(moco, data_loader, num_epochs=10, lr=1e-3):
    optimizer = optim.Adam(moco.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        moco.encoder_q.train()
        for (x1, x2, x3, y) in data_loader:
            optimizer.zero_grad()
            
            # Flatten the input tensors
            x1, x2, x3 = x1.view(x1.size(0), -1), x2.view(x2.size(0), -1), x3.view(x3.size(0), -1)
            
            # Forward pass through the model
            z_q1, logits_q1, z_k1 = moco(x1, x1)
            z_q2, logits_q2, z_k2 = moco(x2, x2)
            z_q3, logits_q3, z_k3 = moco(x3, x3)
            
            # Compute contrastive loss
            contrastive_loss1 = moco.compute_contrastive_loss(z_q1, z_k1)
            contrastive_loss2 = moco.compute_contrastive_loss(z_q2, z_k2)
            contrastive_loss3 = moco.compute_contrastive_loss(z_q3, z_k3)
            
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
            
            # Update queue and momentum encoder
            moco.update_queue(z_k1)
            moco.update_queue(z_k2)
            moco.update_queue(z_k3)
            moco.update_encoder()

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
  moco = MoCoV2(encoder, projection_dim=128)
  train_moco_v2(moco, data_loader)

