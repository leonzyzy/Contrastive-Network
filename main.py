import numpy as np
import torch 
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from Model import *
from CMCLoss import CMCLoss
from CSSLoss import CSSLoss
from Dataset import MyDataset, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

# GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# define a function to train model
def pretrain_model(train_loader, m1, m2, m3, m4, loss1, loss2, optimizer, a):
    for _, (x1, x2, x3, x4, y) in enumerate(train_loader):
        # move x, y to GPU if CUDA is available
        x1 = x1.to(device)
        x2 = x2.to(device)
        x3 = x3.to(device)
        x4 = x4.to(device)
        y = y.to(device)
        
        # forward pass
        f1 = m1(x1)
        f2 = m2(x2)
        f3 = m3(x3)
        #im = image_encoder(x4)
        f4 = m4(x4)
        
        l_cmc = loss1([f1, f2, f3, f4])
        
        # concat all features
        f = torch.cat((f1, f2, f3, f4), dim=1)
        
        l_css = loss2(f, y)
        
        # set as convex combination loss/user can choose their own combination
        train_loss = l_cmc + a*l_css
        
        # backward and optimize
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

# define a function to fine-tune model
def finetune_model(train_loader, m1, m2, m3, m4, m5, loss3, optimizer):
    # training bce
    for _, (x1, x2, x3, x4, y) in enumerate(train_loader):
        # move x, y to GPU if CUDA is available
        x1 = x1.to(device)
        x2 = x2.to(device)
        x3 = x3.to(device)
        x4 = x4.to(device)
        y = y.to(device)
        
        # forward pass
        f1 = m1(x1)
        f2 = m2(x2)
        f3 = m3(x3)
        #im = image_encoder(x4)
        f4 = m4(x4)
        
        # concat all features
        fused_vecs = torch.cat((f1, f2, f3, f4), dim=1)
        probs = m5(fused_vecs)
        l_bce = loss3(probs, y.unsqueeze(1).float())
        
        # backward and optimize
        optimizer.zero_grad()
        l_bce.backward()
        optimizer.step()

# define a function to test model
def test_model(test_loader, m1, m2, m3, m4, m5):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    probs = []
    y_true = []
    with torch.no_grad():
        for _, (x1, x2, x3, x4, y) in enumerate(test_loader):
            m1.eval()
            m2.eval()
            m3.eval()
            m4.eval()
            m5.eval()
            
            # move data to GPU
            x1 = x1.to(device)
            x2 = x2.to(device)
            x3 = x3.to(device)
            x4 = x4.to(device)
            y = y.to(device)

            # forward pass
            f1 = m1(x1)
            f2 = m2(x2)
            f3 = m3(x3)
            #im = image_encoder(x4)
            f4 = m4(x4)
                
            # concat all features
            fused_vecs = torch.cat((f1, f2, f3, f4), dim=1)
            prob = m5(fused_vecs)

            # save the probability and true label
            probs.append(prob.cpu().item())
            y_true.append(y.cpu().item())
            
            # calculate the accuracy, sensitivity, specificity
            predicted = torch.round(prob)
            TP += ((predicted == 1) & (y.unsqueeze(1) == 1)).sum().item()
            TN += ((predicted == 0) & (y.unsqueeze(1) == 0)).sum().item()
            FP += ((predicted == 1) & (y.unsqueeze(1) == 0)).sum().item()
            FN += ((predicted == 0) & (y.unsqueeze(1) == 1)).sum().item()
        
    test_sensitivity = TP / (TP + FN)
    test_specificity = TN / (TN + FP)
    test_ba = (test_sensitivity + test_specificity) / 2
    auc = roc_auc_score(y_true, probs)
    return test_ba, test_sensitivity, test_specificity, auc
    

if __name__=='main':
    # define some parameters
    params = {
        "data_seed": 528,
        "weight_seed":528,
        "b": 10,
        "m": 4,
        "t": 0.7,
        "k": 10,
        "a": 1,
        "learning_rate": 0.001,
        "pretrain_epochs": 1000,
        "finetune_epochs": 100,
        "lambda": 0.001
    }
    # get torch seed
    torch.manual_seed(params["weight_seed"])
    
    # define three random modalities
    x1 = torch.randn(100,87,87)
    x2 = torch.randn(100,87,87)
    x3 = torch.randn(100,87,100)
    #x4 = torch.randn((100, 3, 224, 224, 224))
    x5 = torch.randn(100,100)

    # define a random label with 1 and 0
    y = torch.randint(0, 2, (100,))

    # my dataset, skip imaging data here
    dataset = MyDataset(x1, x2, x3, x5, y)

    # 10-fold cross validation
    splits = KFold(n_splits = params["k"], shuffle = True, random_state = params["data_seed"])

    # record the performance
    ba = []
    sensitivity = []
    specificity = []
    auc = []

    for fold, (train_idx, test_idx) in enumerate(splits.split(dataset)):
        print('Fold : {}'.format(fold))
        dataset_train = Subset(dataset, train_idx)
        dataset_test = Subset(dataset, test_idx)
        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=params['b'], shuffle =True)
        test_loader = torch.utils.data.DataLoader(dataset_test, shuffle=True)

        # define each model
        config_fsc = {
                "num_of_attention_heads": 1,
                "hidden_size": 87,
                "linear_size": 8,
                "ROI_size" : 87,
                "embedding_size" : 8
            }

        config_radiomics = {
                "num_of_attention_heads": 1,
                "hidden_size": 100,
                "linear_size": 8,
                "ROI_size" : 87,
                "embedding_size" : 8
            }

        fc_encoder = AttentionEncoder(config_fsc).to(device)
        sc_encoder = AttentionEncoder(config_fsc).to(device)
        ra_encoder = AttentionEncoder(config_radiomics).to(device)
        #image_encoder = EfficientNet3D.from_name("efficientnet-b0",override_params={'num_classes': 8}, in_channels=3)
        clinical_encoder = ClinicalNet(100).to(device)

        # fused model
        fuse_encoder = FusionNet().to(device)

        # define loss 
        loss1 = CMCLoss(params['b'], params['t'], params['m'])
        loss2 = CSSLoss(params['t'])
        loss3 = nn.BCELoss()

        # definme optimizer
        pretrain_optimizer = torch.optim.Adam(list(fc_encoder.parameters()) + 
                                            list(sc_encoder.parameters()) + 
                                            list(ra_encoder.parameters()) + 
                                            list(clinical_encoder.parameters()), 
                                            lr=params['learning_rate'], 
                                            weight_decay=params['lambda'])

        finetune_optimizer = torch.optim.Adam(fuse_encoder.parameters(), 
                                              lr=params['learning_rate'],
                                              weight_decay=params['lambda'])
        
        # pretrain 
        for epoch in range(params['pretrain_epochs']):
            pretrain_model(train_loader, fc_encoder, sc_encoder, ra_encoder, 
                        clinical_encoder, loss1, loss2, pretrain_optimizer, a = 1)
        # finetune
        for epoch in range(params['finetune_epochs']):
            finetune_model(train_loader, fc_encoder, sc_encoder, ra_encoder, 
                        clinical_encoder, fuse_encoder, loss3, finetune_optimizer)

        # test the model
        ba_, sensitivity_, specificity_, auc_ = test_model(test_loader, fc_encoder, sc_encoder, ra_encoder, 
                        clinical_encoder, fuse_encoder,)
        ba.append(ba_)
        sensitivity.append(sensitivity_)
        specificity.append(specificity_)
        auc.append(auc_)
        
    print('BA : {}'.format(np.mean(ba)))
    print('Sensitivity : {}'.format(np.mean(sensitivity)))
    print('Specificity : {}'.format(np.mean(specificity)))
    print('AUC : {}'.format(np.mean(auc)))
    
