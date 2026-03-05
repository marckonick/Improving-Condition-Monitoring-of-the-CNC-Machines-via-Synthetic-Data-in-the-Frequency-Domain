# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 16:58:28 2025

@author: nikola.markovic
"""

# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import Dataset
import torch.nn.functional as F



class labeled_dataset(Dataset):

      def __init__(self,X, Y):
        self.data = X
        self.labels = Y

      def __len__(self):
          return len(self.data)

      def __getitem__(self,idx):
          return (self.data[idx], self.labels[idx])
      
class labeled_dataset_extended(Dataset):

      def __init__(self, X, Y, Y_OPS):
        self.data = X
        self.labels = Y
        self.ops_labels = Y_OPS

      def __len__(self):
          return len(self.data)

      def __getitem__(self,idx):
          return (self.data[idx], self.labels[idx], self.ops_labels[idx])
        
      
class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification.
    Args:
        alpha (float): Weighting factor for the positive class (anomalies).
        gamma (float): Focusing parameter to reduce loss from easy examples.
        reduction (str): 'mean' | 'sum' | 'none'
        from_logits (bool): If True, assumes model outputs raw logits.
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', from_logits=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.from_logits = from_logits

    def forward(self, inputs, targets):
        # inputs: (N,) or (N, 1)
        # targets: (N,) with values in {0, 1}
        
        inputs = inputs[:,0] # Da bude samo [batch_size]
        if self.from_logits:
            probs = torch.sigmoid(inputs)
        else:
            probs = inputs.clamp(min=1e-6, max=1-1e-6)

        # Compute focal loss components
        ce_loss = F.binary_cross_entropy(probs, targets.float(), reduction='none')
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * (1 - p_t) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
        
        
class VGG_Model(nn.Module):
    def __init__(self, in_channels=3, n_chans1=[8,16,16], k_size = [3,3,3], padding_t='same', N_out = 1):
        super().__init__()
        
        self.chans1=n_chans1
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=self.chans1[0], kernel_size=(k_size[0], k_size[0]),  padding=padding_t)  # add stride=(1,2) to each layer
        self.conv2 = nn.Conv2d(in_channels=n_chans1[0], out_channels=self.chans1[0], kernel_size=(k_size[0],k_size[0]), padding=padding_t)

        self.conv3 = nn.Conv2d(in_channels=self.chans1[0],out_channels=self.chans1[1], kernel_size=(k_size[1],k_size[1]),  padding=padding_t)
        self.conv4 = nn.Conv2d(in_channels=self.chans1[1],out_channels=self.chans1[1], kernel_size=(k_size[1],k_size[1]), padding=padding_t)

        self.conv5 = nn.Conv2d(in_channels=self.chans1[1],out_channels=self.chans1[2], kernel_size=(k_size[2],k_size[2]),  padding=padding_t)
        self.conv6 = nn.Conv2d(in_channels=self.chans1[2],out_channels=self.chans1[2], kernel_size=(k_size[2],k_size[2]),   padding=padding_t)

        self.bn1 = nn.BatchNorm2d(n_chans1[0])
        self.bn2 = nn.BatchNorm2d(n_chans1[0])
        self.bn3 = nn.BatchNorm2d(n_chans1[1])
        self.bn4 = nn.BatchNorm2d(n_chans1[1])
        self.bn5 = nn.BatchNorm2d(n_chans1[2])
        self.bn6 = nn.BatchNorm2d(n_chans1[2])        
        
        self.drop_layer = nn.Dropout(0.2)
        #self.gap = GAP1d(1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        self.fc1 = nn.Linear(n_chans1[-1], N_out, bias=False) # N_out
        
        #self.fc1 = nn.Linear(n_chans1[-1], 64, bias=False) # N_out
        #self.fc2 = nn.Linear(64, N_out, bias=False)        
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        
        x = F.max_pool2d(x, kernel_size=(2,2))

        x = self.conv3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = torch.relu(x)
        
        x = F.max_pool2d(x, kernel_size=(2,2))

        x = self.conv5(x)
        x = self.bn5(x)
        x = torch.relu(x)
        
        x = self.conv6(x)
        x = self.bn6(x)
        x = torch.relu(x)
        
        x = F.max_pool2d(x, kernel_size=(2,2))

        x = self.drop_layer(x)
        x = self.gap(x)
        x = x.reshape(x.shape[0], -1)
        
        x = self.fc1(x)
        
        #x = torch.relu(self.fc1(x))
        #x = self.drop_layer(x)
        #x = self.fc2(x) 
        
        return x

    def number_of_params(self):
         print('Numer of network paramteres:')
         print(sum(p.numel() for p in self.parameters()))      
         
         
class VGG_Model_Extended(nn.Module):
    def __init__(self, in_channels=3, n_chans1=[8,16,16], k_size = [3,3,3], padding_t='same', N_out = 1, num_ops = 11, op_emb_dim = 16):
        super().__init__()
        
        self.chans1=n_chans1
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=self.chans1[0], kernel_size=(k_size[0], k_size[0]),  padding=padding_t)  # add stride=(1,2) to each layer
        self.conv2 = nn.Conv2d(in_channels=n_chans1[0], out_channels=self.chans1[0], kernel_size=(k_size[0],k_size[0]), padding=padding_t)

        self.conv3 = nn.Conv2d(in_channels=self.chans1[0],out_channels=self.chans1[1], kernel_size=(k_size[1],k_size[1]),  padding=padding_t)
        self.conv4 = nn.Conv2d(in_channels=self.chans1[1],out_channels=self.chans1[1], kernel_size=(k_size[1],k_size[1]), padding=padding_t)

        self.conv5 = nn.Conv2d(in_channels=self.chans1[1],out_channels=self.chans1[2], kernel_size=(k_size[2],k_size[2]),  padding=padding_t)
        self.conv6 = nn.Conv2d(in_channels=self.chans1[2],out_channels=self.chans1[2], kernel_size=(k_size[2],k_size[2]),   padding=padding_t)

        self.bn1 = nn.BatchNorm2d(n_chans1[0])
        self.bn2 = nn.BatchNorm2d(n_chans1[0])
        self.bn3 = nn.BatchNorm2d(n_chans1[1])
        self.bn4 = nn.BatchNorm2d(n_chans1[1])
        self.bn5 = nn.BatchNorm2d(n_chans1[2])
        self.bn6 = nn.BatchNorm2d(n_chans1[2])        
        
        
        self.op_emb = nn.Embedding(num_ops + 1, op_emb_dim)
        
        
        self.drop_layer = nn.Dropout(0.2)
        #self.gap = GAP1d(1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        self.fc1 = nn.Linear(op_emb_dim + n_chans1[-1], N_out, bias=False) # N_out
        
        #self.fc1 = nn.Linear(n_chans1[-1], 64, bias=False) # N_out
        #self.fc2 = nn.Linear(64, N_out, bias=False)        
        
    def forward(self, x, y_op):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        
        x = F.max_pool2d(x, kernel_size=(2,2))

        x = self.conv3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = torch.relu(x)
        
        x = F.max_pool2d(x, kernel_size=(2,2))

        x = self.conv5(x)
        x = self.bn5(x)
        x = torch.relu(x)
        
        x = self.conv6(x)
        x = self.bn6(x)
        x = torch.relu(x)
        
        x = F.max_pool2d(x, kernel_size=(2,2))

        x = self.drop_layer(x)
        x = self.gap(x)
        x = x.reshape(x.shape[0], -1)
        
        op_vec = self.op_emb(y_op)
        
        x = torch.cat([x, op_vec], dim=1)
        
        x = self.fc1(x)
        
        #x = torch.relu(self.fc1(x))
        #x = self.drop_layer(x)
        #x = self.fc2(x) 
        
        return x

    def number_of_params(self):
         print('Numer of network paramteres:')
         print(sum(p.numel() for p in self.parameters()))      
         
         
class VGG1D_Model(nn.Module):
    def __init__(self, in_channels=3,  n_chans1=[32,32,32,32], k_size = [3,3,3,3], padding_t='same', N_out = 2):
        super().__init__()
        self.chans1=n_chans1
        self.conv1 = nn.Conv1d(in_channels, self.chans1[0], k_size[0],  padding = padding_t)  # add stride=(1,2) to each layer
        self.conv2 = nn.Conv1d(n_chans1[0], self.chans1[0], k_size[0], padding =padding_t)

        self.conv3 = nn.Conv1d(self.chans1[0],self.chans1[1], k_size[1],  padding =padding_t)
        self.conv4 = nn.Conv1d(self.chans1[1],self.chans1[1], k_size[1], padding =padding_t)

        self.conv5 = nn.Conv1d(self.chans1[1],self.chans1[2], k_size[2],  padding =padding_t)
        self.conv6 = nn.Conv1d(self.chans1[2],self.chans1[2], k_size[2],   padding =padding_t)

        self.conv7 = nn.Conv1d(self.chans1[2],self.chans1[3], k_size[2],  padding =padding_t)
        self.conv8 = nn.Conv1d(self.chans1[3],self.chans1[3], k_size[2],   padding =padding_t)
        
        self.bn1 = nn.BatchNorm1d(n_chans1[0])
        self.bn2 = nn.BatchNorm1d(n_chans1[0])
        self.bn3 = nn.BatchNorm1d(n_chans1[1])
        self.bn4 = nn.BatchNorm1d(n_chans1[1])
        self.bn5 = nn.BatchNorm1d(n_chans1[2])
        self.bn6 = nn.BatchNorm1d(n_chans1[2])
        self.bn7 = nn.BatchNorm1d(n_chans1[3])
        self.bn8 = nn.BatchNorm1d(n_chans1[3])
        
        
        self.drop_layer = nn.Dropout(0.2)
        #self.gap = GAP1d(1)
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        #self.fc1 = nn.Linear(n_chans1[-1], N_out, bias=False) #
        
        self.fc1 = nn.Linear(n_chans1[-1], 64, bias=True) #
        self.fc2 = nn.Linear(64, N_out, bias=False)
        
    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        
        x = F.max_pool1d(x, kernel_size=4)

        x = self.conv3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = torch.relu(x)
        
        x = F.max_pool1d(x, kernel_size=4)
        
        x = self.conv5(x)
        x = self.bn5(x)
        x = torch.relu(x)
        
        x = self.conv6(x)
        x = self.bn6(x)
        x = torch.relu(x)
        
        x = F.max_pool1d(x, kernel_size=2)

        x = self.conv7(x)
        x = self.bn7(x)
        x = torch.relu(x)
        
        x = self.conv8(x)
        x = self.bn8(x)
        x = torch.relu(x)
        
        x = F.max_pool1d(x, kernel_size=2)
        
        x = self.drop_layer(x)
        x = self.gap(x)
        x = x.reshape(x.shape[0], -1)
        
        # 1axis v2 uses fc1 and fc2 
        x = torch.relu(self.fc1(x))
        x = self.drop_layer(x)
        x = self.fc2(x)
        
        #x = self.fc1(x)
        
        return x

    def number_of_params(self):
         print('Number of network paramteres:')
         print(sum(p.numel() for p in self.parameters()))              
         
         
class DNN_Model(nn.Module):
    def __init__(self, in_dim = 2048, in_channels=1, n_hidden=[100]):
        super().__init__()
        

          # add stride=(1,2) to each layer
        
        n_layers = len(n_hidden)
        
        layers = [nn.Linear(in_channels*in_dim, n_hidden[0])]
        for i in range(0, n_layers - 1):
            layers.append(nn.Linear(n_hidden[i], n_hidden[i+1]))
            
        layers.append(nn.Linear(n_hidden[-1], 2))
        
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = torch.relu(x)

        x = self.layers[-1](x)  # Output layer (no activation)
        return x       
    
    def number_of_params(self):
         print('Numer of network paramteres:')
         print(sum(p.numel() for p in self.parameters()))   
 
     
 
    
 
    
 
    