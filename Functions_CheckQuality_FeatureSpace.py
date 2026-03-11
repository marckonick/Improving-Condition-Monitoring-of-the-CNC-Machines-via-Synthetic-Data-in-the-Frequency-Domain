#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 14:24:48 2026

@author: nikola.markovic
"""

import numpy as np
import torch
#import Functions_FeatureExtraction as FFE
import TorchClassificationModels as tm
import torch.utils.data as data
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors




class labeled_dataset(Dataset):

      def __init__(self,X, Y):
        self.data = X
        self.labels = Y

      def __len__(self):
          return len(self.data)

      def __getitem__(self,idx):
          return (self.data[idx], self.labels[idx])
      
        
      
class VGG_Model_simple(nn.Module):
    def __init__(self, in_channels=3, n_chans1=[8,8], k_size = [3,3], padding_t='same', N_out = 2):
        super().__init__()
        
        self.chans1=n_chans1
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=self.chans1[0], kernel_size=(k_size[0], k_size[0]),  padding=padding_t)  # add stride=(1,2) to each layer
        self.conv2 = nn.Conv2d(in_channels=n_chans1[0], out_channels=self.chans1[0], kernel_size=(k_size[0],k_size[0]), padding=padding_t)

        self.bn1 = nn.BatchNorm2d(n_chans1[0])
        self.bn2 = nn.BatchNorm2d(n_chans1[0])
       
        
        self.drop_layer = nn.Dropout(0.2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        self.fc1 = nn.Linear(n_chans1[-1], N_out, bias=False) # N_out

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
            
        x = F.max_pool2d(x, kernel_size=(2,2))

        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        
        x = F.max_pool2d(x, kernel_size=(2,2))

        x = self.drop_layer(x)
        x = self.gap(x)
        x = x.reshape(x.shape[0], -1)
        
        x = self.fc1(x)
        
        return x

    def number_of_params(self):
         print('Numer of network paramteres:')
         print(sum(p.numel() for p in self.parameters()))    
         
         
         
def train_check_q_model(x_fake_melener, x_test_melener, device = 'mps', n_epochs=100):

    # FAKE - ONES
    # REAL - ZEROS 
    
    x_fake_melener_tr, x_fake_melener_val, y_fake_melener_tr, y_fake_melener_val = train_test_split(
    x_fake_melener, np.ones(len(x_fake_melener)), test_size=0.1, random_state=42, shuffle=True)

    x_check_q = np.concatenate((x_test_melener, x_fake_melener_tr), 0)
    y_check_q = np.concatenate((np.zeros(len(x_test_melener)), y_fake_melener_tr))  

    x_check_q = labeled_dataset(x_check_q, y_check_q)
    x_check_q = data.DataLoader(x_check_q, batch_size=32, shuffle=True, num_workers=0, drop_last=True)

    model_cq = VGG_Model_simple(3, [16,16], [3,3]).to(device)
    model_cq.number_of_params()  # prints number of params

    optimizer = optim.Adam(model_cq.parameters(), lr=1e-4) # 1e-5 mellog
    loss_fn = torch.nn.CrossEntropyLoss()

    n_epochs = n_epochs
    for epoch in range(1, n_epochs + 1):
        loss_train = 0.0
        for x, y in x_check_q:
                   x = x.float().to(device=device)
                   y = y.long().to(device=device)

                   outputs = model_cq(x.float())
                   loss = loss_fn(outputs, y.long()) 

                   optimizer.zero_grad()
                   loss.backward()
                   optimizer.step()
                   loss_train += loss.item()

        print(f" Epoch {epoch}/{n_epochs}, loss = {float(loss_train/len(x_check_q))}")     
        
    return model_cq, x_fake_melener_val, y_fake_melener_val   
         
         

# =========================
# Quality metrics in feature / embedding space
# =========================

@torch.no_grad()
def extract_fc1_input_embeddings(model: torch.nn.Module,
                                X: np.ndarray,
                                device: str = "cpu",
                                batch_size: int = 128) -> np.ndarray:

    model.eval()

    captured = {"emb": None}
    def _hook(module, inp, out):
        # inp is a tuple; fc1 input is inp[0], shape [B, D]
        captured["emb"] = inp[0].detach()

    # Try common attribute name "fc1". If not present, fall back to last nn.Linear found.
    handle = None
    if hasattr(model, "fc1") and isinstance(getattr(model, "fc1"), torch.nn.Module):
        handle = model.fc1.register_forward_hook(_hook)
    else:
        last_linear = None
        for m in model.modules():
            if isinstance(m, torch.nn.Linear):
                last_linear = m
        if last_linear is None:
            raise ValueError("No nn.Linear layer found in model; cannot extract last-layer embedding.")
        handle = last_linear.register_forward_hook(_hook)

    embs = []
    X_t = torch.tensor(X).float()
    n = X_t.shape[0]
    for i in range(0, n, batch_size):
        xb = X_t[i:i+batch_size].to(device)
        _ = model(xb)
        if captured["emb"] is None:
            raise RuntimeError("Embedding hook did not capture anything. Check model.fc1 existence.")
        embs.append(captured["emb"].cpu())
    handle.remove()

    emb = torch.cat(embs, dim=0).numpy()
    return emb


def _pairwise_sq_dists(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x·y
    x2 = (X**2).sum(dim=1, keepdim=True)  # [N,1]
    y2 = (Y**2).sum(dim=1, keepdim=True).T  # [1,M]
    return x2 + y2 - 2.0 * (X @ Y.T)


def _median_heuristic_sigma(Z: np.ndarray, max_points: int = 2000, seed: int = 0) -> float:

    rng = np.random.default_rng(seed)
    n = Z.shape[0]
    if n > max_points:
        idx = rng.choice(n, size=max_points, replace=False)
        Zs = Z[idx]
    else:
        Zs = Z

    # compute pairwise distances (upper triangle)
    Zt = torch.tensor(Zs).float()
    d2 = _pairwise_sq_dists(Zt, Zt).cpu().numpy()
    # exclude diagonal
    d2 = d2[np.triu_indices_from(d2, k=1)]
    d2 = d2[d2 > 0]
    if len(d2) == 0:
        return 1.0
    med = np.median(d2)
    return float(np.sqrt(med + 1e-12))


def mmd_rbf(real: np.ndarray,
            synth: np.ndarray,
            sigma = 0.0,
            max_points: int = 4000,
            seed: int = 0,
            device: str = "cpu") -> float:

    rng = np.random.default_rng(seed)
    X = real
    Y = synth
    if X.ndim > 2:
        X = X.reshape(X.shape[0], -1)
    if Y.ndim > 2:
        Y = Y.reshape(Y.shape[0], -1)

    if X.shape[0] > max_points:
        X = X[rng.choice(X.shape[0], size=max_points, replace=False)]
    if Y.shape[0] > max_points:
        Y = Y[rng.choice(Y.shape[0], size=max_points, replace=False)]

    if sigma is None:
        Z = np.concatenate([X, Y], axis=0)
        sigma = _median_heuristic_sigma(Z, max_points=min(2000, Z.shape[0]), seed=seed)

    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    Y_t = torch.tensor(Y, dtype=torch.float32, device=device)

    gamma = 1.0 / (2.0 * (sigma**2) + 1e-12)

    Kxx = torch.exp(-gamma * _pairwise_sq_dists(X_t, X_t))
    Kyy = torch.exp(-gamma * _pairwise_sq_dists(Y_t, Y_t))
    Kxy = torch.exp(-gamma * _pairwise_sq_dists(X_t, Y_t))

    # Unbiased estimate: remove diagonal terms from Kxx, Kyy
    n = X_t.shape[0]
    m = Y_t.shape[0]
    if n < 2 or m < 2:
        return float("nan")

    sum_Kxx = (Kxx.sum() - Kxx.diag().sum()) / (n * (n - 1))
    sum_Kyy = (Kyy.sum() - Kyy.diag().sum()) / (m * (m - 1))
    sum_Kxy = Kxy.mean()

    mmd2 = sum_Kxx + sum_Kyy - 2.0 * sum_Kxy
    return float(mmd2.detach().cpu().item())


def knn_precision_recall(real_emb: np.ndarray,
                         synth_emb: np.ndarray,
                         k: int = 5,
                         metric: str = "euclidean",
                         max_points = 10000,
                         seed: int = 0) -> dict:

    rng = np.random.default_rng(seed)
    R = real_emb
    S = synth_emb

    if max_points is not None:
        if R.shape[0] > max_points:
            R = R[rng.choice(R.shape[0], size=max_points, replace=False)]
        if S.shape[0] > max_points:
            S = S[rng.choice(S.shape[0], size=max_points, replace=False)]

    R = np.asarray(R, dtype=np.float32)
    S = np.asarray(S, dtype=np.float32)

    # Radii in real manifold: kth neighbor within real set (exclude self)
    nn_R = NearestNeighbors(n_neighbors=min(k+1, len(R)), metric=metric).fit(R)
    dRR, idxRR = nn_R.kneighbors(R, return_distance=True)
    # dRR[:,0] is self distance 0
    if dRR.shape[1] < k+1:
        raise ValueError(f"Not enough real samples for k={k}. Need at least k+1.")
    r_real = dRR[:, k]  # radius to kth neighbor

    # Radii in synthetic manifold
    nn_S = NearestNeighbors(n_neighbors=min(k+1, len(S)), metric=metric).fit(S)
    dSS, idxSS = nn_S.kneighbors(S, return_distance=True)
    if dSS.shape[1] < k+1:
        raise ValueError(f"Not enough synthetic samples for k={k}. Need at least k+1.")
    r_synth = dSS[:, k]

    # Precision: synth point close to real manifold
    nn_R1 = NearestNeighbors(n_neighbors=1, metric=metric).fit(R)
    dSR, idxSR = nn_R1.kneighbors(S, return_distance=True)
    nearest_real = idxSR[:, 0]
    precision = float(np.mean(dSR[:, 0] <= r_real[nearest_real] + 1e-12))

    # Recall/Coverage: real point close to synth manifold
    nn_S1 = NearestNeighbors(n_neighbors=1, metric=metric).fit(S)
    dRS, idxRS = nn_S1.kneighbors(R, return_distance=True)
    nearest_synth = idxRS[:, 0]
    recall = float(np.mean(dRS[:, 0] <= r_synth[nearest_synth] + 1e-12))

    return {"k": int(k), "precision": precision, "recall": recall}         