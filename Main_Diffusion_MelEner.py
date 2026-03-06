import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import time
from DiffusionModel_UNet import *
import math 
from torch.utils.data import WeightedRandomSampler


P_UNCOND = 0.15   # probability of dropping the label (10-20% is typical)

def cosine_beta_schedule(T, s=0.008):

    steps = T + 1
    x = torch.linspace(0, T, steps)

    # f(t)
    alphas_bar = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_bar = alphas_bar / alphas_bar[0]  # normalize so alpha_bar(0)=1
    
    # CLAMPING FOR SMOOTHER CURVE, INSTEAD OF SPIKE 
    betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
    betas = torch.clamp(betas, 1e-5, 0.26)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    

    return betas, alphas, alpha_bar



def lowrate_envelope(x, mel_reduce="mean", pool=8, smooth=True):
    """
    x: (B, C=3, H=64 mels, W=128 frames)
    returns: (B, C, W_low) low-rate envelope per axis
    """
    # collapse mel bins -> per-frame energy curve
    if mel_reduce == "mean":
        e = x.mean(dim=2)          # (B,C,W) e - batch x 3 x 128 od batch x 3 x 64 x 128
    elif mel_reduce == "sum":
        e = x.sum(dim=2)           # (B,C,W)
    else:
        raise ValueError("mel_reduce must be 'mean' or 'sum'")

    # optional smoothing at original rate (light)
    if smooth:
        # 1D conv smoothing via avg_pool1d with stride 1
        e = F.avg_pool1d(e, kernel_size=5, stride=1, padding=2) # dimension stays the same bc padding

    # downsample strongly (low-rate)
    if pool is not None and pool > 1:
        # average over windows of length=pool
        e_low = F.avg_pool1d(e, kernel_size=pool, stride=pool) # batch x 3 x 16, ppol = 4
    else:
        e_low = e

    return e_low


def envelope_lowrate_loss(x0_hat, x0, pool=8, mel_reduce="mean",
                          loss_type="l1", normalize=True, smooth=True):
    """
    Matches slow energy modulation between generated x0_hat and real x0.
    x0_hat, x0: (B,C,H,W)
    """
    e_hat = lowrate_envelope(x0_hat, mel_reduce=mel_reduce, pool=pool, smooth=smooth)
    e_ref = lowrate_envelope(x0,     mel_reduce=mel_reduce, pool=pool, smooth=smooth)

    if normalize:
        # per-sample, per-axis normalization so you match SHAPE not absolute offset
        mu_hat = e_hat.mean(dim=-1, keepdim=True) # mu_hat - batch x 3 x 1 <- mean per axis
        sd_hat = e_hat.std(dim=-1, keepdim=True) + 1e-6
        mu_ref = e_ref.mean(dim=-1, keepdim=True)
        sd_ref = e_ref.std(dim=-1, keepdim=True) + 1e-6
        e_hat = (e_hat - mu_hat) / sd_hat
        e_ref = (e_ref - mu_ref) / sd_ref

    if loss_type == "l1":
        return F.l1_loss(e_hat, e_ref)
    elif loss_type == "mse":
        return F.mse_loss(e_hat, e_ref)
    elif loss_type == "cosine":
        # cosine similarity over time (W_low), averaged over batch & axis
        cos = F.cosine_similarity(e_hat, e_ref, dim=-1)   # (B,C)
        return (1.0 - cos).mean()
    else:
        raise ValueError("loss_type must be 'l1', 'mse', or 'cosine'")
    

   
def estimate_x0_from_eps(x_t, eps_hat, alpha_bar):
    """
    x_t: (B,1,L)
    eps_hat: (B,1,L)
    alpha_bar: (B,1,1)
    """
    return (x_t - torch.sqrt(1.0 - alpha_bar) * eps_hat) / torch.sqrt(alpha_bar + 1e-8) # x_0 = f(x_t)

def estimate_x0_from_v(x_t, v_hat, alpha_bar):
    """
    x_t: (B,1,L) or (B,C,H,W) etc.
    v_hat: same shape as x_t
    alpha_bar: broadcastable to x_t (e.g., (B,1,1) or (B,1,1,1))
    """
    return torch.sqrt(alpha_bar) * x_t - torch.sqrt(1.0 - alpha_bar) * v_hat

def estimate_eps_from_v(x_t, v_hat, alpha_bar):
    return torch.sqrt(1.0 - alpha_bar) * x_t + torch.sqrt(alpha_bar) * v_hat

class labeled_dataset(data.Dataset):

      def __init__(self,X, Y):
        self.data = X
        self.labels = Y

      def __len__(self):
          return len(self.data)

      def __getitem__(self,idx):
          return (self.data[idx], self.labels[idx])

def comp_val_loss():
    
  model.eval()
  ema.apply_shadow(model)
  val_loss = 0.0
    
  with torch.no_grad():
   for xt, yt in x_test:

    noyz = torch.randn((len(xt), 3, 64,128), device=device)    
    t  = torch.randint(0, T, (len(xt),), device=device)
    alpha_bar = alpha_bar_t[t].view(len(xt), 1, 1, 1)    
        
    x0 = xt.float()    
    x0 = x0.to(device) 
    yt = yt.to(device)
    xt = x0*torch.sqrt(alpha_bar) + torch.sqrt(1-alpha_bar)*noyz
    
    v_tgt = v_target(x0, noyz, alpha_bar)
    y_pred = model(xt, t, yt)
    
    loss_v_eps = F.mse_loss(y_pred, v_tgt)
    #x0_hat = estimate_x0_from_v(xt, y_pred, alpha_bar)    

    ######## LOSSES #########
    loss_v = loss_v_eps
    val_loss += loss_v.item()
 
  ema.restore(model)
  return val_loss/len(x_test)


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                new_avg = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_avg.clone()

    def apply_shadow(self, model):
        """Use EMA weights (for sampling / evaluation)"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self, model):
        """Restore original training weights"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}
        
class LossEMA:
    def __init__(self, decay=0.99, eps=1e-8):
        self.decay = decay
        self.eps = eps
        self.vals = {}

    def update(self, name, value_float):
        if name not in self.vals:
            self.vals[name] = value_float
        else:
            self.vals[name] = self.decay * self.vals[name] + (1.0 - self.decay) * value_float

    def norm(self, name):
        # divisor for normalization
        return self.vals.get(name, 1.0) + self.eps
        
        

"""
v tends to keep target magnitudes more stable across t,
 often improving detail and reducing oversmoothing.
"""
def v_target(x0, eps, alpha_bar):
    return torch.sqrt(alpha_bar) * eps - torch.sqrt(1 - alpha_bar) * x0
     
def tv_time(x):
    # x: (B,C,H,W) where W is time frames
    return (x[..., 1:] - x[..., :-1]).abs().mean()

# ============================================================
# CFG HELPER: randomly replace labels with null class
# ============================================================
def apply_cfg_dropout(labels, null_class_idx, p_uncond, device):
    """
    Randomly replace class labels with the null (unconditional) index.
    
    Args:
        labels:         (B,) tensor of integer class labels
        null_class_idx: integer index for the null/unconditional embedding
        p_uncond:       probability of dropping the label
        device:         torch device
    
    Returns:
        (B,) tensor with some labels replaced by null_class_idx
    """
    mask = torch.rand(labels.shape[0], device=device) < p_uncond
    labels_cfg = labels.clone()
    labels_cfg[mask] = null_class_idx
    return labels_cfg


# X_melener_segment_filt_train 
x_data = np.load("X_melener_segment_filt_train.npy")# N x 4096 x 3
y_data = np.load("Y_data_segment_filt_train.npy")

x_test = np.load("X_melener_segment_filt_test.npy")
y_test = np.load("Y_data_segment_filt_test.npy")

x_data = np.concatenate((x_data, x_test), 0)
y_data = np.concatenate((y_data, y_test))


embed_classes_unique = np.unique(y_data)
embed_classes_unique = np.sort(embed_classes_unique) # od najmanjeg 

N_ec = len(embed_classes_unique)
cur_label = 0

rel_dict = {}
for i in range(15):
    if i in embed_classes_unique:
        rel_dict[i] = cur_label
        y_data[y_data == i] = cur_label
        cur_label += 1
    else:
        rel_dict[i] = -1    

labels_2_ops = {v: k for k, v in rel_dict.items()}


x_train, x_test, y_train, y_test= train_test_split(x_data, y_data, test_size=0.05, random_state=42)

print(x_train.shape)
print(x_test.shape)

class_counts = np.bincount(y_train)     
sample_weights = np.zeros_like(y_train, dtype=np.float32) # 
for j in np.unique(y_train):
    sample_weights[y_train == j] = 1.0/class_counts[j]
        
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)   

x_train = labeled_dataset(x_train, y_train)
x_test = labeled_dataset(x_test, y_test)

batch_size = 64
x_train = data.DataLoader(x_train, batch_size=batch_size, sampler=sampler, num_workers=0, drop_last=True)
x_test = data.DataLoader(x_test, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

# %%

device = 'mps' # cuda, mps 

T = 400
N_epochs = 201

model = UNet2D(in_channels=3, base_channels=32, time_emb_dim=128, num_ops = N_ec, op_emb_dim=64) # base_channels=64
model = model.to(device)
ema = EMA(model, decay=0.999)

NULL_CLASS_IDX = model.null_class_idx  
print(f"Null class index for CFG: {NULL_CLASS_IDX}")

model.number_of_params()
optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.999), weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_epochs)
loss_fn = torch.nn.MSELoss()

beta_t, alpha_t, alpha_bar_t = cosine_beta_schedule(T)

beta_t = beta_t.to(device=device)
alpha_t = alpha_t.to(device=device)
alpha_bar_t = alpha_bar_t.to(device=device)

losses = []

  
w_x0_mse = 0.4
w_tv = 0.2
w_env = 2.6
env_pool = 4


for epoch in range(N_epochs):
    
  cur_loss = 0.0
  cur_eps_loss = 0.0
  cur_x0_mse_loss = 0.0
  cur_tv_loss = 0.0
  cur_env_loss = 0.0


  for x, y in x_train:

      noyz = torch.randn((batch_size,3,64,128), device=device)
      t  = torch.randint(0, T, (batch_size,), device=device)
      alpha_bar = alpha_bar_t[t].view(batch_size, 1, 1, 1)
        
      snr = alpha_bar.squeeze() / (1.0 - alpha_bar.squeeze() + 1e-8)  # shape (B,)

      # Threshold — SNR >= 1 means at least as much signal as noise
      snr_thresh = 1.0 
      clean_mask = (snr >= snr_thresh)      
      #############################


      x0 = x.float()      
      x0 = x0.to(device = device)
      y = y.to(device = device)

      # =====================================================
      # CFG: randomly drop labels → null class
      # =====================================================
      y_cfg = apply_cfg_dropout(y, NULL_CLASS_IDX, P_UNCOND, device) 
        
      x_t = x0*torch.sqrt(alpha_bar) + torch.sqrt(1 - alpha_bar) * noyz # x_t = f(x0)
      v_tgt = v_target(x0, noyz, alpha_bar)
      
      y_pred = model(x_t, t, y_cfg)     
      ####### LOSS COMPUTATION #######  

      loss_eps = F.mse_loss(y_pred, v_tgt)
      x0_hat = estimate_x0_from_v(x_t, y_pred, alpha_bar)

      if clean_mask.any(): #low_t_mask.any():

        x0_hat_low = x0_hat[clean_mask] 
        x0_low = x0[clean_mask] 
        loss_x0 = F.l1_loss(x0_hat_low, x0_low)
        loss_tv = tv_time(x0_hat_low)
        
        loss_env = envelope_lowrate_loss(
           x0_hat_low, x0_low, #x0_hat, x0,
           pool=env_pool,
           mel_reduce="mean",
           loss_type="l1",
           normalize=True,
           smooth=True) 
      else:
         loss_x0 = loss_tv = loss_env = torch.tensor(0.0, device=device)
      
      loss = loss_eps + w_x0_mse * loss_x0 + w_tv * loss_tv + w_env * loss_env
        
      optimizer.zero_grad(set_to_none=True)
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      optimizer.step()
      ema.update(model)   

      cur_loss += loss.item()
      cur_eps_loss += loss_eps.item()  
      cur_x0_mse_loss += w_x0_mse * loss_x0.item() 
      cur_tv_loss += w_tv * loss_tv.item() 
      cur_env_loss += (w_env * loss_env.item())

   
  losses.append(cur_loss/len(x_train))
  scheduler.step()       
  if epoch % 20 == 0:
      vl = comp_val_loss()
      print(f"  Epoch: {epoch}, Train/Val: {cur_loss/len(x_train):.6f} / {vl:.6f}\n"
            f"  eps loss: {cur_eps_loss/len(x_train):.6f}\n"
            f"  mse x0 loss: {cur_x0_mse_loss/len(x_train):.6f}\n"
            f"  tv loss: {cur_tv_loss/len(x_train):.6f}\n"
            f"  envelope loss: {cur_env_loss/len(x_train):.6f}\n"
            f"  learning rate is: {optimizer.param_groups[0]['lr']:.6f}\n")
      model.train()

int_timestamp = int(time.time())

torch.save({
    f"diffusion_model_melener_M2_M3_ALL_OPS_{T}_steps_{N_epochs}_epochs_{int_timestamp}": model.state_dict(),
    f"ema__melener_{T}_steps_{N_epochs}_epochs_{int_timestamp}": ema.shadow,
}, "ckpt_melener.pt")




