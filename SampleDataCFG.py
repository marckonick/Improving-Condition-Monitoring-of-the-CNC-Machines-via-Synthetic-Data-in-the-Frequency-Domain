
import numpy as np
import torch
import torch.nn.functional as F
import math
from DiffusionModel_UNet import UNet2D
from operator import itemgetter

def cosine_beta_schedule(T, s=0.008):
    steps = T + 1
    x = torch.linspace(0, T, steps)
    alphas_bar = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_bar = alphas_bar / alphas_bar[0]
    betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
    betas = torch.clamp(betas, 1e-5, 0.26)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bar


def estimate_x0_from_v(x_t, v_hat, alpha_bar):
    return torch.sqrt(alpha_bar) * x_t - torch.sqrt(1.0 - alpha_bar) * v_hat

def estimate_eps_from_v(x_t, v_hat, alpha_bar):
    return torch.sqrt(1.0 - alpha_bar) * x_t + torch.sqrt(alpha_bar) * v_hat


# ============================================================
# CORE CFG LOGIC: combines conditional + unconditional predictions
# ============================================================
@torch.no_grad()
def cfg_model_prediction(model, x_t, t, op_id, guidance_scale, null_class_idx):
    """
    Classifier-Free Guidance prediction.
    
    Runs the model twice:
      1. Conditional:   v_cond   = model(x_t, t, op_id)
      2. Unconditional: v_uncond = model(x_t, t, null_class)
    
    Then combines:
      v_guided = v_uncond + w * (v_cond - v_uncond)
               = (1-w) * v_uncond + w * v_cond
    
    When w=1.0, this equals v_cond (no guidance).
    When w>1.0, it amplifies the class-conditional signal.
    
    Args:
        model:          the UNet2D model
        x_t:            (B, 3, 64, 128) noisy input
        t:              (B,) timestep indices
        op_id:          (B,) operation class indices
        guidance_scale: float, the 'w' parameter (1.0 = no guidance)
        null_class_idx: integer index for the null embedding
    
    Returns:
        v_guided: (B, 3, 64, 128) guided v-prediction
    """
    B = x_t.shape[0]
    
    if guidance_scale == 1.0:
        # No guidance needed, save compute
        return model(x_t, t, op_id)
    
    # Conditional prediction
    v_cond = model(x_t, t, op_id)
    
    # Unconditional prediction (replace all labels with null)
    null_labels = torch.full((B,), null_class_idx, dtype=torch.long, device=x_t.device)
    v_uncond = model(x_t, t, null_labels)
    
    # CFG formula
    v_guided = v_uncond + guidance_scale * (v_cond - v_uncond)
    
    return v_guided


# ============================================================
# DDPM Sampler with CFG
# ============================================================
@torch.no_grad()
def sample_ddpm_cfg(
    model,
    op_labels,           # (B,) integer tensor of operation class indices
    T=400,
    guidance_scale=2.0,  # CFG strength
    device='mps',
    clip_x0=True,        # clip predicted x0 to training range
    clip_range=(-5.0, 5.0),
):
    """
    DDPM sampling with classifier-free guidance.
    
    Args:
        model:          trained UNet2D (with null class support)
        op_labels:      (B,) tensor of operation indices to generate
        T:              number of diffusion timesteps (must match training)
        guidance_scale: CFG strength. 1.0=no guidance, 2.0-3.0=recommended
        device:         torch device
        clip_x0:        whether to clip the predicted x0 at each step
        clip_range:     clipping range for x0 predictions
    
    Returns:
        x0: (B, 3, 64, 128) generated samples
    """
    model.eval()
    null_class_idx = model.null_class_idx
    
    beta_t, alpha_t, alpha_bar_t = cosine_beta_schedule(T)
    beta_t = beta_t.to(device)
    alpha_t = alpha_t.to(device)
    alpha_bar_t = alpha_bar_t.to(device)
    
    B = op_labels.shape[0]
    op_labels = op_labels.to(device)
    
    # Start from pure noise
    x_t = torch.randn((B, 3, 64, 128), device=device)
    
    for i in reversed(range(T)):
        t = torch.full((B,), i, dtype=torch.long, device=device)
        
        ab = alpha_bar_t[i]
        a = alpha_t[i]
        b = beta_t[i]
        
        # CFG-guided v-prediction
        v_guided = cfg_model_prediction(
            model, x_t, t, op_labels, guidance_scale, null_class_idx)
        
        # Convert v → eps for the DDPM update step
        eps_guided = estimate_eps_from_v(x_t, v_guided, ab.view(1, 1, 1, 1))
        
        # Optional: clip the implied x0 to prevent drift
        if clip_x0:
            x0_hat = estimate_x0_from_v(x_t, v_guided, ab.view(1, 1, 1, 1))
            x0_hat = x0_hat.clamp(clip_range[0], clip_range[1])
            # Recompute eps from clipped x0
            eps_guided = (x_t - torch.sqrt(ab) * x0_hat) / (torch.sqrt(1 - ab) + 1e-8)
        
        # DDPM reverse step
        if i > 0:
            noise = torch.randn_like(x_t)
        else:
            noise = torch.zeros_like(x_t)
        
        # x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (beta_t / sqrt(1 - alpha_bar_t)) * eps) + sigma * z
        coef_eps = b / (torch.sqrt(1 - ab) + 1e-8)
        x_t = (1.0 / torch.sqrt(a)) * (x_t - coef_eps * eps_guided) + torch.sqrt(b) * noise
    
    return x_t


# ============================================================
# DDIM Sampler with CFG (recommended for better quality)
# ============================================================
@torch.no_grad()
def sample_ddim_cfg(
    model,
    op_labels,           # (B,) integer tensor of operation class indices
    T=400,
    num_steps=50,        # DDIM uses fewer steps (50-100 typical)
    guidance_scale=2.0,
    eta=0.0,             # 0.0=deterministic DDIM, 1.0=DDPM-like stochasticity
    device='mps',
    clip_x0=True,
    clip_range=(-5.0, 5.0),
):
    """
    DDIM sampling with classifier-free guidance.
    
    DDIM is recommended over DDPM because:
      - Faster (50 steps vs 400)
      - eta parameter controls diversity vs quality tradeoff
      - Cleaner samples, especially with CFG
    
    Args:
        model:          trained UNet2D
        op_labels:      (B,) tensor of operation indices
        T:              total diffusion timesteps (must match training schedule)
        num_steps:      number of DDIM sampling steps (50-100 recommended)
        guidance_scale: CFG strength
        eta:            stochasticity (0=deterministic, 0.5=moderate, 1.0=full DDPM noise)
        device:         torch device
        clip_x0:        clip predicted x0
        clip_range:     clipping range
    
    Returns:
        x0: (B, 3, 64, 128) generated samples
    """
    model.eval()
    null_class_idx = model.null_class_idx
    
    _, _, alpha_bar_t = cosine_beta_schedule(T)
    alpha_bar_t = alpha_bar_t.to(device) # goes from 1 to zero in t steps 
    
    B = op_labels.shape[0]
    op_labels = op_labels.to(device)
    
    # Create sub-sequence of timesteps for DDIM
    # Evenly spaced from T-1 down to 0 in num_steps steps
    step_indices = torch.linspace(T - 1, 0, num_steps + 1, dtype=torch.long, device=device)
    
    # Start from pure noise
    x_t = torch.randn((B, 3, 64, 128), device=device)
    
    for idx in range(num_steps):
        t_cur = step_indices[idx]
        t_prev = step_indices[idx + 1]
        
        t_batch = torch.full((B,), t_cur.item(), dtype=torch.long, device=device)
        
        ab_cur = alpha_bar_t[t_cur]
        ab_prev = alpha_bar_t[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=device)
        
        # CFG-guided v-prediction MAKES MODEL PREDICTIONS FOR THE GIVEN LABELS AND THE SAME ANOUNT OF PREDICITONS FOR NULL LABEL AND THEN MERGES THEM 
        v_guided = cfg_model_prediction(
            model, x_t, t_batch, op_labels, guidance_scale, null_class_idx)
        
        # Convert v → x0 and eps
        x0_hat = estimate_x0_from_v(x_t, v_guided, ab_cur.view(1, 1, 1, 1))
        eps_hat = estimate_eps_from_v(x_t, v_guided, ab_cur.view(1, 1, 1, 1))
        
        if clip_x0:
            x0_hat = x0_hat.clamp(clip_range[0], clip_range[1]) # -5, 5
            # Recompute eps to be consistent with clipped x0
            eps_hat = (x_t - torch.sqrt(ab_cur) * x0_hat) / (torch.sqrt(1 - ab_cur) + 1e-8)
        
        # DDIM update
        # sigma controls the stochasticity
        sigma = eta * torch.sqrt((1 - ab_prev) / (1 - ab_cur + 1e-8)) * torch.sqrt(1 - ab_cur / (ab_prev + 1e-8))
        
        # "predicted direction pointing to x_t"
        dir_x_t = torch.sqrt(1 - ab_prev - sigma**2 + 1e-8) * eps_hat
        
        # noise (zero if eta=0 for deterministic sampling)
        if eta > 0 and idx < num_steps - 1:
            noise = torch.randn_like(x_t)
        else:
            noise = torch.zeros_like(x_t)
        
        x_t = torch.sqrt(ab_prev) * x0_hat + dir_x_t + sigma * noise
    
    return x_t


def get_sampled_data(ops_to_generate, samples_per_op, guidance_scale):
    
    
    all_samples = []
    all_labels = []
        
    for op_idx in ops_to_generate:
            w = per_op_guidance.get(op_idx, guidance_scale)
            labels = torch.full((samples_per_op[op_idx],), op_idx, dtype=torch.long)
            
            print(f"Generating OP index {op_idx} with guidance_scale={w} ...")
            
            if samples_per_op[op_idx] > 0:

              # DDIM sampling (recommended)
              samples = sample_ddim_cfg(
                model,
                op_labels=labels,
                T=T,
                num_steps=100,       # 50-100 steps is usually sufficient
                guidance_scale=w,
                eta=0.3,             # small stochasticity for diversity
                device=device,
                clip_x0=True,
                clip_range=(-5.0, 5.0),
              )
            
              all_samples.append(samples.cpu().numpy())
              all_labels.append(np.full(samples_per_op[op_idx], op_idx))
            else:
                print(f"No samples for operation {op_idx}")   
            
            
            print(f"  Done. Shape: {samples.shape}")
        
        
    x_samples = np.concatenate(all_samples, axis=0)
    y_samples = np.concatenate(all_labels, axis=0)
    y_samples_real = itemgetter(*y_samples)(labels_2_ops)
    
    
    return x_samples, y_samples, y_samples_real



y_data = np.load("Y_data_segment_filt_train.npy")

embed_classes_unique = np.unique(y_data)
embed_classes_unique = np.sort(embed_classes_unique)
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


    
device = 'cuda'  # cuda mps
T = 400
N_ec = 11       # number of real operation classes (must match training)
    
# --- Load model ---
model = UNet2D(in_channels=3, base_channels=32, time_emb_dim=128, 
                   num_ops=N_ec, op_emb_dim=64)
    
# Load checkpoint and apply EMA weights
ckpt = torch.load("saved_diff_model/ckpt_melener_f.pt", map_location=device)
# Find the EMA key in the checkpoint
ema_key = [k for k in ckpt.keys() if k.startswith("ema")][0]
model_key = [k for k in ckpt.keys() if k.startswith("diffusion")][0]
   
# Load EMA weights (recommended for sampling)
# The EMA shadow dict has the same keys as model.state_dict()
model.load_state_dict(ckpt[ema_key])
model = model.to(device)
model.eval()
    
print(f"Model loaded. Null class index = {model.null_class_idx}")
    

T = 400
ops_to_generate = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # contiguous indices
samples_per_op = 40
    

guidance_scale = 1.32   # around 1.32 

per_op_guidance = {
        3: 3.2,   # OP 4
        4: 1.12,   # OP 5
        5: 1.12,    # OP 7
        6: 4.05,    # OP 8
        8: 3.6,   # OP 11
        10: 4.5   # OP 14 
        # all others will use the global default
}
    

# GENERATE FOR QUALITY ESTIMATE 
samples_per_op_q = [samples_per_op for _ in range(11)]
x_samples, _, y_samples_real = get_sampled_data(ops_to_generate, samples_per_op_q, guidance_scale)

print(f"\nTotal generated: {x_samples.shape}")
np.save("saved_generated_data/x_samples_diffusion_melener.npy", x_samples)
np.save("saved_generated_data/y_samples_diffusion_melener.npy", y_samples_real)
print("Saved to x_samples_diffusion_melener.npy, y_samples_diffusion_melener.npy")


# GENERATE FOR AUGMENTATION 
#samples_per_op_synth_a = [300,300,300,0,320,100,200,0,0,100,350]
#samples_per_op_synth_a = [10 for _ in range(11)]
#x_samples, y_samples, _ = get_sampled_data(ops_to_generate, samples_per_op_synth_a, guidance_scale)


#print(f"\nTotal generated: {x_samples.shape}")
#np.save("saved_generated_data/X_DIFF_AUG_FEAT.npy", x_samples)
#np.save("saved_generated_data/Y_DIFF_AUG_FEAT.npy", y_samples)

#print("Saved X_DIFF_AUG_FEAT Y_DIFF_AUG_FEAT.npy, Y_DIFF_AUG_FEAT.npy")



