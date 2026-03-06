import numpy as np
import torch
from Functions_CheckQuality_FeatureSpace import _median_heuristic_sigma
import TorchClassificationModels as tm
from  Functions_CheckQuality_FeatureSpace import *
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split       
import matplotlib.pyplot as plt




def plot_roc_curve(true_y, y_prob):
    """
    plots the roc curve based of the probabilities
    """

    fpr, tpr, thresholds = roc_curve(true_y, y_prob)
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate') 
    
    
x_train_melener = np.load("X_melener_segment_filt_train.npy")
x_test_melener = np.load("X_melener_segment_filt_test.npy")
y_train = np.load("Y_data_segment_filt_train.npy")
y_test = np.load("Y_data_segment_filt_test.npy")


x_data = np.concatenate((x_train_melener, x_test_melener), 0)
y_data = np.concatenate((y_train, y_test))

x_train_melener, x_test_melener, y_train, y_test= train_test_split(x_data, y_data, test_size=0.05, random_state=42)


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



# FAKE DATA 
x_fake_melener = np.load("saved_generated_data/x_samples_diffusion_melener_paper.npy")  
y_fake_melener = np.load("saved_generated_data/y_samples_diffusion_melener_paper.npy")


def get_mmd_per_op(real_emb, y_test, sigma_=None):

    for yi in np.unique(y_test):
        mmd2 = mmd_rbf(real_emb[y_test==yi], synth_emb[y_fake_melener==yi], sigma=sigma_, max_points=4000, seed=0, device='cpu')
        print(f"OP {yi} - MMD^2 (RBF) on embeddings: {mmd2:.6f}")

def get_mmd_per_op_train(real_emb, y_test, synth_emb, y_synth, sigma_=None):

    for yi in np.unique(y_test):
        mmd2 = mmd_rbf(real_emb[y_test==yi], synth_emb[y_synth==yi], sigma=sigma_, max_points=4000, seed=0, device='cpu')
        print(f"OP {yi} - MMD^2 (RBF) on embeddings: {mmd2:.6f}")



def get_knn_per_op(real_emb, y_emb, k=5):
    for yi in np.unique(y_emb):
            if yi < 14 or k < 3:
                pr = knn_precision_recall(real_emb[y_emb==yi], synth_emb[y_fake_melener==yi], k=k, metric='euclidean', max_points=10000, seed=0)
                print(f"OP {yi}  - kNN Precision/Recall (k={pr['k']}): precision={pr['precision']:.4f}, recall={pr['recall']:.4f}")



# %% CHECK CLASSIFICAITON PERFORMANCE 
device = 'cpu'
model = tm.VGG_Model(in_channels = 3, n_chans1=[32,32,32], k_size = [3,3,3], N_out = 1).to(device)
model.load_state_dict(torch.load("saved_generated_data/vgg_model_MEL_ENERGY_three_axis_M01_M03_base_q.pth", map_location=torch.device('mps'))) # vgg_model_MEL_ENERGY_three_axis_M02_M03_best
model.eval() 

# %% DISTRIBUTION QUALITY METRICS IN EMBEDDING SPACE (pre-fc1)

real_emb_train = extract_fc1_input_embeddings(model, x_train_melener, device=device, batch_size=256)
#sigma_fixed = _median_heuristic_sigma(real_emb_train, max_points=5000)# 0.17823803424835205


sigma_fixed = 0.17823803424835205 # M1_M3

print(f"sigma_fixed is:", {sigma_fixed})

real_emb_train_1 = extract_fc1_input_embeddings(model, x_train_melener[0:2000], device=device, batch_size=256)
real_emb_train_2 = extract_fc1_input_embeddings(model, x_train_melener[2000:4000], device=device, batch_size=256)
real_emb = extract_fc1_input_embeddings(model, x_test_melener, device=device, batch_size=256)

synth_emb = extract_fc1_input_embeddings(model, x_fake_melener, device=device, batch_size=256)

# 1) MMD (RBF) on embeddings
mmd2 = mmd_rbf(real_emb_train_1, real_emb_train_2, sigma=sigma_fixed, max_points=4000, seed=0, device='cpu')
print(f"MMD^2 (RBF) on train vs train embeddings: {mmd2:.6f}")

mmd2 = mmd_rbf(real_emb_train_1, real_emb, sigma=sigma_fixed, max_points=4000, seed=0, device='cpu')
print(f"MMD^2 (RBF) on train vs test embeddings: {mmd2:.6f}")

mmd2 = mmd_rbf(real_emb_train_1, synth_emb, sigma=sigma_fixed, max_points=4000, seed=0, device='cpu')
print(f"MMD^2 (RBF) on embeddings vs train 1: {mmd2:.6f}")

mmd2 = mmd_rbf(real_emb_train_2, synth_emb, sigma=sigma_fixed, max_points=4000, seed=0, device='cpu')
print(f"MMD^2 (RBF) on embeddings vs train 2: {mmd2:.6f}")
    
mmd2 = mmd_rbf(real_emb, synth_emb, sigma=sigma_fixed, max_points=4000, seed=0, device='cpu')
print(f"MMD^2 (RBF) on embeddings vs test: {mmd2:.6f}")


# 2) kNN precision/recall (coverage) on embeddings
pr = knn_precision_recall(real_emb, synth_emb, k=5, metric='euclidean', max_points=10000, seed=0)
print(f"kNN Precision/Recall (k={pr['k']}): precision={pr['precision']:.4f}, recall={pr['recall']:.4f}")


y_pred = model(torch.tensor(x_fake_melener).float().to(device))
y_pred = torch.sigmoid(y_pred).detach().cpu().numpy()
y_pred = np.concatenate(y_pred)
y_hat = (y_pred >= 0.3)

acc = (1 - sum(y_hat)/len(y_hat))*100
print(f"Detection accuracy of fake as anomaly: {acc}")

ii = np.argsort(y_pred, axis=-1, kind='quicksort', order=None)
ii = ii[::-1]
y_fake_hardest = y_fake_melener[ii]


# %%

get_mmd_per_op(real_emb, y_test, sigma_fixed)
#get_mmd_per_op_train(real_emb, y_test, real_emb_train, y_train, sigma_=None)

#get_mmd_per_op(real_emb_train_1, y_train[0:500])
#get_mmd_per_op(real_emb_train_2, y_train[600:1100])

#get_knn_per_op(real_emb_train_1, y_train[0:500], 5)
get_knn_per_op(real_emb, y_test, 5)






