import numpy as np
import argparse
import os 
import itertools
import TorchClassificationModels as tm
import torch
import torch.utils.data as data
import copy
from sklearn.metrics import confusion_matrix
from torch.utils.data import WeightedRandomSampler


parser = argparse.ArgumentParser()

parser.add_argument("--selected_feature",
                    type = str, default = "MEL_ENERGY") # FFT, STFT, MEL_ENERGY, MelLog,TIME
parser.add_argument("--batch_size",
                    type = int, default = 64)
parser.add_argument("--decision_treshold",
                    type = float, default = 0.3)

parser.add_argument("--one_axis", action="store_true")
parser.add_argument("--use_dnn", action="store_true")
parser.add_argument("--focal_loss", action="store_true")


def get_data(machines, process_names, test_d, add_data = None):
    
    x_v = []
    y_v = []
    
    for mn, pn in itertools.product(machines, process_names):

        save_name_X = 'saved_features/' + selected_feature + '/' + 'X_' + mn + '_' + pn + '.npy'
        save_name_Y = 'saved_features/' + selected_feature + '/' + 'Y_' + mn + '_' + pn + '.npy'

        if os.path.exists(save_name_X):
            x_v.append(np.load(save_name_X))
            y_v.append(np.load(save_name_Y))

    x_v = np.concatenate(x_v, 0) # N, 3 , 2048, -- 2048 jer je fft
    y_v = np.concatenate(y_v)
    y_v = y_v.astype(int)

    cnn_in_layers = 3
    if args.one_axis:
        x_v = x_v[:,0:1,:]
        cnn_in_layers = 1
       
    batch_size = args.batch_size
    use_dnn = args.use_dnn

    if add_data is not None:
       x_v = np.concatenate((x_v, add_data), 0) 
       y_v = np.concatenate((y_v, np.ones(len(add_data))), 0)  
       y_v = y_v.astype(int)
    
    if use_dnn:
        x_v = np.reshape(x_v, (x_v.shape[0], x_v.shape[1]*x_v.shape[2]))  

    class_counts = np.bincount(y_v)     
    sample_weights = np.where(y_v==0, 1.0/class_counts[0], 1.0/class_counts[1])
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)       
    x_v = tm.labeled_dataset(x_v, y_v)
    
    if test_d:
        x_v = data.DataLoader(x_v, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
    else:
        x_v = data.DataLoader(x_v, batch_size=batch_size, num_workers=0, sampler=sampler,  drop_last=True) # sampler=sampler,
            
    
    return x_v, y_v, cnn_in_layers
        
    
def compute_test():
    
    model.eval()
    y_preds = []

    with torch.no_grad():
        for x, y in x_test:
            x = x.float().to(device)
            yp = model(x)
            y_preds.append(np.argmax(yp.detach().cpu().numpy(), axis=1))
   
    y_preds = np.concatenate(y_preds)

    acc = sum(y_preds == y_test)/len(y_test)
    print(f"Accuracy: {100*acc}")

    cm = confusion_matrix(y_test, y_preds)
    
    cm_norm = copy.copy(cm)
    cm_norm = np.array(cm_norm, dtype=float)

    for i in range(0, 2):
        cm_norm[i,:] = cm_norm[i,:]/np.sum(cm, axis=1)[i]

    print('\nConfusion matrix: ' )
    print(np.array_str(cm_norm, precision=4, suppress_small=True))
    print('\n')    
   
def compute_test_focal(decision_treshold):
    
    model.eval()
    y_preds = []
    with torch.no_grad():
        for x, y in x_test:
            x = x.float().to(device)
            logits = model(x).squeeze(-1)          # [N]
            probs = torch.sigmoid(logits)          # p(y=1)
            # pick a conservative initial threshold; tune on a val set later
            y_hat = (probs >= decision_treshold).long()          # try 0.2–0.35 for 4–6% positives
            y_preds.append(y_hat.cpu().numpy())
    y_preds = np.concatenate(y_preds)

    acc = (y_preds == y_test).mean()
    print(f"Accuracy: {100*acc}")

    cm = confusion_matrix(y_test, y_preds)
    cm_norm = cm.astype(float)
    for i in range(2):
        cm_norm[i,:] = cm_norm[i,:] / cm[i,:].sum()
    print('\nConfusion matrix: ')
    print(np.array_str(cm_norm, precision=4, suppress_small=True))
    print('\n')
    
args = parser.parse_args()
selected_feature = args.selected_feature
use_dnn = args.use_dnn
x_add = None
      
machines_test = ["M02"]
process_names_test = ["OP01", "OP02", "OP03", "OP04", "OP05", "OP07","OP08", "OP10", "OP11", "OP12", "OP14"]  # 1,2,4,7,10

print("Loading data ...")
x_test, y_test, cnn_in_layers = get_data(machines_test, process_names_test, test_d = True) 

print("Data loaded \n")
print("Distribution of data in the training dataset (normal/fault)")
print(sum(y_test==0))
print(sum(y_test==1))   
    

# %%
####################### TEST DATA ######################

device = 'cuda' # cuda, mps
N_out = 2


if args.focal_loss:
    N_out = 1

if selected_feature == "FFT" or selected_feature == "TIME":
    if use_dnn:
        in_channels = 1 if args.one_axis else 3
        model = tm.DNN_Model(in_dim = 4096, in_channels=in_channels, n_hidden=[500]).to(device)
    else:    
        model = tm.VGG1D_Model(in_channels = cnn_in_layers, n_chans1=[32,32,32,32], k_size = [5,5,3,3], N_out = N_out).to(device)
else:        
    model = tm.VGG_Model(in_channels = cnn_in_layers, n_chans1=[32,32,32], k_size = [3,3,3], N_out = N_out).to(device)

# vgg_model_MEL_ENERGY_three_axis_M01_M03_SYNTH_PAPER
# vgg_model_MEL_ENERGY_three_axis_M01_M03_BEST_BASE_PAPER
model.load_state_dict(torch.load("SavedClassificationModels/vgg_model_MEL_ENERGY_three_axis_M01_M03_BEST_BASE_PAPER.pth"))  # map_location=torch.device('mps')
model.number_of_params()  # prints number of paramsh
model.eval()

if args.focal_loss:
   compute_test_focal(args.decision_treshold)
else:     
   compute_test()

for pns in process_names_test:
    x_test, y_test, _ = get_data(machines_test, [pns], test_d = True) 

    print(f"\nResults on OP {pns}")
    if args.focal_loss:
       compute_test_focal(args.decision_treshold)
    else:     
       compute_test()
       
       
       
