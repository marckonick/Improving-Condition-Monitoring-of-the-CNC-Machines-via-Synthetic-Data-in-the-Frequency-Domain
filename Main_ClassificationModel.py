import numpy as np
import argparse
import os 
import itertools
import TorchClassificationModels as tm
import torch
import torch.utils.data as data
import torch.optim as optim
import copy
from sklearn.metrics import confusion_matrix
from torch.utils.data import WeightedRandomSampler

    
parser = argparse.ArgumentParser()

parser.add_argument("--selected_feature",
                    type = str, default = "MEL_ENERGY") # FFT, STFT, MEL_ENERGY, MelLog
parser.add_argument("--batch_size",
                    type = int, default = 64)
parser.add_argument("--n_epochs",
                    type = int, default = 60)
parser.add_argument("--imb_weight",
                    type = float, default = 80.0)
parser.add_argument("--decision_treshold",
                    type = float, default = 0.3)

parser.add_argument("--one_axis", action="store_true")
parser.add_argument("--add_synth", action="store_true")
parser.add_argument("--use_dnn", action="store_true")
parser.add_argument("--focal_loss", action="store_true")

# %%


def get_data(machines, process_names, test_d, add_data = None):
    
    x_v = []
    y_v = []
    
    for mn, pn in itertools.product(machines, process_names):

        save_name_X = '../saved_features/' + selected_feature + '/' + 'X_' + mn + '_' + pn + '.npy'
        save_name_Y = '../saved_features/' + selected_feature + '/' + 'Y_' + mn + '_' + pn + '.npy'

        if os.path.exists(save_name_X):
            x_v.append(np.load(save_name_X))
            y_v.append(np.load(save_name_Y))

    x_v = np.concatenate(x_v, 0)
    y_v = np.concatenate(y_v)
    y_v = y_v.astype(int)
    
    print(f"x_v je {x_v.shape}")
    
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
        x_v = data.DataLoader(x_v, batch_size=batch_size, sampler=sampler, num_workers=0, drop_last=True) #  sampler=sampler,
            
    
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
   
# DECISION TRESHOLD 
def compute_test_focal(decision_treshold):
    
    model.eval()
    y_preds = []
    with torch.no_grad():
        for x, y in x_test:
            x = x.to(device).float()
            logits = model(x).squeeze(-1)          
            probs = torch.sigmoid(logits)          
            # pick a conservative initial threshold; tune on a val set later
            y_hat = (probs >= decision_treshold).long()          
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

    
def get_synth_data_():
    
    if selected_feature == 'MelLog':
        x_add = np.load("x_samples_diffusion_MelLog.npy")
    elif selected_feature == 'MEL_ENERGY':
        x_add = np.load("saved_generated_data/X_DIFF_AUG_FEAT.npy") 
    elif selected_feature == 'FFT':
        x_add = np.load("x_samples_diffusion_FFT.npy")
    elif selected_feature == 'TIME':
        x_add = np.load("x_ts_aug.npy")
        
    #x_add = x_add[:,None,:]
    
    print(f"x_add size: {x_add.shape}")
    
    return x_add      
    
args = parser.parse_args()
selected_feature = args.selected_feature
add_synth = args.add_synth 
use_dnn = args.use_dnn

########
x_add = None
if add_synth:
    print("Using synth data...")
    x_add = get_synth_data_()    
############
      
machines = ["M01", "M03"] 
process_names = ["OP01"] #, "OP02", "OP03", "OP04", "OP05", "OP07","OP08", "OP10", "OP11", "OP12", "OP14"] 

machines_test = ["M02"]
process_names_test = ["OP01"]#, "OP02", "OP03", "OP04", "OP05", "OP07","OP08", "OP10", "OP11", "OP12", "OP14"]


print("Loading data ...")

x_train, y_train, cnn_in_layers = get_data(machines, process_names, test_d = False, add_data = x_add) 
x_test, y_test, _ = get_data(machines_test, process_names_test, test_d = True) 

print("Data loaded \n")

print("Distribution of data in the training dataset (normal/fault)")
print(sum(y_train==0))
print(sum(y_train==1))   

# %%
device = 'cuda' # cuda mps 
N_out = 2
if args.focal_loss:
    N_out = 1

if args.one_axis:
    add_string = "one_axis"
else:
    add_string = "three_axis"

    
if selected_feature == "FFT" or selected_feature == "TIME":
    if use_dnn:
        in_channels = 1 if args.one_axis else 3
        model = tm.DNN_Model(in_dim = 4096, in_channels=in_channels, n_hidden=[500]).to(device)
    else:    
        model = tm.VGG1D_Model(in_channels = cnn_in_layers, n_chans1=[32,32,32,23], k_size = [5,3,3,3], N_out = N_out).to(device)
else:        
    model = tm.VGG_Model(in_channels = cnn_in_layers, n_chans1=[32,32,32], k_size = [3,3,3], N_out = N_out).to(device)

    
imb_weight = args.imb_weight
optimizer = optim.Adam(model.parameters(), lr=2e-3) # 1e-5 mellog

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
weight_c = torch.tensor([1.0, imb_weight]).to(device)


if args.focal_loss:
    loss_fn = tm.FocalLoss(alpha=0.52, gamma = 1.6, from_logits=True, reduction = 'sum')
    print("Using focal loss...")
else:
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight_c)

print("Distribution of data in the training dataset (normal/fault)")

model.number_of_params()  # prints number of params
model.train()
n_epochs = args.n_epochs

decision_treshold = args.decision_treshold

for epoch in range(1, n_epochs + 1):
    loss_train = 0.0
    for x, y in x_train:
               x = x.float().to(device=device)
               y = y.to(device=device)

               outputs = model(x)
               loss = loss_fn(outputs, y.long()) 

               optimizer.zero_grad()
               loss.backward()
               torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
               optimizer.step()
               loss_train += loss.item()

    scheduler.step()       
    print(f" Epoch {epoch}/{n_epochs}, loss = {float(loss_train/len(x_train))}, lr = {optimizer.param_groups[0]['lr']:.6f}")
    
    if epoch % 10 == 0:
        if args.focal_loss:
            compute_test_focal(decision_treshold)
        else:     
            compute_test() # THIS SHOULD BE DONE WITH ENTIRE x_test 
        model.train()  
        
    if epoch % 10 == 0:  
        torch.save(model.state_dict(), f"SavedClassificationModels/vgg_model_{selected_feature}_{add_string}_{machines[0]}_{machines[1]}_epoch_{epoch}.pth")    
    
    #if epoch % 1152 == 0:
    #   for pns in process_names_test:
    #         x_test, y_test, _ = get_data(machines_test, [pns], test_d = True) #WATCH THIS LINE 
    #         print(f"\nResults on OP {pns}")
    #         if args.focal_loss:
    #            compute_test_focal(decision_treshold)
    #         else:     
    #            compute_test()
    #         model.train()
            



