import os
import sys 
from pathlib import Path
import itertools 
import numpy as np 
import matplotlib.pyplot as plt 
folder_path = os.path.abspath("../CNC_Machining-main/")
sys.path.append(folder_path)
from utils import data_loader_utils
import sys
import pywt 
import copy
from sklearn.model_selection import train_test_split
import Functions_FeatureExtraction as FFE
import pickle 
from scipy import stats



def normalize_clip(x, center, scale, clip=5.0):
    x = (x - center) / (scale + 1e-6)
    return np.clip(x, -clip, clip)

def segment_array(data, N_leng, step=0):

    data = np.asarray(data)
    if data.ndim == 1:
        data = data[:, None]  

    if N_leng <= 0:
        raise ValueError("N_leng must be positive.")
    if not (0 <= step):
        raise ValueError("overlap must satisfy 0 <= overlap < N_leng.")

    N = data.shape[0]
    if N < N_leng:
        # No full window fits
        return np.empty((0, N_leng, data.shape[1]), dtype=data.dtype)

   
    starts = np.arange(0, N - N_leng + 1, step)
    segments = np.stack([data[s:s + N_leng] for s in starts], axis=0)

   
    if segments.shape[-1] == 1:
        return segments[..., 0]
    return segments


def plot_wpd(components, nodes): 
    
    # Create the 4x4 subplot grid
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    fig.tight_layout(pad=4.0)  # Add padding between subplots

    # Plot each signal in its subplot with corresponding title
    for i, ax in enumerate(axes.flat):
        if i < 16:  # Ensure we don't exceed our data
            ax.plot(components[i])
            ax.set_title(nodes[i], fontsize=10)
            ax.grid(True)
        else:
            ax.axis('off')  # Turn off unused subplots if any

    plt.suptitle('4x4 Grid of Signals with Individual Titles', fontsize=14, y=1.02)
    plt.show()
    
def reconstruct_signal_from_wavelet_packet(wp, nodes, components, original_length):

    
    reconstructed_wp = pywt.WaveletPacket(data=None, wavelet=wp.wavelet, mode=wp.mode)

    for node_path, component in zip(nodes, components):
        reconstructed_wp[node_path] = component
    
    reconstructed_signal = reconstructed_wp.reconstruct(update=False)
    
    return reconstructed_signal[:original_length]
    
def perform_wavelet_packet_decomposition(signal, wavelet='db4', max_level=4, plot_on = False):


    # Perform wavelet packet decomposition
    wp = pywt.WaveletPacket(data=signal, wavelet=wavelet, mode='symmetric', maxlevel=max_level)
    
    # Calculate how many nodes we have at each level
    #  NODES
    #   * 
    #  * *
    # ** ** 
    nodes_per_level = [2**level for level in range(max_level + 1)] #### 
    total_nodes = sum(nodes_per_level)
    
    nodes = [node.path for node in wp.get_level(max_level, order='natural')]
    components = [wp[node].data for node in nodes]
    
    if plot_on:
        plot_wpd(components, nodes) 

    return wp, nodes, components

def perform_dwt_hard_denoising(orig_signal, threshold, comp_to_keep):
    
    wp, nodes, components = perform_wavelet_packet_decomposition(orig_signal , wavelet='db4', max_level=4)

    new_components = copy.copy(components)

    #for cc in new_components:
    #    cc[abs(cc) <= threshold] = 0

    for jj in range(len(new_components)):
       if jj not in comp_to_keep:
           new_components[jj] *= 0
    

    reconstructed_signal = reconstruct_signal_from_wavelet_packet(wp, nodes, new_components, len(orig_signal))
    
    return reconstructed_signal

def plot_time_domain(x_ex):
    
    t = np.arange(0, len(x_ex)/Fs, 1/Fs)

    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(8, 10), sharex=True)

    for i in range(3):
        axs[i].plot(t, x_ex[:, i])
        axs[i].set_title(f"axis {i}", fontsize = 16)
        
    axs[2].set_xlabel("time (s)", fontsize = 16)
    fig.suptitle(f"{y_data}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
   
def normalize_data_per_operation(machines_n, labels_n, process_names_n, q=0.995):

  print("Normalizing signals operation started! \n This should be done for normal and faulty signals together!!! \n")
  OPs_stats = {}
  for process_name in process_names:
    data_lists = []
    for machine, label in itertools.product(machines, labels):
        data_path = os.path.join(path_to_dataset, machine, process_name, label)
        data_list, data_label = data_loader_utils.load_tool_research_data(data_path, label=label)
        if len(data_list) > 0:
            data_lists.append(np.concatenate(data_list, 0))
    data_lists = np.concatenate(data_lists, 0)    
    
    mu = np.mean(data_lists, 0) # np.median, mean
    MAD = stats.median_abs_deviation(data_lists,0)
    RMS = np.sqrt(np.mean(np.square(data_lists), 0))
    center = np.median(data_lists, axis=0)                         # (3,)
    scale  = np.quantile(np.abs(data_lists - center), q, axis=0)   # (3,)
    
    
    OPs_stats[process_name + "_mu"] = mu
    OPs_stats[process_name + "_mad"] = MAD   
    OPs_stats[process_name + "_rms"] = RMS   
    OPs_stats[process_name + "_center"] = center
    OPs_stats[process_name + "_scale"]  = scale
    
  with open("OPs_stats", 'wb') as f:
        pickle.dump(OPs_stats, f)
        print("Saved stats for the process statistics!!!")
        
        
  return OPs_stats 
    

# GET ALL DATA 
machines = ["M01", "M03"]#["M01","M02","M03"]
process_names = ["OP01","OP02","OP03","OP04","OP05","OP07","OP08","OP10","OP11","OP12","OP14"]

labels = ["bad"]
path_to_dataset = Path("../CNC_Machining-main/data/").absolute()


Fs = 2e3
X_data = []
y_data = []
y_operations = []
y_segment_operations = []

OPs_stats_orig = normalize_data_per_operation(machines_n=machines, labels_n=["bad", "good"], process_names_n=process_names)
# Load the dictionary
with open("OPs_stats", 'rb') as f:
    OPs_stats = pickle.load(f)    
    
# %%
# LOAD ORIGINAL DATA
for process_name, machine, label in itertools.product(process_names, machines, labels):
    data_path = os.path.join(path_to_dataset, machine, process_name, label)
    data_list, data_label = data_loader_utils.load_tool_research_data(data_path, label=label)
      
    for i in range(len(data_list)):    
        data_list[i] = normalize_clip(data_list[i], OPs_stats[process_name + "_center"], OPs_stats[process_name + "_scale"], clip=150.0)
  
    
    X_data.extend(data_list)
    y_data.extend(data_label)
    y_operations.append(int(process_name[-2:])*np.ones(len(data_list)))
 
   
# %% TIME-DOMAIN
    
#plot_time_domain(X_data[0])
    
# %% DWT DENOISING

X_data_filtered = []
for x in X_data:
    x_filt = np.zeros_like(x)
    for j in range(3):
        x_filt[:,j] = perform_dwt_hard_denoising(x[:,j], threshold = [400,600,800], comp_to_keep=[0,1,2])
    X_data_filtered.append(x_filt) 
    
# %%
# SEGMENTATION 

y_operations = np.concatenate(y_operations)    
N_segment = 4096
N_step = 512 # 1000 is used for feature extraction  
X_data_segment = []
X_data_segment_filt = []

for x, xf, yop in zip(X_data, X_data_filtered, y_operations):
    X_data_segment.append(segment_array(x, N_segment, N_step))
    X_data_segment_filt.append(segment_array(xf, N_segment, N_step))
    y_segment_operations.append(yop*np.ones(X_data_segment[-1].shape[0]))
    
    
X_data_segment = np.concatenate(X_data_segment, axis=0)    
X_data_segment_filt = np.concatenate(X_data_segment_filt, axis=0)    
y_segment_operations = np.concatenate(y_segment_operations, axis=0)    

x_train, x_test, y_train, y_test = train_test_split(X_data_segment_filt, y_segment_operations, test_size=0.05, random_state=42)

gain = 3.4
x_train *= gain
x_test *= gain

x_train = np.clip(x_train, -5.0, 5.0)  # keep stability
x_test = np.clip(x_test, -5.0, 5.0)  # keep stability

np.save("X_data_segment_filt_train.npy", x_train)
np.save("Y_data_segment_filt_train.npy", y_train.astype(np.int32))

np.save("X_data_segment_filt_test.npy", x_test)
np.save("Y_data_segment_filt_test.npy", y_test.astype(np.int32))

print(x_train.shape)
print(x_test.shape)

x = copy.copy(x_train[:,:,0])
x = x.reshape(x.shape[0], -1)

y = copy.copy(x_train[:,:,1])
y = y.reshape(y.shape[0], -1)

z = copy.copy(x_train[:,:,2])
z = z.reshape(z.shape[0], -1)

rms_per_sample = np.sqrt(np.mean(x**2, axis=1))
print("X - AXIS")
print("dataset std:", x.std())
print("rms median:", np.median(rms_per_sample))
print("rms p90:", np.percentile(rms_per_sample, 90))
print("abs max:", np.max(np.abs(x)))

print("Y - AXIS")
print("dataset std:", y.std())
print("rms median:", np.median(np.sqrt(np.mean(y**2, axis=1))))
print("rms p90:", np.percentile(np.sqrt(np.mean(y**2, axis=1)), 90))
print("abs max:", np.max(np.abs(y)))

print("Z - AXIS")
print("dataset std:", z.std())
print("rms median:", np.median(np.sqrt(np.mean(z**2, axis=1))))
print("rms p90:", np.percentile(np.sqrt(np.mean(z**2, axis=1)), 90))
print("abs max:", np.max(np.abs(z)))


# %% COMPUTING 

print("\n Performing MelEnergy Feature extraction")
X_melener_train = FFE.extract_melener_features_short(x_train)
X_melener_test = FFE.extract_melener_features_short(x_test)

X_me_mean = np.mean(X_melener_train, axis=(0, 3), keepdims=True)
X_me_std  = np.std(X_melener_train, axis=(0, 3), keepdims=True) + 1e-6

X_melener_train = (X_melener_train - X_me_mean) / X_me_std

# TEST 
X_me_mean = np.mean(X_melener_test, axis=(0, 3), keepdims=True)
X_me_std  = np.std(X_melener_test, axis=(0, 3), keepdims=True) + 1e-6

X_melener_test = (X_melener_test - X_me_mean) / X_me_std


np.save("X_melener_segment_filt_train.npy", X_melener_train)
np.save("X_melener_segment_filt_test.npy", X_melener_test)

print("\n Done")
  
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
