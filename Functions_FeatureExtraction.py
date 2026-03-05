import numpy as np
from scipy.fft import fft, fftfreq
from scipy import signal 
from scipy.signal import butter, lfilter
import librosa 
import sys
import os
from pathlib import Path
import itertools 
folder_path = os.path.abspath("../CNC_Machining-main/")
sys.path.append(folder_path)
from utils import data_loader_utils
import copy
import pywt
import pickle
from scipy import stats

labels = ["good","bad"]
path_to_dataset = Path("../CNC_Machining-main/data/").absolute()
Fs = 2000

def normalize_clip(x, center, scale, clip=5.0):
    x = (x - center) / (scale + 1e-6)
    return np.clip(x, -clip, clip)

def standardize_minus1_to_1(data):
    
    """
    Standardization by axis 
    
    Standardize each column of `data` to the range [-1, 1].

    Parameters
    ----------
    data : np.ndarray
        Shape (N, 3) or (N, D).

    Returns
    -------
    np.ndarray
        Shape (N, 3) scaled column-wise to [-1, 1].
    """
    data = np.asarray(data, dtype=float)
    min_vals = data.min(axis=0)
    max_vals = data.max(axis=0)

    # Avoid division by zero
    ranges = np.where(max_vals - min_vals == 0, 1, max_vals - min_vals)

    scaled = 2 * (data - min_vals) / ranges - 1
    return scaled


def segment_array(data, N_leng, step=0):

    data = np.asarray(data)
    if data.ndim == 1:
        data = data[:, None]  # make it (N, 1)

    if N_leng <= 0:
        raise ValueError("N_leng must be positive.")
    if not (0 <= step):
        raise ValueError("overlap must satisfy 0 <= overlap < N_leng.")

    N = data.shape[0]
    if N < N_leng:
        # No full window fits
        return np.empty((0, N_leng, data.shape[1]), dtype=data.dtype)

    #step = N_leng - overlap  # hop size
    starts = np.arange(0, N - N_leng + 1, step)
    segments = np.stack([data[s:s + N_leng] for s in starts], axis=0)

    # If original was 1D, squeeze the last dim for convenience
    if segments.shape[-1] == 1:
        return segments[..., 0]
    return segments


def extract_fft_features_short(X):

    X_features_ml = []
    Fs = 2e3
    
    for xs in X:
        X_features_ml.append(perform_fft(xs, Fs))

    X_features_ml = np.stack(X_features_ml)
    
    return X_features_ml


def extract_mellog_features_short(X, win_len=128, hop_l = 64, n_mels = 64):

    X_features_ml = []
    Fs = 2e3
    
    for xs in X:
        X_features_ml.append(perform_mellog(xs, Fs, win_len, hop_l, n_mels))

    X_features_ml = np.stack(X_features_ml)
    
    return X_features_ml
    

    
def extract_melener_features_short(X, win_len = 128, hop_l = 64, frames = 2, power = 2.0, n_mels = 64):

    X_features_ml = []
    Fs = 2e3
    kwargs = {'frames_mel':frames, 'win_len':win_len, 'hop_l':hop_l, 'n_mels':n_mels, 'power_mel':power}
    for xs in X:
        x_feat_temp_ = [] 
        for i in range(3):
           x_temp_ = xs[:, i]
           x_feat_temp_.append(perform_mel_ener(x_temp_, Fs, **kwargs)) # - 4096
             
        X_features_ml.append(np.stack(x_feat_temp_)) 
        
    X_features_ml = np.stack(X_features_ml)
    
    return X_features_ml

def perform_wavelet_packet_decomposition(signal, wavelet='db4', max_level=4, plot_on = False):
    """
    Perform wavelet packet decomposition and plot all components.
    
    Parameters:
        signal (numpy.ndarray): Input signal of shape (8192,)
        wavelet (str): Wavelet to use (default: 'db1')
        max_level (int): Maximum decomposition level (default: 4)
    """
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
    

    return wp, nodes, components

def reconstruct_signal_from_wavelet_packet(wp, nodes, components, original_length):
 
    # Create a new wavelet packet structure for reconstruction
    reconstructed_wp = pywt.WaveletPacket(data=None, wavelet=wp.wavelet, mode=wp.mode)
    
    # Assign the components back to their nodes
    for node_path, component in zip(nodes, components):
        reconstructed_wp[node_path] = component
    
    # Perform the reconstruction
    reconstructed_signal = reconstructed_wp.reconstruct(update=False)
    
    # Trim to original length in case of odd-length signals
    return reconstructed_signal[:original_length]


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


def perform_stft(X_ex, Fs, win_len, overlap_l):
 
    # noverlap - Number of points to overlap between segments.
    _,_, Xxx = signal.stft(X_ex, Fs, nperseg=win_len, noverlap=overlap_l)

    return np.abs(Xxx)

def perform_fft(X_ex, Fs):

    T = 1.0/Fs
    N = len(X_ex)
    
    yf = fft(X_ex)
    xf = fftfreq(N, T)[:N//2]

    return 2.0/N*np.abs(yf[0:N//2])

def perform_mellog(X_ex, Fs, win_len, hop_l, n_mels):
    mel_signal = librosa.feature.melspectrogram(y = np.array(X_ex, dtype=np.float32)  , sr = Fs, hop_length = hop_l, n_fft = win_len, n_mels=n_mels)
    spectrogram = np.abs(mel_signal)
    spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    
    
    return spectrogram

def perform_mel_ener(X_ex, Fs, **kwargs):
    
    n_mels = kwargs['n_mels']
    power = kwargs['power_mel']
    frames = kwargs['frames_mel']
    win_len = kwargs['win_len']
    hop_l = kwargs['hop_l']
    
    dims = n_mels * frames

    
    mel_spectrogram = librosa.feature.melspectrogram(y=np.array(X_ex, dtype=np.float32),
                                                     sr=Fs,
                                                     n_fft= win_len,
                                                     hop_length = hop_l,
                                                     n_mels=n_mels,
                                                     power=power)
    
    log_mel_spectrogram = 20.0 / power * np.log10(mel_spectrogram + sys.float_info.epsilon)
    # 04 calculate total vector size
    vector_array_size = len(log_mel_spectrogram[0, :]) - frames + 1    

    # 05 skip too short clips
    if vector_array_size < 1:
        return np.empty((0, dims))

    # 06 generate feature vectors by concatenating multiframes
    vector_array = np.zeros((vector_array_size, dims))
    for t in range(frames):
        vector_array[:, n_mels * t: n_mels * (t + 1)] = log_mel_spectrogram[:, t: t + vector_array_size].T
        
    return vector_array



def preprocess_filter_signal(X_ex, filter_type = 'median', Fs=2000, filter_order = 17, cutoff = 1000, tshs = None):
    
    if filter_type == 'median':
        X_ex_filt = np.zeros_like(X_ex)
        for j in range(3):
            X_ex_filt[:,j] = signal.medfilt(X_ex[:,j], filter_order)
        return X_ex_filt
    elif filter_type == 'DWT':
        X_ex_filt = np.zeros_like(X_ex)
        for j in range(3):
            X_ex_filt[:,j] = perform_dwt_hard_denoising(X_ex[:,j], threshold = tshs[j], comp_to_keep=[0,1,2])           
        return X_ex_filt
    elif filter_type == 'lowpass':
         X_ex_filt = np.zeros_like(X_ex)
         b, a = butter(filter_order, cutoff, fs=Fs, btype='low', analog=False)  
         for j in range(3):
             X_ex_filt[:,j] = lfilter(b, a, X_ex[:,j])           
         return X_ex_filt
        
        
def extract_stft_features(N_samples, recording_names,  **kwargs):
    
    if N_samples == -1 or N_samples >len(recording_names):
        N_samples = len(recording_names)
        
    apply_filter = kwargs['apply_filter']        
    win_len = kwargs['win_len'] 
    overlap_l = kwargs['overlap_l']
    folder_name = kwargs['folder_name']
    idxs = np.random.choice(len(recording_names),N_samples)
    
    X_features_fft = []
    for idx in idxs:
        file_2_read = folder_name + recording_names[idx]
        
        #Fs, X_ex = wavfile.read(file_2_read)
        X_ex, Fs = librosa.load(file_2_read, sr=None, mono=True) # ovooo
        #X_ex = X_ex[:,0]        
        if apply_filter:       
           X_ex = preprocess_filter_signal(X_ex, filter_type = kwargs['f_type'], Fs=2000, filter_order = kwargs['filter_order'], cutoff = kwargs['cutoff'])
        
        xstft_temp = perform_stft(X_ex, Fs, win_len, overlap_l)
        X_features_fft.append(xstft_temp)


    return np.array(X_features_fft), N_samples


def extract_fft_features_synth(N_samples, x_name, y_name,  **kwargs):
    
    
    X_features_ml = []
    Y_features = []    
    
    x_data = np.load(x_name)
    y_data = np.load(y_name)
    #x_data = x_data[:,0,:]
    
    
    for xd in x_data: # perform_fft(x_temp_, Fs)
        X_features_ml.append(perform_fft(xd, Fs))  # x_temp_ - (4096,)

    X_features_ml = np.stack(X_features_ml) # N, 64, 65
    X_features_ml = X_features_ml[:,None,:]
    Y_features = np.ones_like(y_data)
    

    return X_features_ml, Y_features


def extract_fft_features(N_samples, recording_names, **kwargs):
    
    if N_samples == -1 or N_samples >len(recording_names):
        N_samples = len(recording_names)
     
    apply_filter = kwargs['apply_filter']        
    seg_slice_len = kwargs['segment_len']
        
    machines = kwargs['machine_ids']
    process_names = kwargs['process_names']
    n_mels = kwargs['n_mels']
    
    X_features_fft = []
    Y_features = []    
    with open("OPs_stats", 'rb') as f:
            OPs_stats = pickle.load(f) 
    
    for process_name, machine, label in itertools.product(process_names, machines, labels):
        data_path = os.path.join(path_to_dataset, machine, process_name, label)
        data_list, data_label = data_loader_utils.load_tool_research_data(data_path, label=label)
        
        for data_l_t in data_list:
          data_l_t = (data_l_t - OPs_stats[process_name + "_mu"])/(OPs_stats[process_name + "_mad"] + 1e-6)
          if apply_filter:  
             data_l_t = preprocess_filter_signal(data_l_t, filter_type = kwargs['f_type'], Fs=2000, filter_order = kwargs['filter_order'], cutoff = kwargs['cutoff'], tshs = [600, 500, 800])
            
          data_l_t = segment_array(data_l_t, seg_slice_len, step = 1000)   
            
          for data_l_t_temp in data_l_t:   
             x_feat_temp_ = [] 
             for i in range(3):
               x_temp_ = data_l_t_temp[:, i]
               x_feat_temp_.append(perform_fft(x_temp_, Fs))
             
             X_features_fft.append(np.stack(x_feat_temp_)) 
              
          if label == 'good':
             Y_features.append(np.zeros(len(data_l_t)))
          else:
             Y_features.append(np.ones(len(data_l_t))) # fault 
   
    if len(X_features_fft)>0:
        X_features_fft = np.stack(X_features_fft) # N X 3 X 2048 (4096/2)
        Y_features = np.concatenate(Y_features)
    
    return X_features_fft, Y_features
   

def extract_mellog_features_synth(N_samples, x_name, y_name,  **kwargs):
    
        
    win_len = kwargs['win_len'] 
    hop_l = kwargs['hop_l']
    n_mels = kwargs['n_mels']
    
    X_features_ml = []
    Y_features = []    
    
    x_data = np.load(x_name)
    y_data = np.load(y_name)
    x_data = x_data[:,0,:]
    print(x_data.shape)
    
    for xd in x_data:
        #xd = perform_dwt_hard_denoising(xd, threshold = 0, comp_to_keep=[0,1,2])  
        X_features_ml.append(perform_mellog(xd, Fs, win_len, hop_l, n_mels))  # x_temp_ - (4096,)

    X_features_ml = np.stack(X_features_ml) # N, 64, 65
    X_features_ml = X_features_ml[:,None,:,:]
    Y_features = np.ones_like(y_data)
    
    return X_features_ml, Y_features

    
    
def extract_mellog_features(N_samples, recording_names,  **kwargs):
    
    if N_samples == -1 or N_samples >len(recording_names):
        N_samples = len(recording_names)
        
    apply_filter = kwargs['apply_filter']        
    win_len = kwargs['win_len'] 
    hop_l = kwargs['hop_l']
    seg_slice_len = kwargs['segment_len']
        
    machines = kwargs['machine_ids']
    process_names = kwargs['process_names']
    n_mels = kwargs['n_mels']
    
    X_features_ml = []
    Y_features = []    
    
    with open("OPs_stats", 'rb') as f:
            OPs_stats = pickle.load(f) 
    
    for process_name, machine, label in itertools.product(process_names, machines, labels):
        data_path = os.path.join(path_to_dataset, machine, process_name, label)
        data_list, data_label = data_loader_utils.load_tool_research_data(data_path, label=label)
        
        for data_l_t in data_list:
          data_l_t = (1/OPs_stats[process_name + "_rms"])*(data_l_t - OPs_stats[process_name + "_mu"])/(OPs_stats[process_name + "_mad"] + 1e-6)
          data_l_t *= 80 # not computed with this gain 
          if apply_filter:  
             data_l_t = preprocess_filter_signal(data_l_t, filter_type = kwargs['f_type'], Fs=2000, filter_order = kwargs['filter_order'], cutoff = kwargs['cutoff'], tshs = [600, 500, 800])
            
          data_l_t = segment_array(data_l_t, seg_slice_len, step = 1000)   
            
          for data_l_t_temp in data_l_t:   
             x_feat_temp_ = [] 
             for i in range(3):
               x_temp_ = data_l_t_temp[:, i]
               x_feat_temp_.append(perform_mellog(x_temp_, Fs, win_len, hop_l,n_mels))
             
             X_features_ml.append(np.stack(x_feat_temp_)) 
              
          if label == 'good':
             Y_features.append(np.zeros(len(data_l_t)))
          else:
             Y_features.append(np.ones(len(data_l_t))) # fault 
   
    if len(X_features_ml)>0:
        X_features_ml = np.stack(X_features_ml)
        Y_features = np.concatenate(Y_features)
    
    return X_features_ml, Y_features
 
    
   
def extract_mel_energ_features(N_samples, recording_names,  **kwargs):
    
    if N_samples == -1 or N_samples >len(recording_names):
        N_samples = len(recording_names)

    apply_filter = kwargs['apply_filter']        
    win_len = kwargs['win_len'] 
    hop_l = kwargs['hop_l']
    seg_slice_len = kwargs['segment_len']
        
    machines = kwargs['machine_ids']
    process_names = kwargs['process_names']
    n_mels = kwargs['n_mels']
    
    X_features_ml = []
    Y_features = []    
    
    with open("OPs_stats", 'rb') as f:
            OPs_stats = pickle.load(f) 
            
    for process_name, machine, label in itertools.product(process_names, machines, labels):
        data_path = os.path.join(path_to_dataset, machine, process_name, label)
        data_list, data_label = data_loader_utils.load_tool_research_data(data_path, label=label)
        
        for data_l_t in data_list:
          if apply_filter:  
             data_l_t = preprocess_filter_signal(data_l_t, filter_type = kwargs['f_type'], Fs=2000, filter_order = kwargs['filter_order'], cutoff = kwargs['cutoff'], tshs = [600, 500, 800])
          #data_l_t = (data_l_t - OPs_stats[process_name + "_mu"])/(OPs_stats[process_name + "_mad"] + 1e-6)        
          data_l_t = normalize_clip(data_l_t, OPs_stats[process_name + "_center"], OPs_stats[process_name + "_scale"], clip=150.0)
          data_l_t *= 3.4
          data_l_t = np.clip(data_l_t, -5.0, 5.0) 
          data_l_t = segment_array(data_l_t, seg_slice_len, step = 1000)   
            
          for data_l_t_temp in data_l_t:   
             x_feat_temp_ = [] 
             for i in range(3):
               x_temp_ = data_l_t_temp[:, i]
               x_feat_temp_.append(perform_mel_ener(x_temp_, Fs, **kwargs))
             
             X_features_ml.append(np.stack(x_feat_temp_)) 
              
          if label == 'good':
             Y_features.append(np.zeros(len(data_l_t)))
          else:
             Y_features.append(np.ones(len(data_l_t))) # fault 
   
    if len(X_features_ml)>0:
        X_features_ml = np.stack(X_features_ml) # N x 3 x 64 x 128
        Y_features = np.concatenate(Y_features)
    
    return X_features_ml, Y_features


def extract_time_features(N_samples, recording_names,  **kwargs):
    
    if N_samples == -1 or N_samples >len(recording_names):
        N_samples = len(recording_names)
     
    apply_filter = kwargs['apply_filter']        
    seg_slice_len = kwargs['segment_len']
        
    machines = kwargs['machine_ids']
    process_names = kwargs['process_names']
    
    X_features_time = []
    Y_features = []    
    
    
    for process_name, machine, label in itertools.product(process_names, machines, labels):
        data_path = os.path.join(path_to_dataset, machine, process_name, label)
        data_list, data_label = data_loader_utils.load_tool_research_data(data_path, label=label)
        
        for data_l_t in data_list:
          if apply_filter:  
             data_l_t = preprocess_filter_signal(data_l_t, filter_type = kwargs['f_type'], Fs=2000, filter_order = kwargs['filter_order'], cutoff = kwargs['cutoff'], tshs = [600, 500, 800])
            
          data_l_t = segment_array(data_l_t, seg_slice_len, step = 1000)   
          X_features_time.append(np.swapaxes(data_l_t, 1, 2))                          
              
          if label == 'good':
             Y_features.append(np.zeros(len(data_l_t)))
          else:
             Y_features.append(np.ones(len(data_l_t))) # fault 
   
    if len(X_features_time) > 0:
        X_features_time = np.concatenate(X_features_time) 
        Y_features = np.concatenate(Y_features)
     
        # compute min and max per axis (channel)
        mins = X_features_time.min(axis=(0, 2), keepdims=True)   # shape (1,3,1)
        maxs = X_features_time.max(axis=(0, 2), keepdims=True)   # shape (1,3,1)

        X_features_time = 2 * ( (X_features_time - mins) / (maxs - mins) ) - 1
    
    return X_features_time, Y_features


def ExtractSelectedFeatures(N_samples, recording_names, selected_feature, **kwargs):
    
    if selected_feature == "FFT":
       X_features_extracted, Y = extract_fft_features(N_samples, recording_names, **kwargs)
    elif selected_feature == "STFT":
       X_features_extracted, Y = extract_stft_features(N_samples, recording_names, **kwargs)
    elif selected_feature == "MelLog":
          X_features_extracted, Y = extract_mellog_features(N_samples, recording_names, **kwargs)
    elif selected_feature == "MEL_ENERGY":
       X_features_extracted, Y = extract_mel_energ_features(N_samples, recording_names, **kwargs)          
    elif selected_feature == "TIME":
       X_features_extracted, Y = extract_time_features(N_samples, recording_names, **kwargs)    
    else:
        print(f" {selected_feature} - nNot a regualr feature type !!!")
        return -1, 0
        
    return X_features_extracted, Y 

def ExtractSelectedFeatures_Synthetic(N_samples, x_name, y_name, selected_feature, **kwargs):
    

    if selected_feature == "MelLog":
          X_features_extracted, Y = extract_mellog_features_synth(N_samples, x_name, y_name, **kwargs)
    elif selected_feature == "FFT":
          X_features_extracted, Y = extract_fft_features_synth(N_samples, x_name, y_name, **kwargs)
    else:
        print(f" {selected_feature} - nNot a regualr feature type !!!")
        return -1, 0
        
    return X_features_extracted, Y 


def get_mel_en_test_idx(idx_anom, mpltkl):
    
    idxn = np.zeros((len(idx_anom), mpltkl), dtype=np.int32)
    cntr = 0
    
    for idxt in idx_anom:
        idxn[cntr,:] = np.linspace(idxt*mpltkl, (idxt+1)*mpltkl-1, mpltkl, dtype=np.int32)
        cntr +=1
    
    return idxn.flatten()
   
# %%
    

   
    
    
   
    
   
    
   