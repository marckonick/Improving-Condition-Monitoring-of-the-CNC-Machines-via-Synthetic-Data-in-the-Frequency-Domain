import numpy as np
import matplotlib.pyplot as plt 
import librosa 

def plot_3axis_mel_feature(feature, Fs=2000, hop_l=64, n_mels=64):
    """
    feature: numpy array of shape (3, n_mels, time_frames)
    Fs: sampling frequency
    hop_l: hop length used in mel extraction
    n_mels: number of mel bands
    """
    
    assert feature.shape[0] == 3, "Feature must have 3 axes"
    
    n_frames = feature.shape[2]
    
    # Time axis (seconds)
    time_axis = np.arange(n_frames) * hop_l / Fs
    
    # Mel frequency axis converted to Hz
    mel_frequencies = librosa.mel_frequencies(n_mels=n_mels, fmin=0, fmax=Fs/2)
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    
    axis_labels = ['Axis X', 'Axis Y', 'Axis Z']
    
    for i in range(3):
        im = axes[i].imshow(
            feature[i],
            aspect='auto',
            origin='lower',
            extent=[time_axis[0], time_axis[-1],
                    mel_frequencies[0], mel_frequencies[-1]]
        )
        
        axes[i].set_ylabel("Frequency [Hz]")
        axes[i].set_title(axis_labels[i])
        fig.colorbar(im, ax=axes[i], format="%+2.0f dB")
    
    axes[-1].set_xlabel("Time [s]")
    plt.tight_layout()
    plt.show()
    
def plot_realfake_per_op(x_r, y_r, x_s, y_s, OP_ix):
    
    Fs = 2e3
    n_frames = x_r.shape[3]
    hop_l = 64
    
    font_sz_title = 28
    font_sz_ax_name = 24
    font_sz_label = 24
    font_sz_ticks = 22
    font_sz_cbar = 20
    
    signals_r = x_r[y_r==OP_ix]
    signals_s = x_s[y_s==OP_ix]
    
    rnd_img_num_r = np.random.choice(len(signals_r))
    rnd_img_num_s = np.random.choice(len(signals_s))
    
    S_r = signals_r[rnd_img_num_r,:]
    S_s = signals_s[rnd_img_num_s,:]
    
    time_axis = np.arange(n_frames) * hop_l / Fs
    mel_frequencies = librosa.mel_frequencies(n_mels=64, fmin=0, fmax=Fs/2)
        
    fig, axes = plt.subplots(3, 2, figsize=(30, 30), sharex=True)

    axis_names = ['X-axis', 'Y-axis', 'Z-axis']

    for i in range(3):
      im = axes[i,0].imshow(
        S_r[i],
        aspect='auto',
        origin='lower',
        cmap='magma',
        extent=[time_axis[0], time_axis[-1],
                mel_frequencies[0], mel_frequencies[-1]]
      )
      axes[i,0].set_title(axis_names[i], fontsize = font_sz_ax_name)
      axes[i,0].set_ylabel('Frequency [Hz]', fontsize = font_sz_label)
      axes[i,0].tick_params(axis='both', labelsize = font_sz_ticks)
      #fig.colorbar(im, ax=axes[i,0], fraction=0.046, pad=0.04)  # Changed here

      im = axes[i,1].imshow(
        S_s[i],
        aspect='auto',
        origin='lower',
        cmap='magma',
        extent=[time_axis[0], time_axis[-1],
                mel_frequencies[0], mel_frequencies[-1]]
      )
      axes[i,1].set_title(axis_names[i], fontsize = font_sz_ax_name)
      axes[i,1].set_ylabel('Frequency [Hz]', fontsize = font_sz_label)
      axes[i,1].tick_params(axis='both', labelsize = font_sz_ticks)
      cbar = fig.colorbar(im, ax=axes[i,1], fraction=0.046, pad=0.04, format="%+2.0f dB")  # Changed here
      cbar.ax.tick_params(labelsize= font_sz_cbar) 
      
    fig.suptitle(f"Real eatures [left] and synthetic features [right] \n tool-operation {OP_ix}", fontsize=font_sz_title)
    axes[-1,0].set_xlabel('Time [s]', fontsize = font_sz_label)
    axes[-1,1].set_xlabel('Time [s]', fontsize = font_sz_label)
   
    for axrow in axes:
     for ax in axrow:
        ax.tick_params(axis='x', pad=10)
        ax.tick_params(axis='y', pad=10)
    plt.tight_layout()
    rndint = np.random.randint(9626)
    plt.savefig(f"saved_generated_images/OP_{OP_ix}_img_{rndint}.eps")
    plt.savefig(f"saved_generated_images/OP_{OP_ix}_img_{rndint}.png")
    plt.show()
    plt.close()
    
    print(f"random image number real {rnd_img_num_r}") # 549
    print(f"random image number synth {rnd_img_num_s}") # 12

    print(f"random save number {rndint}")
    
x_train_melener = np.load("X_melener_segment_filt_train.npy")
x_test_melener = np.load("X_melener_segment_filt_test.npy")
y_train = np.load("Y_data_segment_filt_train.npy")
y_test = np.load("Y_data_segment_filt_test.npy")


x_real = np.concatenate((x_train_melener, x_test_melener), 0)
y_real = np.concatenate((y_train, y_test))

x_synth = np.load("saved_generated_data/x_samples_diffusion_melener_paper.npy")
y_synth = np.load("saved_generated_data/y_samples_diffusion_melener_paper.npy")


# %%
ix_op = 11

for _ in range(10):

    plot_realfake_per_op(x_real, y_real, x_synth, y_synth, ix_op)



    



