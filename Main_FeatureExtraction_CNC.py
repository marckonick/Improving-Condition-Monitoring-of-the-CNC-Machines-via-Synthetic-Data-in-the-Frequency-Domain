import numpy as np
import argparse
import Functions_FeatureExtraction as FFE
import itertools


parser = argparse.ArgumentParser()

parser.add_argument("--machine_ids",
                    type=str,
                    nargs='+', 
                    default=["M01"]) # M02, M03

parser.add_argument("--process_names",
                    type=str,
                    nargs='+',  
                    default=["OP01"])

parser.add_argument("--selected_feature",
                    type = str, default = "MEL_ENERGY") # FFT, STFT, MEL_ENERGY, MelLog, TIME
parser.add_argument("--N_samples_2_extract",
                    type = int, default = -1)  #-1 - means all samples  
parser.add_argument("--segment_len",
                    type = int, default = 4096)  
parser.add_argument("--apply_filter",
                    type = bool, default = True) # 
parser.add_argument("--f_type",
                    type = str, default = "DWT") # DWT, median, lowpass
parser.add_argument("--filter_order",
                    type = int, default = 11)
parser.add_argument("--cutoff",
                    type = int, default = 600)
parser.add_argument("--win_len",
                    type = int, default = 128)
parser.add_argument("--overlap_l",
                    type = int, default = 128)
parser.add_argument("--hop_l",
                    type = int, default = 64)
parser.add_argument("--n_mels",
                    type = int, default = 64)
parser.add_argument("--frames_mel",
                    type = int, default = 2)
parser.add_argument("--power_mel",
                    type = float, default = 2.0)
parser.add_argument("--save_2_npy",
                    type = bool, default = True)
parser.add_argument("--file_folder", 
                    type = str, default = "../")
parser.add_argument("--save_folder",
                    type = str, default = "../saved_features/")


# %%
args = parser.parse_args()
 
selected_feature = args.selected_feature # MEL_ENERGY
N_samples_2_extract = args.N_samples_2_extract

filter_kwarg = {'apply_filter':args.apply_filter, 'f_type':args.f_type, 'filter_order':args.filter_order,'cutoff':args.cutoff} 
kwarg_args = {'win_len':args.win_len, 'overlap_l':args.overlap_l, 'hop_l':args.hop_l, "n_mels":args.n_mels, 
                 'frames_mel':args.frames_mel, 'power_mel':args.power_mel, 'segment_len': args.segment_len}


kwarg_args.update(filter_kwarg)
    
process_names = ["OP01","OP02","OP03","OP04","OP05", "OP07","OP08","OP10","OP11","OP12","OP14"]
machines = ["M01", "M02", "M03"] 

for mn, pn in itertools.product(machines, process_names):
    
    machine_kwargs = {'machine_ids': [mn],'process_names':[pn]}
    kwarg_args.update(machine_kwargs)

    X_features_extracted, Y = FFE.ExtractSelectedFeatures(-1, "", selected_feature, **kwarg_args)    

    save_name_X = 'saved_features/' + args.selected_feature + '/' + 'X_' + mn + '_' + pn + '.npy'
    save_name_Y = 'saved_features/' + args.selected_feature + '/' + 'Y_' + mn + '_' + pn + '.npy'


    if args.save_2_npy and len(X_features_extracted)>0:
        np.save(save_name_X, X_features_extracted)
        np.save(save_name_Y, Y)



