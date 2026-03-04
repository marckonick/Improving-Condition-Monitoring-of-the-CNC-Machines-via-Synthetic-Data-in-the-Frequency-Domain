# Improving-Condition-Monitoring-of-the-CNC-Machines-via-Synthetic-Data-in-the-Frequency-Domain
The official repository for the paper "Improving Condition Monitoring of the CNC-Machines-via Synthetic Data in the Frequency Domain" - Splitech 2026 

## Description 
This paper 

## Dataset info 

The dataset can be downloaded from the following repository - https://github.com/boschresearch/CNC_Machining.

## Files Description

Codes for Condition Monitoring

- Functions_FeatureExtraction.py            - Functions for performing feature extraction spectrograms. Check the path to the data folder -  CNC_Machining-main which should be dowloaded first (see the previous section).  
- Main_FeatureExtraction_CNC.py             - Can be configured to perform various extraction of various features (FFT, Mel Spectrograms, Mel Energy, STFT). Paper uses only Mel Spectrogram.
- TorchClassificationModels.py              - Torch implementaion of NNs for state classification.
- Main_Classification_Model.py              - Script for training the condiiton monitoring model. For the paper the script used the following arguments --focal_loss
- Test_Classification_Model.py              - Load and test models. Should be run with arguments --focal_loss


Codes for Vibration Generation
- Main_Segmentation.py                      - Prepare data for difussion model training. Load -> normalize -> denoise. Also saves per operation statistics for normalization
