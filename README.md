# Improving-Condition-Monitoring-of-the-CNC-Machines-via-Synthetic-Data-in-the-Frequency-Domain
The official repository for the paper "Improving Condition Monitoring of the CNC-Machines-via Synthetic Data in the Frequency Domain" - Splitech 2026 

## Description 
Condition monitoring (CM) of CNC machines is challenging due to the scarce and severely imbalanced fault data, as failure events are usually rare. This paper proposes a frequency-domain synthetic data augmentation pipeline to improve fault detection across machining operations. Using the Bosch CNC Machining Dataset, tri-axial vibration signals from multiple machines are denoised via wavelet packet decomposition and transformed into compact log Mel-filterbank energy features. A lightweight convolutional neural network is trained for state classification under imbalance using balanced batch sampling and focal loss. To enrich the minority faulty class, conditional denoising diffusion probabilistic model is developed on Mel-energy representations. The model employs a U-Net architecture with classifier-free guidance for operation conditioned generation and incorporates a novel composite loss function that combines v-parameterization, clean sample reconstruction, temporal total variation, and macro-envelope matching to ensure both local time-frequency fidelity and global modulation structure preservation. Synthetic quality is assessed in the classifier embedding space using maximum mean discrepancy and kNN precision and recall, and downstream CM impact is measured via per-operation confusion matrices. Results show that adding generated samples generally increases faulty-state recognition while preserving high normal-state accuracy, with strong distributional alignment.

## Dataset info 

The dataset can be downloaded from the following repository - https://github.com/boschresearch/CNC_Machining.

## Files Description

Codes for Condition Monitoring

- Functions_FeatureExtraction.py            - Functions for performing feature extraction spectrograms. Check the path to the data folder -  CNC_Machining-main which should be dowloaded first (see the previous section).  
- Main_FeatureExtraction_CNC.py             - Can be configured to perform various extraction of various features (FFT, Mel Spectrograms, Mel Energy, STFT). Paper uses only MEL_ENERGY.
- TorchClassificationModels.py              - Torch implementaion of NNs for state classification.
- Main_ClassificationModel.py              - Script for training the condiiton monitoring model. For the paper the script used the following arguments --focal_loss
- Test_Classification_Model.py              - Load and test models. Should be run with arguments --focal_loss


Codes for Vibration Generation
- Main_Segmentation.py                      - Prepare data for difussion model training. Load -> normalize -> denoise. Also saves per operation statistics for normalization.
- DiffusionModel_UNet.py                    - Definition of the diffusion model.
- Main_Diffusion_MelEner.py                 - Training script for the diffusion model with additional losses.
- SampleDataCFG.py                          - Sample augmented data for quality esitmaiton and classifier training.
- Functions_CheckQuality_FeatureSpace.py    - Function used for computing MMD and KNN quality metrics.
- Main_ComputeQualityMetrics.py             - Script for computing MMD and KNN quality metrics in the learned feature space. 
