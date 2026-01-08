# Attention-Based Radiomics for-predicting gliomas histological grade

This repository presents an innovative deep learning framework for predicting the histological grade of gliomas. The proposed pipeline begins with the pre-processing of multimodal MRI sequences (T1, contrast-enhanced T1 (CT1), T2, and FLAIR), followed by fully automatic tumor segmentation using HD-GLIO-AUTO. Based on the segmented tumor subregions, radiomics features are extracted and subsequently used as input to a neural network architecture that leverages an attention mechanism to effectively fuse multimodal information for glioma grading.

An overview of the pipeline is reported below:
![Pipeline overview](pipeline.png)

## Pre-Processing and extraction of radiomic features
All images undergo a standardized pre-processing pipeline to ensure consistent orientation, spatial alignment, and removal of non-brain structures. Volumes are first reoriented to the FSL RAS convention. Brain extraction is then performed using HD-BET to remove extra-cranial tissues. For each subject, multimodal images are co-registered by selecting the sequence with the highest spatial resolution as the reference. After registration, brain masking is reapplied to eliminate interpolation artifacts. The resulting images are fully aligned, anonymized, and ready for downstream segmentation. 

MRI scans are automatically segmented using HD-GLIO-AUTO (nnU-Net–based), processing T1, contrast-enhanced T1, T2, and FLAIR sequences. The pipeline outputs two tumor masks: contrast-enhancing regions (active tumor core) and non-enhancing regions (infiltrative tumor and edema), providing accurate and reproducible ROIs for downstream analysis.
