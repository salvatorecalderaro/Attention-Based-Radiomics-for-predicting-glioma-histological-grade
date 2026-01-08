# Attention-Based Radiomics for-predicting gliomas histological grade

This repository presents an innovative deep learning framework for predicting the histological grade of gliomas. The proposed pipeline begins with the pre-processing of multimodal MRI sequences (T1, contrast-enhanced T1 (CT1), T2, and FLAIR), followed by fully automatic tumor segmentation using HD-GLIO-AUTO. Based on the segmented tumor subregions, radiomics features are extracted and subsequently used as input to a neural network architecture that leverages an attention mechanism to effectively fuse multimodal information for glioma grading.

An overview of the pipeline is reported below:
![Pipeline overview](pipeline.png)
