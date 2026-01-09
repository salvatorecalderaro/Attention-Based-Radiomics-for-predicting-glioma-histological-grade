# Attention-Based Glioma Histological Grading

This repository contains the implementation of an attention-based deep learning framework for automatic glioma histological grade classification using multimodal MRI radiomic features, in accordance with the World Health Organization (WHO) criteria.

## Overview
The proposed pipeline integrates deep learning and radiomics to provide an accurate and interpretable assessment of glioma histological grade. The method is designed to support clinical decision-making, prognosis estimation, and early risk stratification.

## Pipeline
1. MRI preprocessing (normalization, registration)
2. Automatic tumor segmentation using state-of-the-art deep learning models
3. Radiomic feature extraction (intensity, texture, shape)
4. Attention-based classification network
5. Model evaluation and external validation

![Pipeline overview](pipeline.png)

## Datasets
- **UCSF-PDGM Dataset**  
  Publicly available dataset for glioma imaging and molecular analysis  
  👉 https://www.cancerimagingarchive.net/collection/ucsf-pdgm/

- **External Clinical Cohort**  
  Independent dataset used for external validation (not publicly available)

## Radiomic Features
Radiomic features are extracted following standardized definitions and include:
- First-order intensity statistics
- Texture features (GLCM, GLRLM, GLSZM)
- Shape and morphological descriptors

## Model
The classification framework employs:
- Attention mechanisms to weight informative radiomic features
- Fully connected layers for grade prediction
- Cross-entropy loss for multiclass classification

## Results
Experimental results on the UCSF-PDGM dataset and an independent clinical cohort demonstrate performance comparable to existing baselines, while offering enhanced interpretability through radiomic feature analysis.

## Clinical Impact
By combining deep learning with interpretable radiomic features, the proposed approach supports:
- Accurate glioma grading
- Prognosis estimation
- Early risk stratification and preventive clinical decision-making

## Requirements
- Python >= 3.8
- PyTorch
- NumPy
- Scikit-learn
- PyRadiomics 
- Nibabel
- FSL 👉 https://fsl.fmrib.ox.ac.uk/fsl/docs/
- HD-BET 👉 https://github.com/MIC-DKFZ/HD-BET 
- HD-GLO-AUTO 👉 https://github.com/CCI-Bonn/HD-GLIO

Ensure all tools are correctly installed and available in your system `PATH`.


This script performs preprocessing, automatic tumor segmentation, and radiomic feature extraction from multimodal MRI data.

### Input Requirements
The input directory must contain the following NIfTI files:

- `T1.nii.gz`
- `CT1.nii.gz` (contrast-enhanced T1)
- `T2.nii.gz`
- `FLAIR.nii.gz`

All images must be in the same directory and correspond to the same patient.

### Running the Pipeline
```bash
python run_pipeline.py \
  -i /path/to/input_directory \
  -o /path/to/output_directory \
  --device 0 \
  --verbose
```
