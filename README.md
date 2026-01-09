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
- FSL 
- HD-BET
- HD-GLO-AUTO


## Usage
