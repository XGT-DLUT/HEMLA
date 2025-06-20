# HEMLA

## Introduction
This repository contains the implementation for the paper **"Hard Example Mining-Driven Label Alignment for LLMs in Aspect Sentiment Triplet Extraction."**  
The proposed approach leverages a T5 model as an intermediate aligner to better align LLM responses with ground-truth labels, enhancing performance in aspect sentiment triplet extraction tasks.

## Environment Setup

Recommended versions:
- Python 3.10  
- PyTorch 2.3.0  
- CUDA 11.8  

To create and activate a virtual environment, run the following commands:

```bash
conda create -n hemla python=3.10
conda activate hemla
pip install -r requirements.txt
```

## Model Training

To start training the model, run:

```bash
python train.py
```