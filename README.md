# HEMLA 🚀

## Introduction 🧩

This repository contains the implementation for the paper **"Hard Example Mining-Driven Label Alignment for LLMs in Aspect Sentiment Triplet Extraction."**  The proposed approach leverages a T5 model as an intermediate aligner to better align LLM responses with ground-truth labels, enhancing performance in aspect sentiment triplet extraction tasks.

## Environment Setup ⚙️

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
## Datasets 📁
The four public datasets used in the paper are located in the `mydataset/ASTE_G` directory.
All responses generated by LLMs are named after the corresponding model. For example, `16res/gpt4o_train.jsonl` represents the results generated by GPT-4o on the 16res training set.

## Model Training ✅

To start training the model, run:

```bash
python train.py
```
## Citation 📚

If you find this work useful, please cite:

```bibtex
@misc{hemla2025,
  author       = {Guangtao Xu},
  title        = {Hard Example Mining-Driven Label Alignment for LLMs in Aspect Sentiment Triplet Extraction},
  year         = {2025},
  howpublished = {https://github.com/XGT-DLUT/HEMLA},
  note         = {GitHub repository}
}
