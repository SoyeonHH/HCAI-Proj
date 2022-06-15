# HCAI-Proj

## Introduction
This is the Human Centered AI Project in Ajou University. The title is `Non-verbal Features Interpretation for Multimodal Sentiment Analysis`. In this project, there are two implementation tasks. The first one is implementation of generalized additive model (GAM) that use regression splines to plot non-linear relationship between linguistic feature and non-verbal features. This can be automatically implemented using `pyGAM` package . The other thing is the interpretation that provide utterance-level feature importance representation and visualization the attention for the predictions. It was implemented using `Captum` , which provide generic implementation of integrated gradients that can be used with any PyTorch model.

## Usage
1. Download the CMU-MOSI and CMU-MOSEI dataset from [Google Drive](https://drive.google.com/drive/folders/1djN_EkrwoRLUt7Vq_QfNZgCl_24wBiIK?usp=sharing) or [Baidu Disk](https://pan.baidu.com/s/1Wxo4Bim9JhNmg8265p3ttQ) (extraction code: g3m2). It can be downloaded in https://github.com/declare-lab/Multimodal-Infomax.

2. Set up the environment (need conda prerequisite)
```
conda env create -f environment.yml
conda activate pytorch
```

3. Start training
```
python main.py --dataset mosi --contrast
```