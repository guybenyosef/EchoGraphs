# Light-weight spatio-temporal graphs for segmentation and ejection fraction prediction in cardiac ultrasound (MICCAI 2022).

<p align="center">
  <br>
    <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
    <a href='url_to_youtube_video' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/Youtube-Video-red?style=flat&logo=youtube&logoColor=red' alt='Youtube Video'>
    </a>
    <a href=https://arxiv.org/abs/2207.02549>
      <img src='https://img.shields.io/badge/Paper-PDF-green?style=flat&logo=arXiv&logoColor=green' alt='Paper PDF'>
    </a>
    <a href='https://conferences.miccai.org/2022/en/'>
      <img src='figures/miccai2022-logo.png' height=15 alt='MICCAI22'>
    </a>
</p>


## Introduction
Accurate and consistent predictions of echocardiography parameters are important for cardiovascular diagnosis and treatment. 
In particular, segmentations of the left ventricle can be used to derive ventricular volume, ejection fraction (EF) and other relevant measurements. 
In this paper we propose a new automated method called EchoGraphs for predicting ejection fraction and segmenting the left ventricle by detecting anatomical keypoints. Models for direct coordinate regression based on Graph Convolutional Networks (GCNs) are used to detect the keypoints. GCNs can learn to represent the cardiac shape based on local appearance of each keypoint, as well as global spatial and temporal structures of all keypoints combined. 
We evaluate our EchoGraph model on the EchoNet benchmark dataset. 
Compared to semantic segmentation, GCNs show accurate segmentation and improvements in robustness and inference run-time. 
EF is computed simultaneously to segmentations and our method also obtains state-of-the-art ejection fraction estimation.

## Paper
This is the source code for MICCAI 2022 paper: [Light-weight spatio-temporal graphs for segmentation and ejection fraction prediction in cardiac ultrasound](https://arxiv.org/abs/2207.02549)

## The EchoGraphs model architecture
EchoGraphs provides a framework for graph-based contour detection for medical ultrasound. 
The repository includes model configurations for
1) predicting the contour of the left ventricle in single ultrasound images (single-frame GCN)
2) predicting two contours of the ED and ES frame and the corresponding EF value for ultrasound sequences with known ED/ES frame (multi-frame GCN)
3) predicting the EF values of arbitrary ultrasound sequences alongside with the occurence of ED and ES and the corresponding frames (multi-frame GCN, extension indicated in grey in the figure)

![plot](./figures/NetworkOverview.png)

## Dataset
The proposed methods were trained and evaluated using the EchoNet dataset which consists of 10.030 echocardiac ultrasound sequences that were de-identified and made publicly available. The usage of those datasets requires registration and is shared with a non-commerical data use agreement.
Additional information on the data and on regulations can be found in [echonet dataset](https://echonet.github.io/dynamic/). If you use the data, please cite the corresponding paper. We provide a preprocessing script to convert the data into formats that are readible by our pipeline (details can be found [here](./GETTING_STARTED.md)).

## Environment
See [INSTALL.md/](./INSTALL.md) for environment setup.

## Getting stated
See [GETTING_STARTED.md](./GETTING_STARTED.md) to get started with training and testing the echographs model. 

## Results 
[<img src="./figures/GCN_MobileNet2_single_frame.gif" width="800"/>](./figures/GCN_MobileNet2_single_frame.gif)

These examples show video sequences with model prediction. Although the single frame Echograph was only trained on the keyframes ED (end diastole) and ES (end systole) it produces accurate and consistent predictions across all other frames.

## Acknowledgements

- The spiral convolution implementation is adapted from the repo Neural3DMM

Bouritsas, G. et al.: Neural 3D Morphable Models: Spiral Convolutional Networks for 3D 
Shape Representation Learning and Generation, ICCV, 2019 
https://github.com/gbouritsas/Neural3DMM

- The regression multi-layer perceptron is inspired by the repo UVT

Reynaud et al.: Ultrasound Video Transformers (UVT) for Cardiac Ejection Fraction Estimation, MICCAI, 2021
https://github.com/HReynaud/UVT

- For evaluation and preprocessing of the networks methods from the echonet-dynamic repo were used.

Ouyang et al.: Video-based AI for beat-to-beat assessment of cardiac function, Nature, 2020 
https://echonet.github.io/dynamic/index.html#code


## Citation
Please consider citing our work if you find it useful:

```
@inproceedings{echographs,
  title={Light-weight spatio-temporal graphs for segmentation and ejection fraction prediction in cardiac ultrasound},
  author={Thomas, Sarina and Gilbert, Andrew and Ben-Yosef, Guy},
  booktitle={MICCAI},
  year={2022}
}
