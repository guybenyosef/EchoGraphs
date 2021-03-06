# Light-weight spatio-temporal graphs for segmentation and ejection fraction prediction in cardiac ultrasound (MICCAI 2022).

<p align="center">
  <br>
    <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
    <a href='url_to_youtube_video' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/Youtube-Video-red?style=flat&logo=youtube&logoColor=red' alt='Youtube Video'>
    </a>
    <a href='url_for_pdf'>
      <img src='https://img.shields.io/badge/Paper-PDF-green?style=flat&logo=arXiv&logoColor=green' alt='Paper PDF'>
    </a>
    <a href='https://conferences.miccai.org/2022/en/'>
      <img src='https://www.google.com/imgres?imgurl=https%3A%2F%2Fconferences.miccai.org%2F2022%2Ffiles%2Fimages%2Flayout%2Fgeneral%2Fmiccai2022-logo.png&imgrefurl=https%3A%2F%2Fconferences.miccai.org%2F2022%2F&tbnid=OPWAc8X5KGdUGM&vet=12ahUKEwi1md24_cL4AhUTohoKHfSPCeYQMygAegUIARCmAQ..i&docid=HBGl5SZg0_tNbM&w=576&h=184&q=miccaI%202022%20logo&safe=active&ved=2ahUKEwi1md24_cL4AhUTohoKHfSPCeYQMygAegUIARCmAQ' alt='MICCAI22'>
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
This is the source code for MICCAI 2022 paper: [Light-weight spatio-temporal graphs for segmentation and ejection fraction prediction in cardiac ultrasound](link_to_arxiv)

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

## Citation
If you feel helpful of this work, please cite it.

```
@inproceedings{echographs,
  title={Light-weight spatio-temporal graphs for segmentation and ejection fraction prediction in cardiac ultrasound},
  author={Thomas, Sarina and Gilbert, Andrew and Ben-Yosef, Guy},
  booktitle={MICCAI},
  year={2022}
}
