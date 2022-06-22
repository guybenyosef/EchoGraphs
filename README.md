# Light-weight spatio-temporal graphs for segmentation and ejection fraction prediction in cardiac ultrasound (MICCAI 2022).


## Introduction
Accurate and consistent predictions of echocardiography parameters are important for cardiovascular diagnosis and treatment. 
In particular, segmentations of the left ventricle can be used to derive ventricular volume, ejection fraction (EF) and other relevant measurements. 
In this paper we propose a new automated method called EchoGraphs for predicting ejection fraction and segmenting the left ventricle by detecting anatomical keypoints. Models for direct coordinate regression based on Graph Convolutional Networks (GCNs) are used to detect the keypoints. GCNs can learn to represent the cardiac shape based on local appearance of each keypoint, as well as global spatial and temporal structures of all keypoints combined. 
We evaluate our EchoGraph model on the EchoNet benchmark dataset. 
Compared to semantic segmentation, GCNs show accurate segmentation and improvements in robustness and inference run-time. 
EF is computed simultaneously to segmentations and our method also obtains state-of-the-art ejection fraction estimation

## Paper
This is the source code for MICCAI 2022 paper: [Light-weight spatio-temporal graphs for segmentation and ejection fraction prediction in cardiac ultrasound](link_to_arxiv)

## The EchoGraphs model architecture
 ![plot](./figures/NetworkOverview.png)

## Dataset
We describe here briefly our preprocess for the [echonet dataset](https://echonet.github.io/dynamic/) and the link to download preprocessed files.

## Environment
See [INSTALL.md/](./INSTALL.md) for environment setup.

## Getting stated
See [GETTING_STARTED.md](./GETTING_STARTED.md) to get started with training and testing the echographs model. 


## Citation
If you feel helpful of this work, please cite it.

```
@inproceedings{echographs,
  title={Light-weight spatio-temporal graphs for segmentation and ejection fraction prediction in cardiac ultrasound},
  author={Thomas, Sarina and Gilbert, Andrew and Ben-Yosef, Guy},
  booktitle={MICCAI},
  year={2022}
}