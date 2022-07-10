---
layout: default
---

# Abstract

#### Motivation
Accurate and consistent predictions of echocardiography parameters are important for cardiovascular diagnosis and 
treatment. In particular, segmentations of the left ventricle can be used to derive ventricular volume, ejection 
fraction (EF) and other relevant measurements.

#### Summary 
 In this paper we propose a new automated method called EchoGraphs for predicting ejection fraction and segmenting the 
 left ventricle by detecting anatomical keypoints. Models for direct coordinate regression based on Graph Convolutional 
 Networks (GCNs) are used to detect the keypoints. GCNs can learn to represent the cardiac shape based on local 
 appearance of each keypoint, as well as global spatial and temporal structures of all keypoints combined. 

####  Results

We evaluate our EchoGraphs model on the EchoNet benchmark dataset. Compared to semantic segmentation, GCNs show accurate
segmentation and improvements in robustness and inference run-time. EF is computed simultaneously to segmentations and 
our method also obtains state-of-the-art ejection fraction estimation.

## Article

The paper is accepted for publication in the 2022 International Conference on Medical Image Computing and Computer 
Assisted Intervention. A pre-print is available [on arXiv here](https://arxiv.org/abs/2207.02549). 

If you find the work interesting, please cite it:

Citation: 

S. Thomas, A. Gilbert, and G. Ben-Yosef, “Light-weight spatio-temporal graphs for segmentation and ejection fraction 
prediction in cardiac ultrasound”, International Conference on Medical Image Computing and Computer-Assisted 
Intervention. Springer, Cham, 2022.




[//]: # (## Supplementary Materials)
[//]: # (Appendices referenced in the article are [available here]&#40;TBD&#41;)

# Data

The paper relied on the EchoNet dataset which is [available from Stanford here](https://echonet.github.io/dynamic/index.html). 
We would like to thank the authors [(Ouyang et. al, 2020)](https://www.nature.com/articles/s41586-020-2145-8) for making these 
resources available. 



# Code

The code is available [on GitHub](https://github.com/guybenyosef/EchoGraphs)


## Authors

Authors: Sarina Thomas<sup>1</sup>, Andrew Gilbert<sup>2</sup>, and Guy Ben-Yosef<sup>3,*</sup>



1: University of Oslo, Oslo, NO

2: GE Vingmed Ultrasound, Oslo, NO

3: GE Research, Niskayuna, New York, USA

*: Corresponding author: guy.ben-yosef@ge.com
