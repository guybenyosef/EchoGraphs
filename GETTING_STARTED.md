In this file necessary steps are described to run a training for EchoGraphs on the EchoNet data and evaluate the results

## Set up enviroment

1) Install all required packages
2) Modify CONST.py to set all global path variables to your pesonal code and data location.

## Preprocess the EchoNet dataset

``` python tools\preprocess_echonet.py ```

If you want to train the EchoGraph model with the EchoNet data, you need to register on the [echonet dataset webpage](https://echonet.github.io/dynamic/) first, agree to the data sharing agreement and download the data. Then you can run the preprocess script providing the data folder to preprocess the original echonet data into a format that is readable by the training code provided in this repository. This script will use the images and the .csv sheet to create training images and labels as well as txt files that store all filenames for training, validation and test according to the splits provided by the EchoNet authors.
40 keypoints are extracted from the .csv file indicated as X1 X2 Y1 Y2 excluding the first two points since they are the apex and basal point and not necessarily part of the contour. Invalid annotations (annotations with multiple structures that were not intended for the LV contouring task) are excluded. Further, the EF, ED volume, ES volume, ED frame index and ES frame index are extracted and stored in a npz file. Some preprocessing functions like i.e. the video reader function are taken from the EchoNet repository. The script assumes that you have set the global variables in CONST.py already to your personal data and code folders. Otherwise the train and validation files are not stored correctly.

## Run training using config files
This framework is built on the use of ConfigNodes that can be conviniently loaded using .yaml files. Those config files allow the user to set up own configurations and changing configurations by parsing parameters on the command line.

Training can be started by executing following command: 

```python train.py --config_file files/configs/Train_single_frame.yaml TRAIN.BATCH_SIZE 8```

which executes the train script with parameters specified in the Train_single_frame config and changes the batch size to 8.
In configs/defaults.py you can see an overview on all default parameters that can be modified by custom yaml files.

The training script can be started with various parameters (all specified in the config files) to adjust the model or the hyperparameter to fit your needs.
You can select different models (i.e. CNNGCN for the single frame GCN), different backbones (i.e. mobilenet2 for a pretrained mobilenet version 2) and different datasets. We also integrated different augmentation bundles and achieved best results with the configuration 'strongkeep'. You can further select the probability of the applied augmentation.

## Overview different models
- GCNCNN 

This is a single frame model which is trained on single images (ED and ES) with a GCN and an image encoder backbone.

- EFNet 

This network is a direct regression network of the EF that is trained on 16-frames of the heart cycle. No graph network is involved.

- EFKptsNet

This network includes a kpts regression module and a regression module which predicts both the EF and the kpts of two frames. It is trained on 16-frames of a heart cycle with known ED/ES.

- EFKptsNetSD

This network includes a classifier for identifying keyframes in an arbitrary sequence and also the EF and the kpts of the key frames. It is trained with a sequence of 16 frames and arbitrary starting points. The kpts are regressed with an MLP.

- EFGCN

This network includes a spatio-temporal GCN and a regression module which predicts both the EF and the kpts of two frames. It is trained on 16-frames of a heart cycle with known ED/ES

- EFGCNSD

This network includes a spatio-temporal GCN and a classifier for identifying keyframes in an arbitrary sequence and also the EF and the kpts of the key frames. It is trained with a sequence of 16 frames and arbitrary starting points

## Overiew Configurations and default parameter
In the following all default parameters are listed that are also used for paper evaluation. 

1) Single Frame approach

    model = 'GCNCNN'
    backbone = 'mobilenet2' (resnet18, resnet50)
    dataset = 'echonet40'
    learning_rate = 1e-4
    augmentation_type = strongkeep
    augmentation_probability = 0.90
    loss = L2
    batch_size = 128 
    optimizer = Adam
    num_frames = 1

2) Multi-frame approach with known ED/ES


    model = 'EFGCN'
    input_size = 112
    backbone = 'r3d_18'
    dataset = 'echonet_cycle'
    augmentation_type = "strong_echo_cycle"
    loss = 'L2'
    num_frames = 16
    batch_size = 4
    optimizer = 'Adam'
    learning_rate = 1e-4
    augmentation_probability = 0.90

3) Multi-frame approach with unknown ED/ES

    dataset = 'echonet_random'
    model = 'EFGCNSD'
    input_size = 112
    augmentation_type = "strong_echo_cycle"
    backbone = 'r3d_18'
    loss = ['L2', 'CrossEnt']
    num_frames = 16 #32
    batch_size = 4
    optimizer = 'Adam'
    learning_rate = 1e-4
    augmentation_probability = 0.90
    

## Monitor training 
You can monitor your results using tensorboard. Depending on your model choice, the mean kpts (averaged over all keypoints), ef (regressed value) and sd (classified frame) error can be tracked for validation.

```tensorboard --bind_all --load_fast=false --logdir= PATH_TO_EXPERIMENT_DIRECTORY```

## Run evaluation
```python eval.py --config_file files/configs/Eval_Plot.yaml EVAL.WEIGHTS=PATH_TO_WEIGHT_FILE```

The evaluation script needs a checkpoint (pth weight file set under EVAL.WEIGHTS) as input and runs the model with the assigned dataloader and weights over the entire test set. 

## Run inference 
tba

## Use your own data 
Here we explain which steps are necessary to create a new dataset that is compatible with EchoGraphs

tba
