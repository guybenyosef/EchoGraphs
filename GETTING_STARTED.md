## Preprocess the EchoNet dataset

If you want to train the EchoGraph model with the EchoNet data, you need to register on the webpage first and then download the data. 
Then you can run the preprocess script providing the data folder to preprocess the original echonet data into a format that is readable by the training code provided in this repository. This script will use the images and the .csv sheet to create training images and labels as well as txt files that store all filenames for training, validation and test according to the splits provided by the EchoNet authors.
40 keypoints are extracted from the .csv file indicated as X1 X2 Y1 Y2 excluding the first two points since they are the apex and basal point and not necessarily part of the contour. Further, the EF, ED volume, ES volume, ED frame index and ES frame index are extracted and stored in a npz file. Some preprocessing functions like i.e. the video reader function are taken from the EchoNet repository.

TODO add the final structure of the label and the cmd arguments

## Use your own data 
TODO write how the general input needs to look like and how 'dataset' would need to be modified


## Run training 
The training script can be started with various parameters to adjust the model or the hyperparameter to fit your needs.
You can select different models (i.e. CNNGCN for the single frame GCN), different backbones (i.e. mobilenet2 for a pretrained mobilenet version 2) and different datasets. We also integrated different augmentation bundles and achieved best results with the configuration XXX. You can further select the probability of the applied augmentation.
TODO explain command with all parameters that can be safely changed (some will probably result in failure)

## Overview different models
TODO 

## Overiew Configurations and default parameter
TODO add all default parameters here

1) Single Frame approach

2) Multi-frame approach with known ED/ES

3) Multi-frame approach with unknown ED/ES

## Run interference

