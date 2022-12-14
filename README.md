# Driver Fixation Prediction based on CNN-Transformer Hybrid Model
## Environment
    Python 3.7.5
    Pytorch 1.11

## Introduction
This is a CNN-Transformer based model named **TransCDNN**. It extracts low-level features on the images from driving scenarios through a multi-layer convolutional neural network and then uses Transformer encoder in the last part of the feature extraction procedure to encode high-level features. With CNN's ability to capture patterns between adjacent pixels and Transformer's excellent capacity for modeling long-range representations, this model performs better in extracting high-level semantic features of images.

## Model Architecture
The model is mainly composed of encoder and decoder. The encoder includes CNN encoder and Transformer encoder. The decoder is the combination of a series of convolution and up-sampling. Skip connection is also implemented in this model

![model](https://github.com/LucasLee-ff/Driver-fixation-prediction/blob/master/demo/model.jpg)

## Project File
File `main.py` is where you can run and train the model.  
Directory `/TransCDNN_model` includes the following files

    config.py
    conv_encoder.py
    ecanet.py
    vit_cdnn_modeling.py
`config.py` determines some configurations of the TransCDNN model.  
`conv_encoder.py` is the modeling of the CNN component.  
`ecanet.py` is not a must, you can insert this channel attention module anywhere you want.  
`vit_cdnn_modeling.py` is the modeling of the entire TransCDNN model. You can also find a model named "DeepCDNN" in this file, which is a fixation prediction model with only CNN and decoder.  

## Datasets
### TrafficGaze
The main dataset used in this project is [TrafficGaze](https://github.com/taodeng/CDNN-traffic-saliency) from University of Electronic Science and Technology of China. It can be directly used after downloading on the url above.

### BDD-A
The BDD-A (Berkeley DeepDrive-Attention) is also an alternative. Details can be found in the [Berkeley DeepDrive official website](https://deepdrive.berkeley.edu/).

## Demo
The demo video can be found in this repository `/demo/fixation_prediction_demo.mp4`

![demo_picture](https://github.com/LucasLee-ff/Driver-fixation-prediction/blob/master/demo/demo.jpg)

## Reference
The code of ViT is mainly from [TransUNet](https://github.com/Beckschen/TransUNet).