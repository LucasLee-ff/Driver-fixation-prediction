# Driver Fixation Prediction based on CNN-Transformer Hybrid Model
## Environment
    Python 3.7.5
    Pytorch 1.11

## Introduction
This is a CNN-Transformer based model named TransCDNN. It extracts low-level features on the images from driving scenarios through a multi-layer convolutional neural network and then uses Transformer encoder in the last part of the feature extraction procedure to encode high-level features. With CNN's ability to capture patterns between adjacent pixels and Transformer's excellent capacity for modeling long-range representations, this model performs better in extracting high-level semantic features of images.

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