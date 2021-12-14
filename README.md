# SRCNN paper replication

## Desciption
SRCNN paper( Image Super-Resolution Using Deep Convolutional Networks 
) replication for 2021 CV Final project

## Prerequisite
* python 3.7
* pytorch 1.8.0
* torchivision 0.9.0

## Files
* SRCNN.ipynb
    * training and evaluation code, save the model and average test psnr list
* SRCNN_classes.py, datasets.py, utils.py
    * classes and functions for SRCNN.ipynb
* reconstructed_image.py
    * extract reconstructed image using SRCNN 
* results.ipynb
    * extract average test psnr graph

## Used Data
* 91 image : training
* Set 5 : evaluating

Reference Code에서 제공하는 upscaling factor 3으로 가공된 데이터를 사용함

## Reference Code
https://github.com/yjn870/SRCNN-pytorch