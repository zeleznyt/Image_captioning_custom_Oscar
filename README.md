# Image captioning pipeline based on Oscar

This repository contains source codes I used in my master thesis Image Captioning using Deep Learning. 

## Captioning pipeline

The work consits of creating the image captioning pipeline based on the existing method [Oscar](https://github.com/microsoft/Oscar). Its input requires the features of the detected objects in the image. For this purpose I used detection architecture Faster-R-CNN implemented in framework [Detectron2](https://github.com/facebookresearch/detectron2). In order to obtain the correct structure that is needed on the input of Oscar, I modified the Detectron2 codes. They can be found on my [fork](https://github.com/zeleznyt/detectron2).

### Installation

In the pipeline, I use two separate environments for each, Detectron2 and Oscar. The builds can be found in appropriate repositories:
* [Detectron2](https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md)
* [Oscar](https://github.com/microsoft/Oscar/blob/master/INSTALL.md)
