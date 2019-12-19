# Project Road Segmentation
## Overview
The aim of this project is to classify patches of satellite into roads or background. To accomplish this task we propose 3 neural network architectures implemented using the framework [Tensorflow 2.0](https://www.tensorflow.org). Specifically we are using the high-level API Keras.   

This repository contains all the files required to train the architectures, generate their history plots and run the best model with its optimum weights.
## Team submission ID on AICrowd

## Contents
Depending on the model to run, the data should be separated in training and validation folders
with the same structure.
```
|── data                                            <- Empty folder, copy the dataset here
├── Report
│   ├── Machine_learning_AAE_project2_report.pdf    <- Report in .pdf format
│   └── Machine_learning_AAE_project2_report.tex    <- Report in LaTeX format
├── models
    └── cnn.py                                      <- CNN architecture file (best)
    └── resnet.py                                   <- RESNET_UNET architecture file
    └── unet.py                                     <- UNET architecture file
├── Project2_CNN.ipynb                              <- Notebook to run the CNN architecture (best model)                               
├── Project2_UNET.ipynb                             <- Notebook to run the UNET architecture
├── Project2_RESNET_UNET.ipynb                      <- Notebook to run the RESNET_UNET architecture
├── Project2_segment_aerial_images.ipynb            <- Notebook to run the baseline provided
├── run.py                                          <- File to run the best model predictions
├── utils.py                                        <- File containing utility functions
├── README.md                                       <- Readme file
```

## Setup

1. Install the last version of anaconda according to you OS
Linux: https://docs.anaconda.com/anaconda/install/linux/   
Windows: https://docs.anaconda.com/anaconda/install/windows/
MacOS: https://docs.anaconda.com/anaconda/install/mac-os/

2. Open an anaconda shell (Windows) or terminal (Linux) and run the following command:
`conda create --name road_seg anaconda python==3.7.5`.

3. In the conda environment recently created, run the following commands:
- `conda install -c conda-forge opencv==3.4.2`
- `pip install tensorflow==2.0.0`

4. If you have access to a CUDA-enabled GPU you can install it following the instructions provided in this link [TensorFlow's installation instructions](https://www.tensorflow.org/install/gpu)

## Running

1. Download the dataset
[CrowdAI page](https://www.crowdai.org/challenges/epfl-ml-road-segmentation).

2. 

To check if your GPU supports CUDA go to the following link: https://developer.nvidia.com/cuda-gpus.

## Setup for GPU-CUDA enabled computers
1. Download and install the latest version of [anaconda](https://www.anaconda.com/distribution/) for the corresponding Python 3.7 according to your OS.
2. Create a Python environment by opening a terminal (in Linux or MacOs or an Anaconda prompt shell in Windows) and running the following command
3. Activate the Python environment with the command `conda activate <replace-name-of-env>`.
**NOTE: Do not install tensorflow yet, we will do it in the next section**
4.  and follow them.

After following these steps we will continue installing all the packages.

### Packages installation
The packages we are working with are:
- opencv = 3.4.2
- pillow = 6.2.1
- tensorflow = 2.0.0
- cudatoolkit = 10.0.130 (It can vary depending on which installation of cuda toolkit you are using)
- cudnn = 7.6.5 (It can vary depending on which installation of cuda toolkit we are using)

Don't close the terminal with the environment yet! We need to install additional packages


## Alternative Setup Google Colab
1. Upload the folder containing all the files and the data to drive
2. Add two cells at the start of your notebook:
- To mount your drive:
`from google.colab import drive
drive.mount('/content/drive')`
- To change the path to the folder containing all files and data:
``%cd drive/My\ Drive/<root-folder>/``

## Implemented solutions
### Baseline
We used a simple CNN model as a baseline, it achieved an F1 score of
 and an accuracy score of
### ConvNet
This implementation is a CNN with several layers
### UNetÂ
This is an architecture
### ResNet-UNet
This arquitecture implements unet




## Alternative Run on Colab





For this choice of project task, we provide a set of satellite images acquired
from GoogleMaps. We also provide ground-truth images where each pixel is labeled
as road or background.

Your task is to train a classifier to segment roads in these images, i.e.
assigns a label `road=1, background=0` to each pixel.

Submission system environment setup:



2. Obtain the python notebook `segment_aerial_images.ipynb` from this github
folder, to see example code on how to extract the images as well as
corresponding labels of each pixel.

The notebook shows how to use `scikit learn` to generate features from each
pixel, and finally train a linear classifier to predict whether each pixel is
road or background. Or you can use your own code as well. Our example code here
also provides helper functions to visualize the images, labels and predictions.
In particular, the two functions `mask_to_submission.py` and
`submission_to_mask.py` help you to convert from the submission format to a
visualization, and vice versa.

3. As a more advanced approach, try `tf_aerial_images.py`, which demonstrates
the use of a basic convolutional neural network in TensorFlow for the same
prediction task.

Evaluation Metric:
 [F1 score](https://en.wikipedia.org/wiki/F1_score)
## Authors
- Andres Ivan Montero Cassab
- Erick Antero Maraz Zuñiga
- Adrian Gabriel Villarroel Navia
