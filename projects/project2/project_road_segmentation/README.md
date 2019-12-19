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
├── models
    └── cnn.py                                      <- CNN architecture file (best)
    └── resnet.py                                   <- RESNET_UNET architecture file
    └── unet.py                                     <- UNET architecture file
├── Project2_CNN.ipynb                              <- Notebook to run the CNN architecture (best model)                               
├── Project2_UNET.ipynb                             <- Notebook to run the UNET architecture
├── Project2_RESNET_UNET.ipynb                      <- Notebook to run the RESNET_UNET architecture
├── Project2_segment_aerial_images.ipynb            <- Notebook to run the baseline provided
├── Project2_CNN_collab.ipynb                       <- Notebook to run the best model in Colab
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

## Running models

1. Download the zip file <name> of the project

2. Got to the path containing the zip file and uncompress it

3. Download the dataset
[CrowdAI page](https://www.crowdai.org/challenges/epfl-ml-road-segmentation).

4. Decompress the dataset in the folder data and make sure that you have the
following folder structure:
`|── data                               
    └── test_set_images
        └── test_1
        └── test_2
        ...
        └── test_50
    └── training
        └── images
        └── groundtruth
`
5. (Important note) For the models UNET and RESNET_UNET you need to create a validation dataset, hence you have to take some images and groundtruths from the training set, create a folder validating and insert them there. For the CCN model just keep the previous folder structure.
`|── data                               
    └── test_set_images
        └── test_1
        └── test_2
        ...
      └── test_50
    └── training
        └── images
        └── groundtruth
    └── validating
        └── images
        └── groundtruth
`
6. Run any of the notebooks
- Project2_CNN
- Project2_UNET.ipynb
- Project2_RESNET_UNET.ipynb
- Project2_segment_aerial_images.ipynb

## Run the predictions
**Note: to run the predictions you  must keep the previous folder structure**

1. Move to the root of the project folder.

2. Run the following command:
`python run.py`

3. After the program has run completely, the submission file best_cnn.csv will be generated.

## Alternative Setup Google Colab
1. Upload the folder containing all the files and the data to drive.
2. Add two cells at the start of your notebook:
- To mount your drive:
`from google.colab import drive
drive.mount('/content/drive')`
- To change the path to the folder containing all files and data:
``%cd drive/My\ Drive/<root-folder>/``

## Alternative Run on Colab

1. Open colab in a browser an upload the notebook:
Project2_CNN_collab.ipynb  
2. Upload the data with the folder structure like the previouse section, point 5.
3. In the colab notebook, replace the line
`cd <put_your_path_to_the_project_here>` with the path to the folder that has the dataset
4. Run the cells and the submission file "best_cnn.csv" will be generated in the drive folder.


## Authors
- Andres Ivan Montero Cassab
- Erick Antero Maraz Zuñiga
- Adrian Gabriel Villarroel Navia
