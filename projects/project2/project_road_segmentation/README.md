# Project Road Segmentation
## Overview
The aim of this project is to classify patches of satellite images into roads or background. To accomplish this task we propose 3 neural network architectures implemented using the framework [Tensorflow 2.0](https://www.tensorflow.org). Specifically we are using the high-level API Keras.   For a detailed explanation of the architectures used please review the file Machine_learning_AAE_project2_report.pdf.

This repository contains all the files required to train the architectures, generate their history plots and run the best model with its optimum weights.
## Team submission ID on AICrowd

Andres Montero: 31018    

## Contents
Depending on the model to run, the data should be separated in training and validating folders
with the same structure.

```
|── data                                            <- Empty folder, copy the dataset here
├── Report
│   ├── Machine_learning_AAE_project2_report.pdf    <- Report in .pdf format
├── models
    └── cnn.py                                      <- CNN architecture file (best)
    └── resnet.py                                   <- RESNET_UNET architecture file
    └── unet.py                                     <- UNET architecture file
├── Project2_CNN.ipynb                              <- Notebook to run the CNN architecture (best model)         ├── best_cnn_colab.h5 
├── Project2_UNET.ipynb                             <- Notebook to run the UNET architecture
├── Project2_RESNET_UNET.ipynb                      <- Notebook to run the RESNET_UNET architecture
├── Project2_segment_aerial_images.ipynb            <- Notebook to run the baseline provided
├── Project2_CNN_colab.ipynb                       <- Notebook to run the best model in Colab
├── run.py                                          <- File to run the best model predictions
├── utils.py                                        <- File containing utility functions
├── README.md                                       <- Readme file
```

## Setup
1. Install the last version of anaconda according to you OS
* Linux: https://docs.anaconda.com/anaconda/install/linux/   
* Windows: https://docs.anaconda.com/anaconda/install/windows/
* MacOS: https://docs.anaconda.com/anaconda/install/mac-os/
2. Open an anaconda shell (Windows) or terminal (Linux/MacOS) and run the following command:
  `conda create --name road_seg anaconda python==3.7.5`.
3. In the conda environment recently created, run the following commands:
- `conda install -c conda-forge opencv==3.4.2`
- `pip install tensorflow==2.0.0`

4. If you have access to a CUDA-enabled GPU you can install it following the instructions provided in this link [TensorFlow's installation instructions](https://www.tensorflow.org/install/gpu)

## Running models

1. Download the zip file AAE_project2.zip of the project
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
5. (Important note) For the models UNET and RESNET_UNET you need to create a validating dataset, hence you have to take some images (we used 80-20), and groundtruths from the training set, create a folder named validating and insert them there. For the CCN model just keep the previous folder structure with the entire set of images.
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
- Project2_CNN_colab.ipynb
- Project2_CNN.ipynb
- Project2_UNET.ipynb
- Project2_RESNET_UNET.ipynb
- Project2_segment_aerial_images.ipynb

To obtain the weights used for the final submission please run the Project2_CNN_colab.ipynb without changing the parameters following the guide in section "Alternative Setup Google Colab". 

## Run the predictions
**Note: to run the predictions you  must keep the previous folder structure**

1. Move to the root of the project folder.
2. Run the following command:
`python run.py`
3. After the program has run completely, the submission file best_cnn.csv will be generated.
4. By default the program will load the best weights trained in colab. If you want to train locally (you need to change the flag "FLAG_LOAD_WEIGTHS" of the run.py to False )

## Alternative Setup on Google Colab
1. Upload the uncompressed zip AAE_project2 to a folder in Google Drive
2. Add two cells at the start of your notebook:
- Execute the cell containing:
`from google.colab import drive
drive.mount('/content/drive')`
- To change the path to the folder containing the uncompressed data:
``cd <put_your_path_to_the_project_here>``

3. Open the notebook Project2_CNN_colab.ipynb  in cola

4. Run the cells and the submission file "best_cnn_colab.csv" will be generated in the drive folder.


## Authors
- Andres Ivan Montero Cassab
- Erick Antero Maraz Zuñiga
- Adrian Gabriel Villarroel Navia
