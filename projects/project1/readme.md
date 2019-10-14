# EPFL CS-433 Machine Learning - Project I

This project consists in the training of a model to classify if a given particule is a Higgs Bosson or not, the data used is from CERN. 

## Overview of the project's code

```
├── data                             <- Copy Data .csv Files inside
├── results                          <- Results of the models
├── Scripts                          <- .py files of the models. utils and helpers
│   ├── implementations_utils.py     <- utils for the main 6 models created
│   ├── implementations.py           <- 6 models 
│   ├── proj1_cross_validation.py    <- Cross validation functions 
│   ├── proj1_helpers.py             <- Helpers provided 
│   ├── proj1_utils.py               <- Data pre-processing functions
│   └──  proj1_visualization.py       <- Visualization functions
├── run.py                           <- Best Model 
├── Report
│   ├── ML_AAE_2019.pdf              <-Report in .pdf format
│   └── ML_AAE_2019.tex              <-Report in LaTeX format
```

## Prerequisites

1. You will need Anaconda to run this project:

   | Operating System | Tutorial                                            |
   | ---------------- | --------------------------------------------------- |
   | Windows          | https://docs.anaconda.com/anaconda/install/windows/ |
   | Ubuntu           | https://docs.anaconda.com/anaconda/install/linux/   |
   | Mac              | https://docs.anaconda.com/anaconda/install/mac-os/  |


2. Then install **matplotlib** python package:

   ```
   conda install matplotlib
   ```

## Installing

1. In a terminal, change your directory to the location of the compressed zip file of the project.

   ```
   cd {path_of_zip_file}
   ```

2. Unzip the zip file of the project.

   ```
   unzip -a mlProject1.zip
   ```

3. Download the "test.csv" and "train.csv" files from the following link: https://www.aicrowd.com/challenges/epfl-machine-learning-higgs-2019/dataset_files

4. Copy the downloaded files inside the **/data** folder.

## Running the best method and generating a predictions file

1. If your terminal is not in the location of the project files, change your directory to that location.

   ```
   cd {path_of_project_files}
   ```

2. Run the run.py script in the terminal.

   ```
   python run.py
   ```

   A new file called "submission.csv" will be generated, which contains the predictions and can be uploaded to https://www.aicrowd.com/challenges/epfl-machine-learning-higgs-2019/submissions/new


## Authors

- **Adrian Villaroel Navia**
- **Andres Ivan Montero Cassab**
- **Erick **
