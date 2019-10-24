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
In order to run this project, you'll need the following tools:
[git], [conda], [python], [jupyter], [numpy], [scipy], [matplotlib].

[conda]: https://conda.io
[python]: https://www.python.org
[jupyter]: https://jupyter.org
[git]: https://git-scm.com
[numpy]: https://www.numpy.org
[matplotlib]: https://matplotlib.org


These are the steps to follow:

1. Download the Python 3.7 installer for Windows, macOS, or Linux from <https://conda.io/miniconda.html> and install with default settings.
   Skip this step if you have conda already installed Miniconda or Anaconda).
   * Windows: double-click on `Miniconda3-latest-Windows-x86_64.exe`.
   * macOS: double-click on `Miniconda3-latest-MacOSX-x86_64.pkg` or run `bash Miniconda3-latest-MacOSX-x86_64.sh` in a terminal.
   * Linux: run `bash Miniconda3-latest-Linux-x86_64.sh` in a terminal or use your package manager.
2. Open a terminal.
   Windows: open the Anaconda Prompt from the Start menu.
3. Change the directory to the location of the compressed zip file of the project or wherever you want to unzip it.
4. Unzip the zip file of the project with `unzip -a mlProject1.zip`
5. Create the environment required with the packages required for the project with `conda env create -f environment.yml`.

Every time you want to run the project you should:

6. Open a terminal.
   Windows: open the Anaconda Prompt from the Start menu.
7. Activate the environment with `conda activate ml_pro1`.
8. Navigate to the folder where you stored the course material with `cd path/to/folder/ml_pro1`.
9. Start Jupyter with `jupyter lab`.
10. Once done, you can run `conda deactivate` to leave the `ml_pro1` environment.


## Running the best method and generating a predictions file

1. If your terminal is not in the location of the project files, change your directory to that location.
2. In the correct directory run the run.py script in the terminal `python run.py`
   A new file called "submission.csv" will be generated, which contains the predictions and can be uploaded to https://www.aicrowd.com/challenges/epfl-machine-learning-higgs-2019/submissions/new

## View the process
If you want to try the other methods or change some parameters, it is possible to do it in the project1.ipynb through jupyter lab.


## Authors

- **Adrian Gabriel Villaroel Navia**
- **Andres Ivan Montero Cassab**
- **Erick Antero Maraz Zuñiga**

