- # EPFL CS-433 Machine Learning - Project I

  This project consists in the training of a machine learning model to identify the pressence or abscence of a Higgs boson. The dataset used comes from CERN.

  ## Overview of the project's code

  ```
  ├── data                             <- Copy Data .csv Files inside
  ├── results                          <- Results of the models
  ├── Scripts                          <- .py files of the models. utils and helpers
  │   ├── implementations_utils.py     <- utils for the main 6 models created
  │   ├── implementations.py           <- 6 models
  │   ├── project1.py    <- Cross validation functions
  │   ├── proj1_helpers.py             <- Helpers provided
  │   ├── proj1_utils.py               <- Data pre-processing functions
  │   └── proj1_visualization.py       <- Visualization functions
  ├── run.py                           <- Best Model
  ├── Report
  │   ├── ML_AAE_2019.pdf              <-Report in .pdf format
  │   └── ML_AAE_2019.tex              <-Report in LaTeX format
  ```

  ## Prerequisites

  In order to run this project, you'll need the following tools:
  [git], [conda], [python], [jupyter], [numpy], [matplotlib].

  [conda]: https://conda.io
  [python]: https://www.python.org
  [jupyter]: https://jupyter.org
  [git]: https://git-scm.com
  [numpy]: https://www.numpy.org
  [matplotlib]: https://matplotlib.org

  These are the steps to follow:

  1. Download the Python 3.7 installer for Windows, macOS, or Linux from <https://conda.io/miniconda.html> and install with default settings.
     (Skip this step if you have conda already installed Miniconda or Anaconda).
     - Windows: double-click on `Miniconda3-latest-Windows-x86_64.exe`.
     - macOS: double-click on `Miniconda3-latest-MacOSX-x86_64.pkg` or run `bash Miniconda3-latest-MacOSX-x86_64.sh` in a terminal.
     - Linux: run `bash Miniconda3-latest-Linux-x86_64.sh` in a terminal or use your package manager.
  2. Open a terminal.
     Windows: open the Anaconda Prompt from the Start menu.
  3. Change the directory to the location of the compressed zip file of the project or wherever you want to unzip it.
  4. Unzip the zip file of the project with `unzip -a mlProject1.zip`
  5. Copy your data files to the folder named 'data' or unzip the provided ones there.
  6. Open a terminal.
     Windows: open the Anaconda Prompt from the Start menu.
  7. Navigate to the folder where you stored the course material with `cd path/to/folder/ml_pro1`.

  ## Running the best method and generating a predictions file

  1. In the correct directory run the run.py script in the terminal `python run.py`
     A new file called "submission.csv" will be generated, which contains the predictions and can be uploaded to https://www.aicrowd.com/challenges/epfl-machine-learning-higgs-2019/submissions/new

  ## Try it yourself

  If you want to try the other methods, change some parameters or check the crossvalidation, it is possible to do it on the notebook project1.ipynb.

  ## Authors

  - **Adrian Gabriel Villaroel Navia**
  - **Andres Ivan Montero Cassab**
  - **Erick Antero Maraz Zuniga**
