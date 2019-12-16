# Project Road Segmentation
## Overview
The aim of this project is to be able to classify patches of satellite images and tell if they correspond to roads or correspond to background. To accomplish this
task we
## Implemented solutions
### Baseline
We used a logistic regression model as a baseline, it achieved an f1 score of
 and an accuracy score of
### ConvNet
This implementation is a CNN with several layers
### UNet
This is an architecture
### ResNet-UNet
This arquitecture implements unet

## Contents
+-- _config.yml
+-- _drafts
|   +-- begin-with-the-crazy-ideas.textile
|   +-- on-simplicity-in-technology.markdown
+-- _includes
|   +-- footer.html
|   +-- header.html
+-- _layouts
|   +-- default.html
|   +-- post.html
+-- _posts
|   +-- 2007-10-29-why-every-programmer-should-play-nethack.textile
|   +-- 2009-04-26-barcamp-boston-4-roundup.textile
+-- _data
|   +-- members.yml
+-- _site
+-- index.html

## Setup
### Packages

## Usage





For this choice of project task, we provide a set of satellite images acquired
from GoogleMaps. We also provide ground-truth images where each pixel is labeled
as road or background.

Your task is to train a classifier to segment roads in these images, i.e.
assigns a label `road=1, background=0` to each pixel.

Submission system environment setup:

1. The dataset is available from the
[CrowdAI page](https://www.crowdai.org/challenges/epfl-ml-road-segmentation).

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
