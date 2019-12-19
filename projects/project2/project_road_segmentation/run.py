""" Course: Machine Learning
    Projeect 2: Satellite Images Road Segmentation
    Authors:
        - Maraz Erick
        - Montero Andres
        - Villarroel Adrian
    This script runs the best ML model and generates the
    submission file.
"""
# Imports needed for the project to run
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # to avoid verbose of cuda 
import tensorflow as tf
from utils import *
from models.cnn import *


############ Set random seed and print available CPU and GPU
np.random.seed(8)
print(device_lib.list_local_devices())


############ Define Parameters of the model
BATCH_SIZE = 300
WINDOW_SIZE = 64
PATCH_SIZE = 16
EPOCHS = 200
STEPS_PER_EPOCH = 100
WIDTH = 448
print("############################# Instansiating Model")
model = CNN(
    shape=(WINDOW_SIZE, WINDOW_SIZE, 3),
    BATCH_SIZE=BATCH_SIZE,
    WINDOW_SIZE=WINDOW_SIZE,
    PATCH_SIZE=PATCH_SIZE,
    EPOCHS=EPOCHS,
    STEPS_PER_EPOCH=STEPS_PER_EPOCH,
    WIDTH=WIDTH,
)
FLAG_LOAD_WEIGTHS = True # Set this to False if you want to train again the model, we recomend using the jupyter notebook Project2_CNN_collab on google colab, please review readme 

if not FLAG_LOAD_WEIGTHS:
    ############ Load Images
    print("Importing data for training...")
    image_dir_train = "data/training/images/"
    files = os.listdir(image_dir_train)
    n_train = len(files)
    print(f"Loading training images, images loaded: {n_train} ")
    imgs_train = np.asarray(
        [load_img(image_dir_train + files[i]) for i in range(n_train)]
    )
    gt_dir_train = "data/training/groundtruth/"
    print(f"Loading training groundtruth images, images loaded: {n_train} ")
    gt_imgs_train = np.asarray(
        [load_img(gt_dir_train + files[i]) for i in range(n_train)]
    )

    ############ Data Augmentation
    print("Data augmentation, please wait...")
    X_train, Y_train = imag_rotation_aug(imgs_train, gt_imgs_train)
    X_train = np.asarray(X_train)
    Y_train = np.asarray(Y_train)
    print(X_train.shape)
    print(Y_train.shape)
    n_train = Y_train.shape[0]
    history = model.train(X_train, Y_train, n_train)
    model.save("best_cnn.h5")

############ Generate submission file
print("############################# Loading Weights.......")
model.load("best_cnn_colab.h5")

print("############################# Loading Test Images.......")
image_filenames = []
for i in range(1, 51):
    image_filename = "data/test_set_images/test_" + str(i) + "/test_" + str(i) + ".png"
    image_filenames.append(image_filename)

print("############################# Generating Submission file")
submission_filename = "best_cnn_colab.csv"
generate_submission(model, submission_filename, *image_filenames)
print("############################# File generated", submission_filename)
