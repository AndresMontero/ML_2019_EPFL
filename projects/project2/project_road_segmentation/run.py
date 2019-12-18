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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf
from utils import *
from models.cnn import *


############ Set random seed and print available CPU and GPU
np.random.seed(8)
print(device_lib.list_local_devices())


############ Define Parameters of the model
BATCH_SIZE = 1000
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
FLAG_LOAD_WEIGTHS = True

if not FLAG_LOAD_WEIGTHS:
    ############ Load Images
    print("Importing data...")
    image_dir_train = "data/training/images/"
    files = os.listdir(image_dir_train)
    n_train = len(files)
    print(f"Loading training images, images loaded: {n_train} ")
    imgs_train = np.asarray(
        [load_image(image_dir_train + files[i]) for i in range(n_train)]
    )
    gt_dir_train = "data/training/groundtruth/"
    print(f"Loading groundtruth images, images loaded: {n_train} ")
    gt_imgs_train = np.asarray(
        [load_image(gt_dir_train + files[i]) for i in range(n_train)]
    )

    ############ Data Augmentation
    print("Data augmentation...")
    X_train, Y_train = imag_rotation_aug(imgs_train, gt_imgs_train)
    X_train = np.asarray(X_train)
    Y_train = np.asarray(Y_train)
    print(X_train.shape)
    print(Y_train.shape)
    n_train = Y_train.shape[0]
    history = model.train()
    model.save("best_cnn.h5")
else:
    model.load("best_cnn.h5")

############ Generate submission file
print("############################# Loading Weights.......")
model.load("best_cnn.h5")

print("############################# Loading Test Images.......")
image_filenames = []
for i in range(1, 51):
    image_filename = "data/test_set_images/test_" + str(i) + "/test_" + str(i) + ".png"
    image_filenames.append(image_filename)

print("############################# Generating Submission file")
submission_filename = "best_cnn.csv"
generate_submission(model, submission_filename, *image_filenames)
print("############################# File generated", submission_filename)
