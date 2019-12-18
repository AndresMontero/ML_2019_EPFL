""" Course: Machine Learning
    Projeect 2: Satelite Images Road Segmentation
    Authors:
        - Maraz Erick
        - Montero Andres
        - Villarroel Adrian
    This script runs the best ML model and generates the
    submission file.
"""
# Imports needed for the project to run
import os
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

if !FLAG_LOADWEIGTHS:
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
    model.save("best_model.h5")
else:
    model.load("best_model.h5")


############ Generate submission file
model.model.summary()
test_images = []
for i in range(1, 51):
    test_images = "data/test_set_images/test_" + str(i) + "/test_" + str(i) + ".png"
    test_images.append(test_images)

submission_filename = "best_model.csv"
generate_submission(model, submission_filename, *test_images)
print("File generated: ", submission_filename)
