import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf


from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    LearningRateScheduler,
    ReduceLROnPlateau,
    EarlyStopping,
    TensorBoard,
)
from tensorflow.keras import backend as K
from tensorflow.keras.layers import GaussianNoise
from tensorflow.keras.regularizers import l2
from tensorflow.compat.v2.keras.layers import BatchNormalization
from tensorflow.python.client import device_lib
from utils import *




class CNN:
    def __init__(self, shape, BATCH_SIZE, WINDOW_SIZE, PATCH_SIZE, EPOCHS, STEPS_PER_EPOCH, WIDTH ):
        self.shape = shape
        self.model = self.initialize_U_NET(shape)
        self.BATCH_SIZE = BATCH_SIZE
        self.WINDOW_SIZE = WINDOW_SIZE
        self.PATCH_SIZE = PATCH_SIZE
        self.EPOCHS = EPOCHS
        self.STEPS_PER_EPOCH = STEPS_PER_EPOCH
        self.WIDTH = WIDTH
        
    def load(self, filename):
        """Loads Saved Model.
        Args:
           filename (string): name of the model
           
        """
        dependencies = {
            "recall": recall,
            "f1": f1,
        }
        self.model = load_model(filename, custom_objects=dependencies)

    def save(self, filename):
        """Saves trained model.
        Args:
           filename (string): name of the model
           
        """
        self.model.save(filename)
        
    def initialize_U_NET(self, shape):
        """Create Network Architecture.
        Args:
            shape (triplet): Size of the input layer height x width x colors (64 x 64 x 3)
        Returns:
            model (Neural Network): Architecture of the model
        """
        KERNEL3 = (3, 3)
        KERNEL5 = (5, 5)

        model = Sequential()

        model.add(Conv2D(64, KERNEL5, input_shape=shape, padding="same"))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, KERNEL3, padding="same"))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
        model.add(Dropout(0.25))

        model.add(Conv2D(256, KERNEL3, padding="same"))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
        model.add(Dropout(0.25))

        model.add(Conv2D(512, KERNEL3, padding="same"))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
        model.add(Dropout(0.25))

        model.add(Conv2D(1024, KERNEL3, padding="same"))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(
            Dense(
                128, kernel_regularizer=l2(0.000001), activity_regularizer=l2(0.000001)
            )
        )
        model.add(LeakyReLU(alpha=0.1))
        model.add(Dropout(0.5))

        model.add(
            Dense(2, kernel_regularizer=l2(0.000001), activity_regularizer=l2(0.000001))
        )
        model.add(Activation("sigmoid"))
        model.compile(
            loss="binary_crossentropy",
            optimizer=Adam(learning_rate=0.001),
            metrics=["accuracy", recall, f1],
        )

        model.summary()

        return model

    def train(self, X_train, Y_train, n_train):
        """Train the Model.

        Returns:
            History (History_Keras): History of the training
        """
        early_stopping = EarlyStopping(
            monitor="loss", patience=10, verbose=1, restore_best_weights=True,
        )
        lr_callback = ReduceLROnPlateau(
            monitor="loss", factor=0.5, patience=4, verbose=1, cooldown=1,
        )
        save_best = ModelCheckpoint(
            "CNN_dropout_0.25_1024-{epoch:03d}-{f1:03f}.h5",
            save_best_only=True,
            monitor="loss",
            verbose=1,
        )
        callbacks = [lr_callback, save_best, early_stopping]

        history = self.model.fit_generator(
            create_minibatch(
                 X_train, Y_train, n_train, self.WINDOW_SIZE, self.BATCH_SIZE, self.PATCH_SIZE, self.WIDTH
            ),
            steps_per_epoch=self.STEPS_PER_EPOCH,
            epochs=self.EPOCHS,
            use_multiprocessing=False,
            workers=1,
            callbacks=callbacks,
            verbose=1,

        )
        return history

    def classify_patches(self, X):
        """Classify image patches as either road or not.
        Args:
            X (image): part of the image to classify
        Returns:
            Predictions : Predictions for each patch
        """
        img_patches = create_patches(X, 16, 16, padding=24)
        predictions = self.model.predict(img_patches)
        predictions = (predictions[:, 0] < predictions[:, 1]) * 1

        return predictions.reshape(X.shape[0], -1)

