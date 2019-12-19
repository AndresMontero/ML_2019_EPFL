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


def down(input_layer, filters, pool=True):
    """Create convloutional and residual layers to reduce dimensions.
    Args:
        input_layer (layer): input layer before convolution
        filters (numpy.int64): number of filters
    Returns:
        max_pool (layer): layer after max-pooling
        residual (connection ): connection to connect with next layers
    """
    batchnorm = BatchNormalization()(input_layer)
    conv1 = Conv2D(filters, (5, 5), padding="same", activation="relu")(batchnorm)
    residual = Conv2D(filters, (3, 3), padding="same", activation="relu")(conv1)
    if pool:
        max_pool = MaxPool2D()(residual)
        return max_pool, residual
    else:
        return residual


def up(input_layer, residual, filters):
    """Create convloutional and residual layers to increase dimensions.
    Args:
        input_layer (layer): input layer before convolution\
        residual (connection ): connection to connect with next layers
        filters (numpy.int64): number of filters
    Returns:
        conv2 (layer): convolutional layer
    """
    filters = int(filters)
    batchnorm = BatchNormalization()(input_layer)
    upsample = UpSampling2D()(batchnorm)
    upconv = Conv2D(filters, kernel_size=(2, 2), padding="same")(upsample)
    concat = Concatenate(axis=3)([residual, upconv])
    conv1 = Conv2D(filters, (5, 5), padding="same", activation="relu")(concat)
    conv2 = Conv2D(filters, (3, 3), padding="same", activation="relu")(conv1)
    return conv2


class U_NET:
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
        # Load the model (used for submission)
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
        # Make a custom U-nets implementation.
        filters = 64
        input_layer = Input(shape=shape)
        layers = [input_layer]
        residuals = []

        # Down 1, 64
        d1, res1 = down(input_layer, filters)
        residuals.append(res1)
        filters *= 2

        # Down 2, 32
        d2, res2 = down(d1, filters)
        residuals.append(res2)
        filters *= 2

        # Down 3, 16
        d3, res3 = down(d2, filters)
        residuals.append(res3)
        filters *= 2

        # Down 4, 8
        d4, res4 = down(d3, filters)
        residuals.append(res4)
        filters *= 2

        # Up 1, 8
        up1 = up(d4, residual=residuals[-1], filters=filters / 2)
        filters /= 2

        # Up 2,  16
        up2 = up(up1, residual=residuals[-2], filters=filters / 2)
        filters /= 2

        # Up 3, 32
        up3 = up(up2, residual=residuals[-3], filters=filters / 2)
        filters /= 2

        # Up 4, 64
        up4 = up(up3, residual=residuals[-4], filters=filters / 2)

        conv_1 = Conv2D(1, 1, activation="relu")(up4)
        flaten = Flatten()(conv_1)
        batch_1 = BatchNormalization()(flaten)
        out = Dense(
            2,
            activation="sigmoid",
            kernel_regularizer=l2(0.00001),
            activity_regularizer=l2(0.00001),
        )(batch_1)

        model = Model(input_layer, out)
        model.compile(
            loss="binary_crossentropy",
            optimizer=Adam(learning_rate=0.001),
            metrics=["accuracy", recall, f1],
        )

        return model

    def train(self, X_train, Y_train, n_train,  X_val, Y_val, n_val):
        """Train the Model.

        Returns:
            History (History_Keras): History of the training
        """
        early_stopping = EarlyStopping(
            monitor="val_loss", patience=10, verbose=1, restore_best_weights=True,
        )
        lr_callback = ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=5, verbose=1, cooldown=1,
        )
        save_best = ModelCheckpoint(
            "saved_models/Unet_batchnorm_validation-{epoch:03d}-{val_f1:03f}.h5",
            save_best_only=True,
            monitor="val_loss",
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
            validation_data=create_minibatch(
                X_val, Y_val, n_val, self.WINDOW_SIZE, self.BATCH_SIZE, self.PATCH_SIZE, self.WIDTH
            ),
            validation_steps=self.STEPS_PER_EPOCH / 3,
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
