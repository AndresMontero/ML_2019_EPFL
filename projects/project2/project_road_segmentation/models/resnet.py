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
from tensorflow.keras.applications.resnet50 import ResNet50

from tensorflow.compat.v2.keras.layers import BatchNormalization
from tensorflow.python.client import device_lib
from utils import *

class resnet_unet_model:
    """This class creates the ResNet-UNet model

    Loads a ResNet50 with the weights pre-trained on the "Imagenet"
    dataset, then adds decoder blocks and finally a block with dense
    layers"

    """
    def __init__(
        self, shape, batch_normalization, activation, dropout, amsgrad=False, lr=1e-4, BATCH_SIZE=100, WINDOW_SIZE=64, PATCH_SIZE=16, EPOCHS=80, STEPS_PER_EPOCH=100, WIDTH=448
    ):
        """Initialize the resnet_unet model
                Args:
                    shape: (tuple):             input shape
                    batch_normalization (bool): use batch normalization
                    activation (str):           select which activation to use
                    dropout (float):            select the probability of dropout
                    amsgrad (bool):             use amsgrad for Adam optimizer
                    lr (float):                 learning rate
                Returns:
                    resnet unet model object
        """
        self.shape = shape
        self.batch_normalization = batch_normalization
        self.activation = activation
        self.dropout = dropout
        self.amsgrad = amsgrad
        self.lr = lr
        self.model = self.create_resnet_unet_model()
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
        dependencies = {"recall": recall, "f1": f1}
        self.model = load_model(filename, custom_objects=dependencies)

    def save(self, filename):
        """Saves trained model.
        Args:
           filename (string): name of the model

        """
        self.model.save(filename)

    def conv_act(self, inputs, out_filters, activation="relu"):
        """Create a 2D convolutional layer with an activation
            Args:
                inputs (tensorflow.python.framework.ops.Tensor): inputs to the block
                out_filters (int)                              : number of output filters
                activation (str):                              : activation function
            Returns:
                A tensorflow.python.framework.ops.Tensor object
        """
        return Conv2D(
            filters=out_filters,
            activation=activation,
            kernel_size=3,
            strides=1,
            padding="same",
        )(inputs)

    def decoder(
        self,
        inputs,
        mid_filters=512,
        out_filters=256,
        activation="relu",
        block_name="decoder",
    ):
        """ Create a decoder block
            Args:
                inputs (tensorflow.python.framework.ops.Tensor): inputs to the block
                mid_filters (int):                             : number of mid filters
                out_filters (int)                              : number of output filters
                activation (str):                              : activation function
                block_name (str):                              : name of the block to use
            Returns:
                A tensorflow.python.framework.ops.Tensor object
        """
        with K.name_scope(block_name):
            if activation == "leaky_relu":
                activation = None
                conv = LeakyReLU(alpha=0.3)(
                    self.conv_act(inputs, mid_filters, activation)
                )
            else:
                conv = self.conv_act(inputs, mid_filters, activation)
            conv_tr = Conv2DTranspose(
                filters=out_filters,
                activation=activation,
                kernel_size=4,
                strides=2,
                padding="same",
            )(conv)
        return conv_tr

    def create_resnet_unet_model(self):
        """ Create the model
        """

        max_pooling_size = 2
        max_pooling_strd = 2

        num_classes = 2
        resnet50 = ResNet50(
            include_top=False,
            weights="imagenet",
            classes=num_classes,
            input_shape=self.shape,
        )

        resnet50.compile(
            optimizer=Adam(lr=self.lr, amsgrad=self.amsgrad), loss="binary_crossentropy"
        )

        conv5_out = resnet50.get_layer("conv5_block3_out").output
        conv4_out = resnet50.get_layer("conv4_block6_out").output
        conv3_out = resnet50.get_layer("conv3_block4_out").output
        conv2_out = resnet50.get_layer("conv2_block3_out").output

        pool = MaxPooling2D(max_pooling_size, strides=max_pooling_strd, padding="same")(
            resnet50.get_output_at(0)
        )

        dec_center = self.decoder(
            pool,
            mid_filters=self.shape[0] * 2,
            out_filters=self.shape[0],
            activation=self.activation,
            block_name="decoder_center",
        )
        if self.batch_normalization:
            dec_center = BatchNormalization()(dec_center)
        if self.dropout > 0:
            dec_center = Dropout(dropout)(dec_center)

        cat1 = Concatenate()([dec_center, conv5_out])
        dec5 = self.decoder(
            cat1,
            mid_filters=int(self.shape[0] * 2),
            out_filters=int(self.shape[0]),
            activation=self.activation,
            block_name="decoder5",
        )
        if self.batch_normalization:
            dec5 = BatchNormalization()(dec5)
        if self.dropout > 0:
            dec5 = Dropout(self.dropout)(dec5)

        cat2 = Concatenate()([dec5, conv4_out])
        dec4 = self.decoder(
            cat2,
            mid_filters=int(self.shape[0] * 2),
            out_filters=int(self.shape[0]),
            activation=self.activation,
            block_name="decoder4",
        )
        if self.batch_normalization:
            dec4 = BatchNormalization()(dec4)
        if self.dropout > 0:
            dec4 = Dropout(self.dropout)(dec4)

        cat3 = Concatenate()([dec4, conv3_out])
        dec3 = self.decoder(
            cat3,
            mid_filters=int(self.shape[0]),
            out_filters=int(self.shape[0] // 4),
            activation=self.activation,
            block_name="decoder3",
        )
        if self.batch_normalization:
            dec3 = BatchNormalization()(dec3)
        if self.dropout > 0:
            dec3 = Dropout(self.dropout)(dec3)

        cat2 = Concatenate()([dec3, conv2_out])
        dec2 = self.decoder(
            cat2,
            mid_filters=int(self.shape[0] // 2),
            out_filters=int(self.shape[0] // 2),
            activation=self.activation,
            block_name="decoder2",
        )
        if self.batch_normalization:
            dec2 = BatchNormalization()(dec2)
        if self.dropout > 0:
            dec2 = Dropout(self.dropout)(dec2)

        dec1 = self.decoder(
            dec2,
            mid_filters=int(self.shape[0] // 2),
            out_filters=int(self.shape[0] // 8),
            activation=self.activation,
            block_name="decoder1",
        )
        if self.batch_normalization:
            dec1 = BatchNormalization()(dec1)
        if self.dropout > 0:
            dec1 = Dropout(self.dropout)(dec1)

        dec0 = self.conv_act(dec1, out_filters=int(self.shape[0] // 8))
        conv_f = Conv2D(1, 1, activation="relu", padding="same")(dec0)
        flatten_0 = Flatten()(conv_f)
        dense_0 = Dense(
            self.shape[0] / 2,
            kernel_regularizer=l2(1e-6),
            activity_regularizer=l2(1e-6),
        )(flatten_0)
        dropout_0 = Dropout(0.5)(dense_0)
        act_0 = Activation("relu")(dropout_0)
        output = Dense(2,activation="sigmoid")(act_0)
        model = Model(inputs=resnet50.get_input_at(0), outputs=output)

        model.compile(
            loss="binary_crossentropy",
            optimizer=Adam(lr=self.lr, amsgrad=self.amsgrad),
            metrics=["accuracy", recall, f1],
        )

        model.summary()

        return model

    def train(self, X_train, Y_train, n_train,  X_val, Y_val, n_val):
        """ Train the model

        Returns:
            History (History_Keras): History of the training
        """

        e_stop_patience = 10
        min_delta = 0
        e_stop = EarlyStopping(
            monitor="val_loss", min_delta=min_delta, patience=e_stop_patience, verbose=1, mode="auto", restore_best_weights=True
        )

        red_lr_patience = 10
        red_lr = ReduceLROnPlateau(
            monitor="loss", factor=0.5, patience = red_lr_patience, verbose=1, mode="auto"
        )

        weights_filename = "best_resnet_unet.h5"

        save_best_model = ModelCheckpoint(
            weights_filename,
            save_best_only=True,
            monitor="val_loss",
            mode="auto",
            verbose=1,
        )

        cbs = [save_best_model, red_lr, e_stop]

        history = self.model.fit_generator(
            create_minibatch(
                X_train,
                Y_train,
                n_train,
                self.WINDOW_SIZE,
                self.BATCH_SIZE,
                self.PATCH_SIZE,
                self.WIDTH
            ),
            steps_per_epoch=self.STEPS_PER_EPOCH,
            epochs=self.EPOCHS,
            use_multiprocessing=False,
            workers=1,
            callbacks=cbs,
            verbose=1,
            validation_data=create_minibatch(
                X_val, Y_val, n_val, self.WINDOW_SIZE, self.BATCH_SIZE, self.PATCH_SIZE, self.WIDTH
            ),
            validation_steps=self.STEPS_PER_EPOCH,
        )

        return history

    def classify_patches(self, X):
        """Classify Image as either road or not.
        Args:
            X (image): part of the image to classify
        Returns:
            Predictions : Predictions for each patch
        """
        img_patches = create_patches(X, 16, 16, padding=24)
        predictions = self.model.predict(img_patches)
        predictions = (predictions[:, 0] < predictions[:, 1]) * 1
        return predictions.reshape(X.shape[0], -1)