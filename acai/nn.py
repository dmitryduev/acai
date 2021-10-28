"""
    A bowl of Deep Neural Networks for acai

    Author: Dr Dmitry A. Duev
    November 2020
"""

__all__ = [
    "DNN",
]

import datetime
import os

import tensorflow as tf

from .models import AbstractClassifier


class DNN(AbstractClassifier):
    """
    Baseline model with a statically-defined graph
    """

    def setup(
        self,
        features_shape=(25,),
        triplet_shape=(63, 63, 3),
        dense_branch=True,
        conv_branch=True,
        loss="binary_crossentropy",
        optimizer="adam",
        callbacks=("early_stopping", "tensorboard"),
        tag=None,
        logdir="logs",
        **kwargs,
    ):

        tf.keras.backend.clear_session()

        self.model = self.build_model(
            features_shape=features_shape,
            triplet_shape=triplet_shape,
            dense_branch=dense_branch,
            conv_branch=conv_branch,
        )

        self.meta["loss"] = loss
        if optimizer == "adam":
            lr = kwargs.get("lr", 3e-4)
            beta_1 = kwargs.get("beta_1", 0.9)
            beta_2 = kwargs.get("beta_2", 0.999)
            epsilon = kwargs.get("epsilon", 1e-7)  # None?
            decay = kwargs.get("decay", 0.0)
            amsgrad = kwargs.get("amsgrad", 3e-4)
            self.meta["optimizer"] = tf.keras.optimizers.Adam(
                lr=lr,
                beta_1=beta_1,
                beta_2=beta_2,
                epsilon=epsilon,
                decay=decay,
                amsgrad=amsgrad,
            )
        elif optimizer == "sgd":
            lr = kwargs.get("lr", 3e-4)
            momentum = kwargs.get("momentum", 0.9)
            decay = kwargs.get("epsilon", 1e-6)
            nesterov = kwargs.get("nesterov", True)
            self.meta["optimizer"] = tf.keras.optimizers.SGD(
                lr=lr, momentum=momentum, decay=decay, nesterov=nesterov
            )
        else:
            print("Could not recognize optimizer, using Adam with default params")
            self.meta["optimizer"] = tf.keras.optimizers.Adam(
                lr=3e-4,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-7,
                decay=0.0,
                amsgrad=False,
            )
        # self.meta['epochs'] = epochs
        # self.meta['patience'] = patience
        # self.meta['weight_per_class'] = weight_per_class

        self.meta["metrics"] = [
            tf.keras.metrics.TruePositives(name="tp"),
            tf.keras.metrics.FalsePositives(name="fp"),
            tf.keras.metrics.TrueNegatives(name="tn"),
            tf.keras.metrics.FalseNegatives(name="fn"),
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ]

        self.meta["callbacks"] = []
        # self.meta['callbacks'] = [TqdmCallback(verbose=1)]
        for callback in set(callbacks):
            if callback == "early_stopping":
                # halt training if no gain in <monitor> metric over <patience> epochs
                monitor = kwargs.get("monitor", "val_loss")
                patience = kwargs.get("patience", 10)
                restore_best_weights = kwargs.get("restore_best_weights", True)
                early_stopping_callback = tf.keras.callbacks.EarlyStopping(
                    monitor=monitor,
                    patience=patience,
                    restore_best_weights=restore_best_weights,
                )
                self.meta["callbacks"].append(early_stopping_callback)

            elif callback == "tensorboard":
                # logs for TensorBoard:
                time_tag = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                if tag:
                    log_tag = f'{self.name.replace(" ", "_")}-{tag}-{time_tag}'
                else:
                    log_tag = f'{self.name.replace(" ", "_")}-{time_tag}'
                logdir_tag = os.path.join("logs", log_tag)
                tensorboard_callback = tf.keras.callbacks.TensorBoard(
                    os.path.join(logdir_tag, log_tag), histogram_freq=1
                )
                self.meta["callbacks"].append(tensorboard_callback)

        self.model.compile(
            optimizer=self.meta["optimizer"],
            loss=self.meta["loss"],
            metrics=self.meta["metrics"],
        )

    @staticmethod
    def build_model(
        dense_branch: bool = True,
        conv_branch: bool = True,
        **kwargs,
    ):
        if (not dense_branch) and (not conv_branch):
            raise ValueError("model must have at least one branch")

        features_input = tf.keras.Input(
            shape=kwargs.get("features_shape", (40,)), name="features"
        )
        triplet_input = tf.keras.Input(
            shape=kwargs.get("triplet_shape", (26, 26, 1)), name="triplets"
        )

        # dense branch to digest features
        if dense_branch:
            x_dense = tf.keras.layers.Dense(64, activation="relu", name="dense_fc_1")(
                features_input
            )
            x_dense = tf.keras.layers.Dropout(0.25)(x_dense)
            x_dense = tf.keras.layers.Dense(32, activation="relu", name="dense_fc_2")(
                x_dense
            )

        # CNN branch to digest image cutouts
        if conv_branch:
            x_conv = tf.keras.layers.SeparableConv2D(
                16, (3, 3), activation="relu", name="conv_conv_1"
            )(triplet_input)
            x_conv = tf.keras.layers.SeparableConv2D(
                16, (3, 3), activation="relu", name="conv_conv_2"
            )(x_conv)
            x_conv = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x_conv)
            x_conv = tf.keras.layers.Dropout(0.25)(x_conv)

            x_conv = tf.keras.layers.SeparableConv2D(
                32, (3, 3), activation="relu", name="conv_conv_3"
            )(x_conv)
            x_conv = tf.keras.layers.SeparableConv2D(
                32, (3, 3), activation="relu", name="conv_conv_4"
            )(x_conv)
            x_conv = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x_conv)
            x_conv = tf.keras.layers.Dropout(0.25)(x_conv)

            x_conv = tf.keras.layers.GlobalAveragePooling2D()(x_conv)

        # concatenate
        if dense_branch and conv_branch:
            x = tf.keras.layers.concatenate([x_dense, x_conv])
        elif dense_branch:
            x = x_dense
        elif conv_branch:
            x = x_conv

        # one more dense layer?
        x = tf.keras.layers.Dense(16, activation="relu", name="fc_1")(x)

        # Logistic regression to output the final score
        x = tf.keras.layers.Dense(1, activation="sigmoid", name="score")(x)

        m = tf.keras.Model(inputs=[features_input, triplet_input], outputs=x)

        return m

    def train(
        self,
        train_dataset,
        val_dataset,
        steps_per_epoch_train,
        steps_per_epoch_val,
        epochs=300,
        class_weight=None,
        verbose=0,
    ):

        if not class_weight:
            # all our problems here are binary classification ones:
            class_weight = {i: 1 for i in range(2)}

        self.meta["history"] = self.model.fit(
            train_dataset,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch_train,
            validation_data=val_dataset,
            validation_steps=steps_per_epoch_val,
            class_weight=class_weight,
            callbacks=self.meta["callbacks"],
            verbose=verbose,
        )

    def evaluate(self, test_dataset, **kwargs):
        return self.model.evaluate(test_dataset, **kwargs)

    def predict(self, x, **kwargs):
        return self.model.predict(x, **kwargs)

    def load(self, path_model, **kwargs):
        self.model = tf.keras.models.load_model(path_model, **kwargs)

    def save(self, output_path="./", output_format="hdf5", tag=None):

        if output_format not in ("SavedModel", "hdf5"):
            raise ValueError("unknown output format")

        output_name = self.name if not tag else f"{self.name}.{tag}"

        if (output_path != "./") and (not os.path.exists(output_path)):
            os.makedirs(output_path)

        if output_format == "SavedModel":
            self.model.save(os.path.join(output_path, output_name))
        elif output_format == "hdf5":
            self.model.save(os.path.join(output_path, f"{output_name}.h5"))
