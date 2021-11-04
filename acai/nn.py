"""
    A bowl of Deep Neural Networks for acai

    Author: Dr Dmitry A. Duev
    2020 - 2021
"""

__all__ = [
    "DNN",
]

import datetime
import os
import tensorflow as tf
from typing import Tuple


from .models import AbstractClassifier


# turn off the annoying tensorflow messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class DNN(AbstractClassifier):
    """Hybrid MLP+CNN model"""

    def setup(
        self,
        features_shape=(25,),
        triplet_shape=(63, 63, 3),
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
            **kwargs,
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
    def build_model(**kwargs):
        """Build a tunable model that can consume hyper-parameter suggestions"""

        def dense_block(
            x,
            units: int = 64,
            # scale_factor: Union[int, float] = 0.5,
            activation: str = "relu",
            dropout_rate: float = 0.25,
        ):
            x = tf.keras.layers.Dense(units, activation=activation)(x)
            x = tf.keras.layers.Dropout(dropout_rate)(x)
            # x = tf.keras.layers.Dense(int(front_units * scale_factor), activation=activation)(x)
            return x

        def conv_block(
            x,
            conv_layer_type: str = "SeparableConv2D",
            pool_layer_type: str = "MaxPooling2D",
            filters: int = 16,
            filter_size: Tuple[int, int] = (3, 3),
            activation: str = "relu",
            pool_size: Tuple[int, int] = (2, 2),
            dropout_rate: float = 0.25,
        ):
            conv_layer = getattr(tf.keras.layers, conv_layer_type)
            pool_layer = getattr(tf.keras.layers, pool_layer_type)
            x = conv_layer(filters, filter_size, activation=activation)(x)
            x = conv_layer(filters, filter_size, activation=activation)(x)
            x = pool_layer(pool_size=pool_size)(x)
            x = tf.keras.layers.Dropout(dropout_rate)(x)
            return x

        num_dense_blocks = kwargs.get("dense_blocks", 2)
        num_conv_blocks = kwargs.get("conv_blocks", 2)
        num_head_blocks = kwargs.get("head_blocks", 1)

        if num_dense_blocks == 0 and num_conv_blocks == 0:
            raise ValueError("model must have at least one branch")

        features_input = tf.keras.Input(
            shape=kwargs.get("features_shape", (25,)), name="features"
        )
        triplet_input = tf.keras.Input(
            shape=kwargs.get("triplet_shape", (63, 63, 3)), name="triplets"
        )

        # dense branch to digest features
        if num_dense_blocks > 0:
            x_dense = features_input
            dense_block_scale_factor = kwargs.get("dense_block_scale_factor", 0.5)
            for i in range(num_dense_blocks):
                x_dense = dense_block(
                    x=x_dense,
                    units=kwargs.get("dense_block_units", 64)
                    * (dense_block_scale_factor ** i),
                    activation=kwargs.get("dense_activation", "relu"),
                    dropout_rate=kwargs.get("dense_dropout_rate", 0.25),
                )

        # CNN branch to digest image cutouts
        if num_conv_blocks > 0:
            x_conv = triplet_input
            conv_block_scale_factor = kwargs.get("conv_block_scale_factor", 2)
            for i in range(num_conv_blocks):
                x_conv = conv_block(
                    x=x_conv,
                    conv_layer_type=kwargs.get(
                        "conv_conv_layer_type", "SeparableConv2D"
                    ),
                    pool_layer_type=kwargs.get("conv_pool_layer_type", "MaxPooling2D"),
                    filters=kwargs.get("conv_block_filters", 16)
                    * (conv_block_scale_factor ** i),
                    filter_size=kwargs.get("conv_block_filter_size", (3, 3)),
                    activation=kwargs.get("conv_activation", "relu"),
                    pool_size=kwargs.get("conv_block_pool_size", (2, 2)),
                    dropout_rate=kwargs.get("conv_dropout_rate", 0.25),
                )

            x_conv = tf.keras.layers.GlobalAveragePooling2D()(x_conv)

        # concatenate
        if num_dense_blocks and num_conv_blocks:
            x = tf.keras.layers.concatenate([x_dense, x_conv])
        elif num_dense_blocks:
            x = x_dense
        elif num_conv_blocks:
            x = x_conv

        # dense head
        head_block_scale_factor = kwargs.get("head_block_scale_factor", 1)
        for i in range(num_head_blocks):
            x = dense_block(
                x=x,
                units=kwargs.get("head_block_units", 16)
                * (head_block_scale_factor ** i),
                activation=kwargs.get("head_activation", "relu"),
                dropout_rate=kwargs.get("head_dropout_rate", 0),
            )

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
        """Execute full training cycle"""
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
        """Run inference using classifier.model

        :param x:
        :param kwargs:
        :return:

        Example:
        - Get ZTF alerts and convert them into dict. For example, using Kowalski
        >>> from penquins import Kowalski
        >>> kowalski = Kowalski(...)
        >>> query = {
        >>>     "query_type": "find",
        >>>     "query": {
        >>>         "catalog": "ZTF_alerts",
        >>>         "filter": {
        >>>             "objectId": "ZTF20abyzoof",
        >>>         }
        >>>     }
        >>> }
        >>> alerts = kowalski.query(query=query)["data"]
        - Extract features and image cutout triplets from the grabbed alerts
        >>> feature_names = (
        >>>     "drb",
        >>>     "diffmaglim",
        >>>     "ra",
        >>>     "dec",
        >>>     "magpsf",
        >>>     "sigmapsf",
        >>>     "chipsf",
        >>>     "fwhm",
        >>>     "sky",
        >>>     "chinr",
        >>>     "sharpnr",
        >>>     "sgscore1",
        >>>     "distpsnr1",
        >>>     "sgscore2",
        >>>     "distpsnr2",
        >>>     "sgscore3",
        >>>     "distpsnr3",
        >>>     "ndethist",
        >>>     "ncovhist",
        >>>     "scorr",
        >>>     "nmtchps",
        >>>     "clrcoeff",
        >>>     "clrcounc",
        >>>     "neargaia",
        >>>     "neargaiabright",
        >>> )
        >>>
        >>> import acai
        >>> samples = [
        >>>     acai.DataSample(
        >>>         alert=alert,
        >>>         feature_names=feature_names,
        >>>     )
        >>>     for alert in alerts
        >>> ]
        >>>
        >>> import numpy as np
        >>> features = []
        >>> triplets = []
        >>> for sample in samples:
        >>>     features.append(np.expand_dims(sample.data["features"], axis=[-1]))
        >>>     triplets.append(sample.data["triplet"])
        >>> features = np.array(features)
        >>> triplets = np.array(triplets)
        - Load a pre-trained model and run inference
        >>> classifier = acai.DNN(name="h")
        >>> classifier.load("models/acai_h.d1_dnn_20201130.h5")
        >>> h = classifier.predict([features, triplets])
        """
        return self.model.predict(x, **kwargs)

    def load(self, path_model, **kwargs):
        """Load pre-trained model"""
        self.model = tf.keras.models.load_model(path_model, **kwargs)

    def save(self, output_path="./", output_format="hdf5", tag=None):
        """Save model"""
        import pathlib

        if output_format not in ("tf", "hdf5"):
            raise ValueError("unknown output format")

        output_name = self.name if not tag else f"{self.name}.{tag}"

        path = pathlib.Path(output_path)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)

        if output_format == "tf":
            self.model.save_weights(path / tag / output_name, save_format="tf")
        elif output_format == "hdf5":
            self.model.save(path / tag / f"{output_name}.h5")
