__all__ = [
    "DataSample",
    "DataSet",
    "forgiving_true",
    "load_config",
    "log",
    "time_stamp",
]


from astropy.io import fits
import bson.json_util as bju
from copy import deepcopy
import datetime
import gzip
import io
import numpy as np
import pandas as pd
import pathlib
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tqdm import tqdm
from typing import Union
import yaml


braai = tf.keras.models.load_model(
    pathlib.Path(__file__).parent.parent / "models/braai/braai_d6_m9.h5"
)


def load_config(config_path: Union[str, pathlib.Path]):
    """
    Load config and secrets
    """
    with open(config_path) as config_yaml:
        config = yaml.load(config_yaml, Loader=yaml.FullLoader)

    return config


def time_stamp():
    """
    :return: UTC time as a formatted string
    """
    return datetime.datetime.utcnow().strftime("%Y%m%d_%H:%M:%S")


def log(message: str):
    print(f"{time_stamp()}: {message}")


def forgiving_true(expression):
    return True if expression in ("t", "True", "true", "1", 1, True) else False


class DataSample:
    def __init__(self, alert, label=None, **kwargs):
        self.kwargs = kwargs

        self.label = label

        _a = deepcopy(alert)

        triplet_normalize = kwargs.get("triplet_normalize", True)
        to_tpu = kwargs.get("to_tpu", False)
        feature_names = kwargs.get("feature_names", ("",))
        feature_norms = kwargs.get("feature_norms", None)

        self.triplet = self.make_triplet(_a, normalize=triplet_normalize, to_tpu=to_tpu)
        self.features = self.make_features(
            _a, feature_names=feature_names, norms=feature_norms
        )

        self.data = {
            "triplet": self.triplet,
            "features": self.features,
        }

    @staticmethod
    def make_triplet(alert, normalize: bool = True, to_tpu: bool = False):
        """
        Feed in alert packet
        """
        cutout_dict = dict()

        for cutout in ("science", "template", "difference"):
            cutout_data = bju.loads(
                bju.dumps([alert[f"cutout{cutout.capitalize()}"]["stampData"]])
            )[0]

            # unzip
            with gzip.open(io.BytesIO(cutout_data), "rb") as f:
                with fits.open(io.BytesIO(f.read())) as hdu:
                    data = hdu[0].data
                    # replace nans with zeros
                    cutout_dict[cutout] = np.nan_to_num(data)
                    # normalize
                    if normalize:
                        cutout_dict[cutout] /= np.linalg.norm(cutout_dict[cutout])

            # pad to 63x63 if smaller
            shape = cutout_dict[cutout].shape
            if shape != (63, 63):
                cutout_dict[cutout] = np.pad(
                    cutout_dict[cutout],
                    [(0, 63 - shape[0]), (0, 63 - shape[1])],
                    mode="constant",
                    constant_values=1e-9,
                )

        triplet = np.zeros((63, 63, 3))
        triplet[:, :, 0] = cutout_dict["science"]
        triplet[:, :, 1] = cutout_dict["template"]
        triplet[:, :, 2] = cutout_dict["difference"]

        if to_tpu:
            # Edge TPUs require additional processing
            triplet = np.rint(triplet * 128 + 128).astype(np.uint8).flatten()

        return triplet

    def make_features(self, alert, feature_names=None, norms=None):
        features = []
        for feature_name in feature_names:
            feature = alert["candidate"].get(feature_name)
            if feature is None and feature_name != "drb":
                print(alert["objectId"], alert["candid"], feature_name)
            if feature is None and feature_name == "drb":
                feature = float(
                    braai.predict(x=np.expand_dims(self.triplet, axis=0))[0]
                )
            if norms is not None:
                feature /= norms.get(feature, 1)
            features.append(feature)

        features = np.array(features)
        return features


class DataSet:
    # collection of DataSamples -> tf.data
    def __init__(
        self,
        tag: str,
        path_labels: str,
        path_labels_add: str = None,
        path_data: str = "./",
        feature_names=(
            "drb",
            "diffmaglim",
            "ra",
            "dec",
            "magpsf",
            "sigmapsf",
            "chipsf",
            "fwhm",
            "sky",
            "chinr",
            "sharpnr",
            "sgscore1",
            "distpsnr1",
            "sgscore2",
            "distpsnr2",
            "sgscore3",
            "distpsnr3",
            "ndethist",
            "ncovhist",
            "scorr",
            "nmtchps",
            "clrcoeff",
            "clrcounc",
            "neargaia",
            "neargaiabright",
        ),
        verbose: bool = False,
        **kwargs,
    ):
        self.verbose = verbose
        self.tag = tag

        self.path_data = pathlib.Path(path_data)
        self.path_labels = pathlib.Path(path_labels)
        self.path_labels_add = (
            pathlib.Path(path_labels_add) if path_labels_add is not None else None
        )

        self.labels = pd.read_csv(self.path_labels)
        self.labels_add = (
            pd.read_csv(self.path_labels_add) if path_labels_add is not None else None
        )

        self.target = None
        self.target_label = None

        self.feature_names = feature_names
        self.feature_norms = {feature: 1.0 for feature in self.feature_names}

        self.features, self.triplets = None, None

        if self.verbose:
            print("Labels:")
            print(self.labels)
            if path_labels_add is not None:
                print("Additional labels:")
                print(self.labels_add)

    def load_data(self, labels, **kwargs):
        """Parse alert json files into features and image triplets

        :param labels:
        :param kwargs:
        :return:
        """
        features = []
        triplets = []

        entries = (
            tqdm(labels.itertuples(), total=len(labels))
            if self.verbose
            else labels.itertuples()
        )
        for ei, entry in enumerate(entries):
            with open(self.path_data / f"{entry.candid}.json", "r") as f:
                alert = bju.loads(f.read())
                data_sample = DataSample(
                    alert=alert,
                    label=entry.label,
                    feature_names=self.feature_names,
                    feature_norms=self.feature_norms,
                )
                features.append(data_sample.data.get("features"))
                triplets.append(data_sample.data.get("triplet"))

        features = np.array(features)
        triplets = np.array(triplets)

        return features, triplets

    def make(
        self,
        target_label: str = "h",
        balance=None,
        weight_per_class: bool = True,
        test_size: float = 0.1,
        val_size: float = 0.1,
        random_state: int = 42,
        path_norms=None,
        path_features=None,
        path_triplets=None,
        batch_size: int = 256,
        shuffle_buffer_size: int = 256,
        epochs: int = 300,
        **kwargs,
    ):
        """Make datasets for target_label
        :param target_label: corresponds to training.classes.<label> in config
        :param threshold: our labels are floats [0, 0.25, 0.5, 0.75, 1]
        :param balance: balance ratio for the prevalent class. if null - use all available data
        :param weight_per_class:
        :param scale_features: min_max | median_std
        :param test_size:
        :param val_size:
        :param random_state: set this for reproducibility
        :param feature_stats: feature_stats to use to standardize features.
                              if None, stats are computed from the data, taking balance into account
        :param batch_size
        :param shuffle_buffer_size
        :param epochs
        :return:
        """
        mask_positive = self.labels.label == target_label
        self.target = np.expand_dims(mask_positive, axis=1)
        self.target_label = target_label

        neg, pos = np.bincount(self.target.flatten())
        total = neg + pos
        if self.verbose:
            print(
                f"Examples:\n  Total: {total}\n  Positive: {pos} ({100 * pos / total:.2f}% of total)\n"
            )

        index_positive = self.labels.loc[mask_positive].index
        mask_negative = ~mask_positive
        index_negative = self.labels.loc[mask_negative].index

        # balance positive and negative examples?
        index_dropped = None
        if balance:
            underrepresented = min(np.sum(mask_positive), np.sum(mask_negative))
            overrepresented = max(np.sum(mask_positive), np.sum(mask_negative))
            sample_size = int(min(overrepresented, underrepresented * balance))
            if neg > pos:
                index_negative = (
                    self.labels.loc[mask_negative]
                    .sample(n=sample_size, random_state=1)
                    .index
                )
                index_dropped = self.labels.loc[
                    list(
                        set(self.labels.loc[mask_negative].index) - set(index_negative)
                    )
                ].index
            else:
                index_positive = (
                    self.labels.loc[mask_positive]
                    .sample(n=sample_size, random_state=1)
                    .index
                )
                index_dropped = self.labels.loc[
                    list(
                        set(self.labels.loc[mask_positive].index) - set(index_positive)
                    )
                ].index

        ds_indexes = index_positive.to_list() + index_negative.to_list()

        # Train/validation/test split (we will use an 81% / 9% / 10% data split by default):

        train_indexes, test_indexes = train_test_split(
            ds_indexes, shuffle=True, test_size=test_size, random_state=random_state
        )
        train_indexes, val_indexes = train_test_split(
            train_indexes, shuffle=True, test_size=val_size, random_state=random_state
        )

        # load and normalize data
        if path_norms is not None:
            with open(path_norms) as f:
                self.feature_norms = yaml.load(f, Loader=yaml.FullLoader)

        if path_features is not None and path_triplets is not None:
            self.features = np.load(path_features)
            self.triplets = np.load(path_triplets)
        else:
            self.features, self.triplets = self.load_data(self.labels)
        # self.features = np.zeros((total, len(self.feature_names), 1))
        # self.triplets = np.zeros((total, 63, 63, 3))

        # extend dataset with additional data that is guaranteed to be used in training
        # helpful in active learning
        if self.labels_add is not None:
            features_add, triplets_add = self.load_data(self.labels_add)
            total_add = features_add.shape[0]
            train_indexes_add, test_indexes_add = train_test_split(
                list(range(total, total + total_add)),
                shuffle=True,
                test_size=test_size,
                random_state=random_state,
            )
            train_indexes_add, val_indexes_add = train_test_split(
                train_indexes_add,
                shuffle=True,
                test_size=val_size,
                random_state=random_state,
            )
            self.features = np.concatenate((self.features, features_add), axis=0)
            self.triplets = np.concatenate((self.triplets, triplets_add), axis=0)
            train_indexes += train_indexes_add
            val_indexes += val_indexes_add
            test_indexes += test_indexes_add

            mask_positive_add = self.labels_add.label == target_label
            target_add = np.expand_dims(mask_positive_add, axis=1)
            self.target = np.concatenate((self.target, target_add), axis=0)

        # make tf.data.Dataset's:
        train_dataset = tf.data.Dataset.from_tensor_slices(
            (
                {
                    "features": self.features[train_indexes],
                    "triplets": self.triplets[train_indexes],
                },
                # self.target[train_indexes],
                np.array(self.target[train_indexes], dtype=np.float64),
            )
        )
        val_dataset = tf.data.Dataset.from_tensor_slices(
            (
                {
                    "features": self.features[val_indexes],
                    "triplets": self.triplets[val_indexes],
                },
                # self.target[val_indexes],
                np.array(self.target[val_indexes], dtype=np.float64),
            )
        )
        test_dataset = tf.data.Dataset.from_tensor_slices(
            (
                {
                    "features": self.features[test_indexes],
                    "triplets": self.triplets[test_indexes],
                },
                # self.target[test_indexes],
                np.array(self.target[test_indexes], dtype=np.float64),
            )
        )
        dropped_samples = (
            tf.data.Dataset.from_tensor_slices(
                (
                    {
                        "features": self.features[index_dropped],
                        "triplets": self.triplets[index_dropped],
                    },
                    # self.target[index_dropped],
                    np.array(self.target[index_dropped], dtype=np.float64),
                )
            )
            if balance
            else None
        )

        # Shuffle and batch the datasets:

        train_dataset = (
            train_dataset.shuffle(shuffle_buffer_size).batch(batch_size).repeat(epochs)
        )
        val_dataset = val_dataset.batch(batch_size).repeat(epochs)
        test_dataset = test_dataset.batch(batch_size)

        dropped_samples = dropped_samples.batch(batch_size) if balance else None

        datasets = {
            "train": train_dataset,
            "val": val_dataset,
            "test": test_dataset,
            "dropped_samples": dropped_samples,
        }

        indexes = {
            "train": np.array(train_indexes),
            "val": np.array(val_indexes),
            "test": np.array(test_indexes),
            "dropped_samples": np.array(index_dropped.to_list())
            if index_dropped is not None
            else None,
        }

        # How many steps per epoch?

        steps_per_epoch_train = len(train_indexes) // batch_size - 1
        steps_per_epoch_val = len(val_indexes) // batch_size - 1
        steps_per_epoch_test = len(test_indexes) // batch_size - 1

        steps_per_epoch = {
            "train": steps_per_epoch_train,
            "val": steps_per_epoch_val,
            "test": steps_per_epoch_test,
        }
        if self.verbose:
            print(f"Steps per epoch: {steps_per_epoch}")

        # Weight training data depending on the number of samples?
        # Very useful for imbalanced classification, especially when in the cases with a small number of examples.

        if weight_per_class:
            # weight data class depending on number of examples?
            # num_training_examples_per_class = np.array([len(target) - np.sum(target), np.sum(target)])
            num_training_examples_per_class = np.array(
                [len(index_negative), len(index_positive)]
            )

            assert (
                0 not in num_training_examples_per_class
            ), "found class without any examples!"

            # fewer examples -- larger weight
            weights = (1 / num_training_examples_per_class) / np.linalg.norm(
                (1 / num_training_examples_per_class)
            )
            normalized_weight = weights / np.max(weights)

            class_weight = {i: w for i, w in enumerate(normalized_weight)}

        else:
            # working with binary classifiers only
            class_weight = {i: 1 for i in range(2)}

        return datasets, indexes, steps_per_epoch, class_weight
