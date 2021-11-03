__all__ = [
    "DataSample",
    "DataSet",
    "forgiving_true",
    "load_config",
    "log",
    "time_stamp",
    "threshold",
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
from typing import Optional, Sequence, Union
import yaml


braai = tf.keras.models.load_model(
    pathlib.Path(__file__).parent.parent / "models/braai/braai_d6_m9.h5"
)


def load_config(config_path: Union[str, pathlib.Path]) -> dict:
    """Load config and secrets"""
    with open(config_path) as config_yaml:
        config = yaml.load(config_yaml, Loader=yaml.FullLoader)

    return config


def time_stamp() -> str:
    """UTC time stamps for logging
    :return: UTC time as a formatted string
    """
    return datetime.datetime.utcnow().strftime("%Y%m%d_%H:%M:%S")


def log(message: str):
    """Log message"""
    print(f"{time_stamp()}: {message}")


def forgiving_true(expression: Union[str, int, bool]):
    """Forgivingly evaluate an expression meaning True"""
    return True if expression in ("t", "True", "true", "1", 1, True) else False


def threshold(a, t: float = 0.5):
    """Convert raw probabilities into 0/1 labels"""
    b = np.zeros_like(a, dtype=np.int64)
    b[np.array(a) > t] = 1
    return b


class DataSample:
    def __init__(self, alert: dict, label: Optional[str] = None, **kwargs):
        """Data structure to store alert features and image cutouts"""
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
    def make_triplet(
        alert: dict,
        normalize: bool = True,
        nan_to_median: bool = False,
        to_tpu: bool = False,
    ) -> np.array:
        """Standardize alert cutout images"""
        cutout_dict = dict()

        for cutout in ("science", "template", "difference"):
            cutout_data = bju.loads(
                bju.dumps([alert[f"cutout{cutout.capitalize()}"]["stampData"]])
            )[0]

            # unzip
            with gzip.open(io.BytesIO(cutout_data), "rb") as f:
                with fits.open(io.BytesIO(f.read()), ignore_missing_simple=True) as hdu:
                    data = hdu[0].data
                    # replace nans with zeros by default or with median if requested
                    if nan_to_median:
                        img = np.array(data)
                        # replace dubiously large values
                        xl = np.greater(np.abs(img), 1e20, where=~np.isnan(img))
                        if img[xl].any():
                            img[xl] = np.nan
                        if np.isnan(img).any():
                            median = float(np.nanmean(img.flatten()))
                            img = np.nan_to_num(img, nan=median)
                        cutout_dict[cutout] = img
                    else:
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
                    constant_values=1e-9
                    if not nan_to_median
                    else float(np.nanmean(cutout_dict[cutout].flatten())),
                )

        triplet = np.zeros((63, 63, 3))
        triplet[:, :, 0] = cutout_dict["science"]
        triplet[:, :, 1] = cutout_dict["template"]
        triplet[:, :, 2] = cutout_dict["difference"]

        if to_tpu:
            # Edge TPUs require additional processing
            triplet = np.rint(triplet * 128 + 128).astype(np.uint8).flatten()

        return triplet

    def make_features(
        self,
        alert: dict,
        feature_names: Sequence[str] = tuple(),
        norms: Optional[dict] = None,
    ) -> np.array:
        """Extract and standardize alert features"""
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
    def __init__(
        self,
        tag: str,
        features: dict,
        path_labels: str,
        path_data: str = "./",
        verbose: bool = False,
        **kwargs,
    ):
        """Data structure to store a collection of DataSamples as tf.data.Dataset's"""
        self.verbose = verbose
        self.tag = tag

        self.path_data = pathlib.Path(path_data)
        self.path_labels = pathlib.Path(path_labels)

        self.labels = pd.read_csv(self.path_labels)

        self.target = None
        self.target_label = None

        self.feature_names = list(features.keys())
        self.feature_norms = features

        self.features, self.triplets, self.meta = None, None, None

        if self.verbose:
            print("Labels:")
            print(self.labels)

    def load_data(self, labels: pd.DataFrame, **kwargs):
        """Load from disk and parse alert json files into features and image triplets

        :param labels:
        :param kwargs:
        :return:
        """
        features = []
        triplets = []
        meta = []

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
                meta.append(
                    {
                        "oid": alert["objectId"],
                        "candid": alert["candid"],
                        "ra": alert["candidate"]["ra"],
                        "dec": alert["candidate"]["dec"],
                    }
                )

        features = np.array(features)
        triplets = np.array(triplets)
        meta = np.array(meta)

        return features, triplets, meta

    def make(
        self,
        target_label: str = "h",
        balance: Optional[Union[int, float]] = None,
        weight_per_class: bool = True,
        test_size: float = 0.1,
        val_size: float = 0.1,
        random_state: int = 42,
        batch_size: int = 256,
        shuffle_buffer_size: int = 256,
        epochs: int = 300,
        **kwargs,
    ):
        """Make datasets for target_label
        :param target_label: corresponds to training.classes.<label> in config
        :param balance: balance ratio for the prevalent class. if null - use all available data
        :param weight_per_class:
        :param test_size:
        :param val_size:
        :param random_state: set this for reproducibility
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
            elif pos < neg:
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
        path_features: Optional[Union[str, pathlib.Path]] = kwargs.get("path_features")
        path_triplets: Optional[Union[str, pathlib.Path]] = kwargs.get("path_triplets")
        if path_features is not None and path_triplets is not None:
            self.features = np.load(path_features)
            self.triplets = np.load(path_triplets)
        else:
            self.features, self.triplets, self.meta = self.load_data(self.labels)
        # self.features = np.zeros((total, len(self.feature_names), 1))
        # self.triplets = np.zeros((total, 63, 63, 3))

        # make tf.data.Dataset's:
        train_dataset = tf.data.Dataset.from_tensor_slices(
            (
                {
                    "features": self.features[train_indexes],
                    "triplets": self.triplets[train_indexes],
                },
                np.array(self.target[train_indexes], dtype=np.float64),
            )
        )
        train_meta = self.meta[train_indexes] if self.meta is not None else None
        val_dataset = tf.data.Dataset.from_tensor_slices(
            (
                {
                    "features": self.features[val_indexes],
                    "triplets": self.triplets[val_indexes],
                },
                np.array(self.target[val_indexes], dtype=np.float64),
            )
        )
        val_meta = self.meta[val_indexes] if self.meta is not None else None
        test_dataset = tf.data.Dataset.from_tensor_slices(
            (
                {
                    "features": self.features[test_indexes],
                    "triplets": self.triplets[test_indexes],
                },
                np.array(self.target[test_indexes], dtype=np.float64),
            )
        )
        test_meta = self.meta[test_indexes] if self.meta is not None else None
        dropped_samples = (
            tf.data.Dataset.from_tensor_slices(
                (
                    {
                        "features": self.features[index_dropped],
                        "triplets": self.triplets[index_dropped],
                    },
                    np.array(self.target[index_dropped], dtype=np.float64),
                )
            )
            if balance
            else None
        )
        dropped_meta = (
            self.meta[train_indexes] if balance and (self.meta is not None) else None
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

        metadata = {
            "train": train_meta,
            "val": val_meta,
            "test": test_meta,
            "dropped_samples": dropped_meta,
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

        return datasets, metadata, indexes, steps_per_epoch, class_weight
