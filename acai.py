#!/usr/bin/env python
import matplotlib.pyplot as plt
from astropy.visualization import (
    AsymmetricPercentileInterval,
    ImageNormalize,
    LinearStretch,
    LogStretch,
)
from contextlib import contextmanager
import datetime
from deepdiff import DeepDiff
import fire
import json
import io
import numpy as np
import os
import pandas as pd
import pathlib
from penquins import Kowalski
from pprint import pprint
import random
import questionary
import subprocess
import sys
from typing import Optional, Sequence
import tqdm
import wandb
from wandb.keras import WandbCallback
import yaml

from acai.utils import DataSample, DataSet, forgiving_true, load_config, log, threshold


# turn off the annoying tensorflow messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


@contextmanager
def status(message):
    """Display a fancy status message
    Borrowed from https://github.com/cesium-ml/baselayer/

    :param message: message to print
    :return:
    """
    print(f"[·] {message}", end="")
    sys.stdout.flush()
    try:
        yield
    except Exception:
        print(f"\r[✗] {message}")
        raise
    else:
        print(f"\r[✓] {message}")


def check_configs(config_wildcards: Sequence = ("config.*yaml",)):
    """
    - Check if config files exist
    - Offer to use the config files that match the wildcards
    - For config.yaml, check its contents against the defaults to make sure nothing is missing/wrong

    :param config_wildcards:
    :return:
    """
    path = pathlib.Path(__file__).parent.absolute()

    for config_wildcard in config_wildcards:
        config = config_wildcard.replace("*", "")
        # use config defaults if configs do not exist?
        if not (path / config).exists():
            answer = questionary.select(
                f"{config} does not exist, do you want to use one of the following"
                " (not recommended without inspection)?",
                choices=[p.name for p in path.glob(config_wildcard)],
            ).ask()
            subprocess.run(["cp", f"{path / answer}", f"{path / config}"])

        # check contents of config.yaml WRT config.defaults.yaml
        if config == "config.yaml":
            with open(path / config.replace(".yaml", ".defaults.yaml")) as config_yaml:
                config_defaults = yaml.load(config_yaml, Loader=yaml.FullLoader)
            with open(path / config) as config_yaml:
                config_wildcard = yaml.load(config_yaml, Loader=yaml.FullLoader)
            deep_diff = DeepDiff(config_defaults, config_wildcard, ignore_order=True)
            difference = {
                k: v for k, v in deep_diff.items() if k in ("dictionary_item_removed",)
            }
            if len(difference) > 0:
                log("config.yaml structure differs from config.defaults.yaml")
                pprint(difference)
                raise KeyError("Fix config.yaml before proceeding")


class ACAI:
    def __init__(self):
        """CLI commands"""
        # check configuration
        with status("Checking configuration"):
            check_configs(config_wildcards=["config.*yaml"])

            self.config = load_config(
                pathlib.Path(__file__).parent.absolute() / "config.yaml"
            )

            # use token specified as env var (if exists)
            kowalski_token_env = os.environ.get("KOWALSKI_TOKEN")
            if kowalski_token_env is not None:
                self.config["kowalski"]["token"] = kowalski_token_env

        # try setting up K connection if token is available
        if self.config["kowalski"]["token"] is not None:
            with status("Setting up Kowalski connection"):
                self.kowalski = Kowalski(
                    token=self.config["kowalski"]["token"],
                    protocol=self.config["kowalski"]["protocol"],
                    host=self.config["kowalski"]["host"],
                    port=self.config["kowalski"]["port"],
                )
        else:
            self.kowalski = None
            # raise ConnectionError("Could not connect to Kowalski.")
            log("Kowalski not available")

    @staticmethod
    def develop():
        """Install developer tools"""
        subprocess.run(["pre-commit", "install"])

    @classmethod
    def lint(cls):
        """Lint sources"""
        try:
            import pre_commit  # noqa: F401
        except ImportError:
            cls.develop()

        try:
            subprocess.run(["pre-commit", "run", "--all-files"], check=True)
        except subprocess.CalledProcessError:
            sys.exit(1)

    @staticmethod
    def fetch_datasets(gcs_path: str = "gs://ztf-acai"):
        """Fetch ACAI datasets from GCP

        :return:
        """
        path_datasets = pathlib.Path(__file__).parent / "data" / "training"
        if not path_datasets.exists():
            path_datasets.mkdir(parents=True, exist_ok=True)

        for extension in ("csv", "npy"):
            command = [
                "gsutil",
                "-m",
                "cp",
                "-n",
                "-r",
                os.path.join(gcs_path, f"*.{extension}"),
                str(path_datasets),
            ]
            p = subprocess.run(command, check=True)
            if p.returncode != 0:
                raise RuntimeError("Failed to fetch ACAI datasets")

    @staticmethod
    def alert_images(path_data: str, candid: int) -> dict:
        """Turn alert image cutouts into matplotlib figures
        to be posted to wandb
        """
        normalizer = AsymmetricPercentileInterval(
            lower_percentile=1, upper_percentile=100
        )

        images = {"science": None, "reference": None, "difference": None}

        path_alert = pathlib.Path(path_data) / f"{candid}.json"
        if not path_alert.exists():
            return images

        with open(path_alert, "r") as f:
            alert = json.load(f)

        triplet = DataSample.make_triplet(
            alert=alert, normalize=False, nan_to_median=True
        )

        for i, img_tag in enumerate(("science", "reference", "difference")):
            image = triplet[:, :, i]

            buff = io.BytesIO()
            plt.close("all")
            fig = plt.figure()
            fig.set_size_inches(2, 2, forward=False)
            ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
            ax.set_axis_off()
            fig.add_axes(ax)

            stretcher = LinearStretch() if img_tag == "difference" else LogStretch()

            image_norm = ImageNormalize(image, stretch=stretcher)(image)
            vmin, vmax = normalizer.get_limits(image_norm)
            ax.imshow(image_norm, cmap="bone", origin="lower", vmin=vmin, vmax=vmax)
            fig.savefig(buff, dpi=42)
            # buff.seek(0)
            # plt.close("all")
            images[img_tag] = fig

        return images

    def train(
        self,
        tag: str,
        path_labels: str,
        path_data: str,
        gpu: Optional[int] = None,
        verbose: bool = False,
        **kwargs,
    ):
        """Train classifier

        :param tag: classifier designation, refers to model tag in config.models
        :param path_labels: local path to csv file with the labels
        :param path_data: local path to alert json files
        :param gpu: GPU id to use, zero-based. check tf.config.list_physical_devices('GPU') for available devices
        :param verbose:
        :param kwargs: refer to config.models + utils.DNN.setup and utils.Dataset.make
        :return:
        """
        import tensorflow as tf

        if gpu is not None:
            # specified a GPU to run on?
            gpus = tf.config.list_physical_devices("GPU")
            tf.config.experimental.set_visible_devices(gpus[gpu], "GPU")
        else:
            # otherwise run on CPU
            tf.config.experimental.set_visible_devices([], "GPU")

        from acai import DNN

        train_config = self.config["models"][tag]
        features = self.config["features"][train_config["features"]]

        train_config["parameters"]["features_shape"] = (len(features),)
        train_config["parameters"]["triplet_shape"] = (63, 63, 3)

        # overwrite training parameters with kwargs from CLI
        train_config["parameters"] = {
            **train_config["parameters"],
            **kwargs,
        }

        ds = DataSet(
            tag=tag,
            path_labels=path_labels,
            path_data=path_data,
            features=features,
            verbose=verbose,
            **train_config["parameters"],
        )

        label = train_config["label"]

        datasets, metadata, indexes, steps_per_epoch, class_weight = ds.make(
            target_label=label,
            **train_config["parameters"],
        )

        classifier = DNN(name=tag)

        classifier.setup(**train_config["parameters"])

        if verbose:
            print(classifier.model.summary())

        pretrained_model = train_config["parameters"].get("pretrained_model", None)
        if pretrained_model is not None:
            classifier.load(pretrained_model)

        time_tag = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        if not kwargs.get("test", False):
            wandb.login(key=self.config["wandb"]["token"])
            wandb.init(
                project=self.config["wandb"]["project"],
                tags=[tag],
                name=f"{tag}-{time_tag}",
                config=train_config["parameters"],
            )
            classifier.meta["callbacks"].append(WandbCallback())

        classifier.train(
            datasets["train"],
            datasets["val"],
            steps_per_epoch["train"],
            steps_per_epoch["val"],
            epochs=train_config["parameters"].get("epochs", 100),
            class_weight=class_weight,
            verbose=verbose,
        )

        if not kwargs.get("test", False):
            # log some example
            test_array = list(datasets["test"].unbatch().as_numpy_iterator())
            test_features, test_triplets, test_labels = [], [], []

            for item in tqdm.tqdm(test_array):
                test_features.append(np.expand_dims(item[0]["features"], axis=[0, -1]))
                test_triplets.append(np.expand_dims(item[0]["triplets"], axis=[0]))
                test_labels.append(int(item[1]))

            test_features = np.vstack(test_features)
            test_triplets = np.vstack(test_triplets)

            preds = classifier.model.predict([test_features, test_triplets]).flatten()
            # apply threshold to float predictions
            preds_int = threshold(
                preds, t=train_config["parameters"].get("threshold", 0.5)
            )
            df_test = pd.DataFrame.from_records(
                [
                    {
                        "label": test_label,
                        "pred": pred,
                        "pred_int": pred_int,
                        "miss": bool(test_label ^ pred_int),
                        **{
                            key: value if key != "candid" else str(value)
                            for key, value in meta.items()
                        },
                        **{
                            key: wandb.Image(fig)
                            for key, fig in self.alert_images(
                                path_data=path_data, candid=meta["candid"]
                            ).items()
                            if fig is not None
                        },
                    }
                    for test_label, pred, pred_int, meta in zip(
                        test_labels, preds, preds_int, metadata["test"]
                    )
                ]
            )
            if verbose:
                print(df_test)

            wandb.log({"Test set": wandb.Table(dataframe=df_test)})

        if verbose:
            print("Evaluating on test set:")
        stats = classifier.evaluate(datasets["test"], verbose=verbose)
        if verbose:
            print(stats)

        param_names = (
            "loss",
            "tp",
            "fp",
            "tn",
            "fn",
            "accuracy",
            "precision",
            "recall",
            "auc",
        )
        if not kwargs.get("test", False):
            # log model performance on the test set
            for param, value in zip(param_names, stats):
                wandb.run.summary[f"test_{param}"] = value
            p, r = wandb.run.summary["test_precision"], wandb.run.summary["test_recall"]
            wandb.run.summary["test_f1"] = 2 * p * r / (p + r)

        if datasets["dropped_samples"] is not None:
            # log model performance on the dropped samples
            if verbose:
                print("Evaluating on samples dropped from the training set:")
            stats = classifier.evaluate(datasets["dropped_samples"], verbose=verbose)
            if verbose:
                print(stats)

            if not kwargs.get("test", False):
                for param, value in zip(param_names, stats):
                    wandb.run.summary[f"dropped_samples_{param}"] = value
                p, r = (
                    wandb.run.summary["dropped_samples_precision"],
                    wandb.run.summary["dropped_samples_recall"],
                )
                wandb.run.summary["dropped_samples_f1"] = (
                    2 * p * r / (p + r) if p + r != 0 else "undefined"
                )

        if train_config["parameters"].get("save", False):
            output_path = str(pathlib.Path(__file__).parent.absolute() / "models" / tag)
            if verbose:
                print(f"Saving model to {output_path}")
            classifier.save(
                output_path=output_path,
                output_format="hdf5",
                tag=time_tag,
            )

            return time_tag

    def sweep(
        self,
        tag: str,
        path_labels: str,
        path_data: str,
        gpu: Optional[int] = None,
        verbose: bool = False,
        **kwargs,
    ):
        """Run hyper-parameter tuning with W&B Sweeps

        :param tag:
        :param path_labels:
        :param path_data:
        :param gpu:
        :param verbose:
        :param kwargs:
        :return:
        """
        wandb.login(key=self.config["wandb"]["token"])

        project = self.config["wandb"]["project"]

        import tensorflow as tf

        if gpu is not None:
            # specified a GPU to run on?
            gpus = tf.config.list_physical_devices("GPU")
            tf.config.experimental.set_visible_devices(gpus[gpu], "GPU")
        else:
            # otherwise run on CPU
            tf.config.experimental.set_visible_devices([], "GPU")

        from acai import DNN

        train_config = self.config["training"]["classes"][tag]
        # train_config["sweep"] contains sweep configuration

        sweep_id = wandb.sweep(
            sweep=train_config["sweep"],
            project=project,
        )

        features = self.config["features"][train_config["features"]]

        ds = DataSet(
            tag=tag,
            path_labels=path_labels,
            path_data=path_data,
            features=features,
            verbose=verbose,
            **kwargs,
        )

        label = train_config["label"]

        def sweep_train():
            time_tag = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            wandb.init(
                tags=[tag],
                name=f"{tag}-{time_tag}",
                config={
                    "tag": tag,
                    "label": label,
                    "dataset": pathlib.Path(path_labels).name,
                },
            )
            dataset_parameters = {
                key: wandb.config.get(key, default_value)
                for (key, default_value) in (
                    ("threshold", 0.5),
                    ("balance", None),
                    ("weight_per_class", False),
                    ("scale_features", "min_max"),
                    ("test_size", 0.1),
                    ("val_size", 0.1),
                    ("random_state", 42),
                    ("feature_stats", None),
                    ("batch_size", 32),
                    ("shuffle_buffer_size", 128),
                    ("epochs", 100),
                    ("path_features", None),
                    ("path_triplets", None),
                )
            }

            datasets, metadata, indexes, steps_per_epoch, class_weight = ds.make(
                target_label=label,
                **dataset_parameters,
            )

            # set up and train model
            model_parameters = {
                key: wandb.config.get(key, default_value)
                for (key, default_value) in (
                    ("features_shape", (len(features),)),
                    ("triplet_shape", (63, 63, 3)),
                    ("dense_blocks", 2),
                    ("dense_block_units", 64),
                    ("dense_block_scale_factor", 0.5),
                    ("dense_activation", "relu"),
                    ("dense_dropout_rate", 0.25),
                    ("conv_blocks", 2),
                    ("conv_conv_layer_type", "SeparableConv2D"),
                    ("conv_pool_layer_type", "MaxPooling2D"),
                    ("conv_block_filters", 16),
                    ("conv_block_filter_size", (3, 3)),
                    ("conv_block_pool_size", (2, 2)),
                    ("conv_block_scale_factor", 2),
                    ("conv_dropout_rate", 0.25),
                    ("head_blocks", 1),
                    ("head_block_units", 16),
                    ("head_block_scale_factor", 0.5),
                    ("head_activation", "relu"),
                    ("head_dropout_rate", 0),
                    ("loss", "binary_crossentropy"),
                    ("optimizer", "adam"),
                    ("learning_rate", 3e-4),
                    ("momentum", 0.9),
                    ("monitor", "val_loss"),
                    ("patience", 5),
                    ("callbacks", ("reduce_lr_on_plateau", "early_stopping")),
                    ("save", False),
                )
            }
            # parse boolean args
            for bool_param in ("save",):
                model_parameters[bool_param] = forgiving_true(
                    model_parameters[bool_param]
                )

            classifier = DNN(name=tag)

            classifier.setup(**model_parameters)

            classifier.meta["callbacks"].append(WandbCallback())

            classifier.train(
                datasets["train"],
                datasets["val"],
                steps_per_epoch["train"],
                steps_per_epoch["val"],
                epochs=dataset_parameters.get("epochs"),
                class_weight=class_weight,
                verbose=verbose,
            )

        wandb.agent(sweep_id, function=sweep_train)

    def test(self):
        """Test different workflows"""
        import shutil
        import string
        import uuid

        # create a mock dataset and check that the training pipeline finishes
        labels = f"{uuid.uuid4().hex}.csv"

        path_mock = pathlib.Path(__file__).parent.absolute() / "data" / "mock"
        path_features = path_mock / f"{uuid.uuid4().hex}.npy"
        path_triplets = path_mock / f"{uuid.uuid4().hex}.npy"

        try:
            if not path_mock.exists():
                path_mock.mkdir(parents=True, exist_ok=True)

            n_samples = 2000

            np.save(str(path_features), np.random.random((n_samples, 25)))
            np.save(str(path_triplets), np.random.random((n_samples, 63, 63, 3)))

            entries = []
            for i in range(n_samples):
                entry = dict(
                    oid=f"ZTF87{''.join(random.choices(string.ascii_lowercase, k=7))}",
                    candid=random.randint(600000000000000000, 1600000000000000000),
                    label=random.choice(("h", "o", "n", "b", "v")),
                )
                entries.append(entry)

            df_mock = pd.DataFrame.from_records(entries)
            df_mock.to_csv(path_mock / labels, index=False)

            tag = "acai_h"
            time_tag = self.train(
                tag=tag,
                path_labels=str(path_mock / labels),
                path_data=str(path_mock / "alerts"),
                path_features=path_features,
                path_triplets=path_triplets,
                batch_size=16,
                epochs=3,
                verbose=True,
                save=True,
                test=True,
            )
            path_model = (
                pathlib.Path(__file__).parent.absolute() / "models" / tag / time_tag
            )
            shutil.rmtree(path_model)
        finally:
            # clean up after thyself
            (path_mock / labels).unlink()
            path_features.unlink()
            path_triplets.unlink()


if __name__ == "__main__":
    fire.Fire(ACAI)
