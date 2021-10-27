#!/usr/bin/env python
from contextlib import contextmanager
import datetime
from deepdiff import DeepDiff
import fire

# import numpy as np
import os

# import pandas as pd
import pathlib
from penquins import Kowalski
from pprint import pprint
import questionary
import subprocess
import sys
from typing import Optional, Sequence, Union
import yaml

from acai.utils import forgiving_true, load_config, log


@contextmanager
def status(message):
    """
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


class Scope:
    def __init__(self):
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
        """
        Fetch ACAI datasets from GCP

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

    def train(
        self,
        tag: str,
        path_dataset: Union[str, pathlib.Path],
        gpu: Optional[int] = None,
        verbose: bool = False,
        **kwargs,
    ):
        """Train classifier

        :param tag: classifier designation, refers to "class" in config.taxonomy
        :param path_dataset: local path to csv file with the dataset
        :param gpu: GPU id to use, zero-based. check tf.config.list_physical_devices('GPU') for available devices
        :param verbose:
        :param kwargs: refer to utils.DNN.setup and utils.Dataset.make
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

        import wandb
        from wandb.keras import WandbCallback

        from acai.nn import DNN
        from acai.utils import Dataset

        train_config = self.config["training"]["classes"][tag]

        features = self.config["features"][train_config["features"]]

        ds = Dataset(
            tag=tag,
            path_dataset=path_dataset,
            features=features,
            verbose=verbose,
            **kwargs,
        )

        label = train_config["label"]

        # values from kwargs override those defined in config. if latter is absent, use reasonable default
        threshold = kwargs.get("threshold", train_config.get("threshold", 0.5))
        balance = kwargs.get("balance", train_config.get("balance", None))
        weight_per_class = kwargs.get(
            "weight_per_class", train_config.get("weight_per_class", False)
        )
        scale_features = kwargs.get("scale_features", "min_max")

        test_size = kwargs.get("test_size", train_config.get("test_size", 0.1))
        val_size = kwargs.get("val_size", train_config.get("val_size", 0.1))
        random_state = kwargs.get("random_state", train_config.get("random_state", 42))
        feature_stats = self.config.get("feature_stats", None)

        batch_size = kwargs.get("batch_size", train_config.get("batch_size", 64))
        shuffle_buffer_size = kwargs.get(
            "shuffle_buffer_size", train_config.get("shuffle_buffer_size", 512)
        )
        epochs = kwargs.get("epochs", train_config.get("epochs", 100))

        datasets, indexes, steps_per_epoch, class_weight = ds.make(
            target_label=label,
            threshold=threshold,
            balance=balance,
            weight_per_class=weight_per_class,
            scale_features=scale_features,
            test_size=test_size,
            val_size=val_size,
            random_state=random_state,
            feature_stats=feature_stats,
            batch_size=batch_size,
            shuffle_buffer_size=shuffle_buffer_size,
            epochs=epochs,
        )

        # set up and train model
        dense_branch = kwargs.get("dense_branch", True)
        conv_branch = kwargs.get("conv_branch", True)
        loss = kwargs.get("loss", "binary_crossentropy")
        optimizer = kwargs.get("optimizer", "adam")
        lr = float(kwargs.get("lr", 3e-4))
        momentum = float(kwargs.get("momentum", 0.9))
        monitor = kwargs.get("monitor", "val_loss")
        patience = int(kwargs.get("patience", 20))
        callbacks = kwargs.get("callbacks", ("reduce_lr_on_plateau", "early_stopping"))
        run_eagerly = kwargs.get("run_eagerly", False)
        pre_trained_model = kwargs.get("pre_trained_model")
        save = kwargs.get("save", False)

        # parse boolean args
        dense_branch = forgiving_true(dense_branch)
        conv_branch = forgiving_true(conv_branch)
        run_eagerly = forgiving_true(run_eagerly)
        save = forgiving_true(save)

        classifier = DNN(name=tag)

        classifier.setup(
            dense_branch=dense_branch,
            features_input_shape=(len(features),),
            conv_branch=conv_branch,
            dmdt_input_shape=(26, 26, 1),
            loss=loss,
            optimizer=optimizer,
            learning_rate=lr,
            momentum=momentum,
            monitor=monitor,
            patience=patience,
            callbacks=callbacks,
            run_eagerly=run_eagerly,
        )

        if verbose:
            print(classifier.model.summary())

        if pre_trained_model is not None:
            classifier.load(pre_trained_model)

        time_tag = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        if not kwargs.get("test", False):
            wandb.login(key=self.config["wandb"]["token"])
            wandb.init(
                project=self.config["wandb"]["project"],
                tags=[tag],
                name=f"{tag}-{time_tag}",
                config={
                    "tag": tag,
                    "label": label,
                    "dataset": pathlib.Path(path_dataset).name,
                    "scale_features": scale_features,
                    "learning_rate": lr,
                    "epochs": epochs,
                    "patience": patience,
                    "random_state": random_state,
                    "batch_size": batch_size,
                    "architecture": "scope-net",
                    "dense_branch": dense_branch,
                    "conv_branch": conv_branch,
                },
            )
            classifier.meta["callbacks"].append(WandbCallback())

        classifier.train(
            datasets["train"],
            datasets["val"],
            steps_per_epoch["train"],
            steps_per_epoch["val"],
            epochs=epochs,
            class_weight=class_weight,
            verbose=verbose,
        )

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
                wandb.run.summary["dropped_samples_f1"] = 2 * p * r / (p + r)

        if save:
            output_path = str(pathlib.Path(__file__).parent.absolute() / "models" / tag)
            if verbose:
                print(f"Saving model to {output_path}")
            classifier.save(
                output_path=output_path,
                output_format="tf",
                tag=time_tag,
            )

            return time_tag

    def test(self):
        """Test different workflows

        :return:
        """
        # import uuid
        # import shutil

        # # create a mock dataset and check that the training pipeline works
        # dataset = f"{uuid.uuid4().hex}.csv"
        # path_mock = pathlib.Path(__file__).parent.absolute() / "data" / "training"
        #
        # try:
        #     if not path_mock.exists():
        #         path_mock.mkdir(parents=True, exist_ok=True)
        #
        #     feature_names = self.config["features"]["ontological"]
        #     class_names = [
        #         self.config["training"]["classes"][class_name]["label"]
        #         for class_name in self.config["training"]["classes"]
        #     ]
        #
        #     entries = []
        #     for i in range(1000):
        #         entry = {
        #             **{
        #                 feature_name: np.random.normal(0, 0.1)
        #                 for feature_name in feature_names
        #             },
        #             **{
        #                 class_name: np.random.choice([0, 1])
        #                 for class_name in class_names
        #             },
        #             **{"non-variable": np.random.choice([0, 1])},
        #             **{"dmdt": np.abs(np.random.random((26, 26))).tolist()},
        #         }
        #         entries.append(entry)
        #
        #     df_mock = pd.DataFrame.from_records(entries)
        #     df_mock.to_csv(path_mock / dataset, index=False)
        #
        #     tag = "h"
        #     time_tag = self.train(
        #         tag=tag,
        #         path_dataset=path_mock / dataset,
        #         batch_size=32,
        #         epochs=3,
        #         verbose=True,
        #         save=True,
        #         test=True,
        #     )
        #     path_model = (
        #         pathlib.Path(__file__).parent.absolute() / "models" / tag / time_tag
        #     )
        #     shutil.rmtree(path_model)
        # finally:
        #     # clean up after thyself
        #     (path_mock / dataset).unlink()


if __name__ == "__main__":
    fire.Fire(Scope)
