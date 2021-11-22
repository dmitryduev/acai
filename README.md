# ACAI: Alert-Classifying AI for the Zwicky Transient Facility

[![arXiv](https://img.shields.io/badge/arXiv-2111.XXXXX-brightgreen)](https://arxiv.org/abs/2111.XXXXX)

(presented at the NeurIPS 2021 [ML4PS workshop](https://ml4physicalsciences.github.io/2021/))

Astronomy has been experiencing an explosive increase in the data volumes,
doubling every two years or so. At the forefront of this revolution,
the [Zwicky Transient Facility (ZTF)](https://ztf.caltech.edu) – a robotic sky survey –
registers millions of transient events
(supernova explosions, asteroid detections, variable stars changing their brightness etc.)
in the dynamic sky every (clear) night.

Alert-Classifying AI (ACAI) is an open-source deep-learning framework
for the phenomenological classification of ZTF astronomical event alerts.

For more information and context, please see the
[Report on ACAI made with W&B](https://wandb.ai/dimaduev/acai/reports/Classification-of-astrophysical-events-with-ACAI--VmlldzoxMTkwNjYx).


## Command-Line Interface (CLI)

Use `acai.py` to execute actions such as fetching training data, training a model,
running hyper-parameter tuning or linting sources.
Requires a config file, `config.yaml` - please use `config.defaults.yaml` as an example.

Example usages of the CLI:

- Fetch training data including the individual alert packets converted to the `json` format:

```sh
./acai.py fetch-datasets --fetch_json_alerts=true
```

- Train a "hosted" classifier overloading the default parameters set in `config.yaml`:

```sh
./acai.py train \
  --tag=acai_h \
  --path-labels=data/d1.csv --path-data=data/alerts \
  --optimizer=adam --lr=0.001 \
  --epochs=300 --patience=50 \
  --batch_size=128 --balance=2.5 \
  --dense_block_units=64 --conv_block_filters=16 --head_block_units=16 \
  --parallel-data-load \
  --gpu=1 \
  --verbose
```

- Running hyper-parameter optimization:

```sh
./acai.py sweep --tag=acai_h --path-labels=data/labels.mini1k.csv --path-data=data/alerts
```
