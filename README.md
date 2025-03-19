# Implementation of "An Evaluation Metric for Exploration based on Preference Elicitation in Recommender Systems"

## Setup

Requires `uv` for project setup.

```shell
$ uv --version
uv 0.5.20 (1c17662b3 2025-01-15)

$ uv sync
```

## How to reproduce experiment results

### Datasets

1. For Dunnhumby: download `Dunnhumby: The complete journey` from https://www.dunnhumby.com/source-files/, and place folders to `data/`
2. For Tafeng: download csv file from https://www.kaggle.com/datasets/chiranjivdas09/ta-feng-grocery-dataset, and place to `data/tafeng`
3. Run preprocess scripts
    ```shell
    $ uv run inv preprocess
    ```

### Simulation

```shell
$ uv run inv simulation --root-out-dir={LOG_SIMULATION_DIR} --mode=valid
```

### Analysis

```shell
$ uv run inv simulation --root-out-dir={LOG_EVALUATION_DIR} --mode=valid --psi-strategy=bernoulli
```

### Creating figures

```shell
$ uv run inv create-preprint-figure --root-out-dir={LOG_SIMULATION_DIR}
$ uv run inv create-fullpaper-figure --root-out-dir={LOG_SIMULATION_DIR}
```

## Configs

### `src/simulation.py`

```shell
$ uv run src/simulation.py --help
usage: simulation.py [-h] [--data_path DATA_PATH] --out_dir OUT_DIR --methods
                     [METHODS ...] [--mode MODE] [--seed SEED]
                     [--topic_nums TOPIC_NUMS] [--trial_nums TRIAL_NUMS]
                     [--iter_max ITER_MAX] [--total_round TOTAL_ROUND]
                     [--num_psi NUM_PSI] [--eps EPS]
                     [--psi_strategy {bernoulli,embedding}]

options:
  -h, --help            show this help message and exit
  --data_path DATA_PATH
  --out_dir OUT_DIR
  --methods [METHODS ...]
  --mode MODE
  --seed SEED
  --topic_nums TOPIC_NUMS
  --trial_nums TRIAL_NUMS
  --iter_max ITER_MAX
  --total_round TOTAL_ROUND
  --num_psi NUM_PSI
  --eps EPS
  --psi_strategy {bernoulli,embedding}
```

### `src/evaluation.py`

```shell
$ uv run src/evaluation.py --help
usage: evaluation.py [-h] [--data_path DATA_PATH] --out_dir OUT_DIR --methods
                     [METHODS ...] [--mode MODE] [--ranking RANKING] [--seed SEED]
                     [--topic_nums TOPIC_NUMS] [--iter_max ITER_MAX] [--eps EPS]
                     [--max_eval_user MAX_EVAL_USER] [--top_k TOP_K]
                     [--num_psi NUM_PSI] [--psi_strategy {bernoulli,embedding}]

options:
  -h, --help            show this help message and exit
  --data_path DATA_PATH
  --out_dir OUT_DIR
  --methods [METHODS ...]
  --mode MODE
  --ranking RANKING
  --seed SEED
  --topic_nums TOPIC_NUMS
  --iter_max ITER_MAX
  --eps EPS
  --max_eval_user MAX_EVAL_USER
  --top_k TOP_K
  --num_psi NUM_PSI
  --psi_strategy {bernoulli,embedding}
```
