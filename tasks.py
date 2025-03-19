import pathlib
import subprocess
from typing import Optional

import invoke
import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from lib.const import ALL_METHODS, LANGUANGE_MAP_JP, PREPRINT_METHODS, translate


def read_df(
    data_path: pathlib.Path,
    methods: Optional[list[str]] = None,
    lan_map: Optional[dict[str, str]] = None,
) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    if methods is not None:
        df = df[df.method.isin(set(methods))]
    if lan_map is not None:
        df.method = df.method.apply(lambda s: translate(s, lan_map=lan_map))
        df.columns = list(map(lambda s: translate(s, lan_map=lan_map), df.columns))

    return df


@invoke.task
def preprocess(c: invoke.context.Context) -> None:
    subprocess.run("uv run src/preprocess.py", shell=True)


@invoke.task
def integration_test(c: invoke.context.Context) -> None:
    log_simulation_dir = "log/test/simulation"
    # simulation(c, root_out_dir=log_simulation_dir, mode="valid", is_debug=True)
    # evaluation(c, root_out_dir="log/test/evaluation", mode="valid", is_debug=True)
    create_preprint_figure(c, root_out_dir=log_simulation_dir)
    create_fullpaper_figure(c, root_out_dir=log_simulation_dir)


@invoke.task
def unit_test(c: invoke.context.Context) -> None:
    subprocess.run("pytest", shell=True)


@invoke.task
def simulation(
    c: invoke.context.Context, root_out_dir: str, mode: str, is_debug: bool = False
) -> None:
    methods = " ".join(ALL_METHODS)
    iter_max = 3 if is_debug else 50
    trial_nums = 3 if is_debug else 50
    total_rounds = 3 if is_debug else 200
    root_dir = pathlib.Path(root_out_dir)
    root_dir.mkdir(parents=True, exist_ok=True)
    for data_path, out_dir in [
        ("data/preprocessed/dunnhumby/data.json", root_dir / "dunnhumby"),
        ("data/preprocessed/tafeng/data.json", root_dir / "tafeng"),
        (None, root_dir / "artificial"),
    ]:
        cmd = " ".join(
            [
                "uv run src/simulation.py",
                f"--data_path={data_path}" if data_path is not None else "",
                f"--out_dir={out_dir}",
                f"--methods {methods}",
                f"--iter_max={iter_max}",
                f"--mode={mode}",
                f"--trial_nums={trial_nums}",
                f"--total_round={total_rounds}",
                f"| tee -a {root_dir / 'a.log'}",
            ]
        )
        print(cmd)
        subprocess.run(cmd, shell=True)


@invoke.task
def evaluation(
    c: invoke.context.Context,
    root_out_dir: str,
    mode: str,
    psi_strategy: str = "bernoulli",
    is_debug: bool = False,
) -> None:
    methods = " ".join(ALL_METHODS)
    iter_max = 3 if is_debug else 50
    max_eval_user = 3 if is_debug else 100
    eps = 0.5
    root_dir = pathlib.Path(root_out_dir)
    root_dir.mkdir(parents=True, exist_ok=True)
    for data_path, out_dir in [
        ("data/preprocessed/dunnhumby/data.json", root_dir / "dunnhumby"),
        ("data/preprocessed/tafeng/data.json", root_dir / "tafeng"),
    ]:
        cmd = " ".join(
            [
                "uv run src/evaluation.py",
                f"--data_path={data_path}" if data_path is not None else "",
                f"--out_dir={out_dir}",
                f"--methods {methods}",
                f"--iter_max={iter_max}",
                f"--mode={mode}",
                f"--eps={eps}",
                f"--max_eval_user={max_eval_user}",
                f"--psi_strategy={psi_strategy}",
                f"| tee -a {root_dir / 'a.log'}",
            ]
        )
        print(cmd)
        subprocess.run(cmd, shell=True)


@invoke.task
def create_preprint_figure(c: invoke.context.Context, root_out_dir: str) -> None:
    lan_map = LANGUANGE_MAP_JP
    _, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 6))
    metrics = list(
        map(lambda s: translate(s, lan_map=lan_map), ["entropy", "test-recall"])
    )
    datas = ["artificial", "dunnhumby", "tafeng"]
    root_out_dir = pathlib.Path(root_out_dir)
    for i, metric in enumerate(metrics):
        for j, data_name in enumerate(datas):
            legend = (i, j) == (0, 2)  # 右上だけ凡例を表示する
            sns.lineplot(
                x=translate("round", lan_map=lan_map),
                y=metric,
                hue=translate("method", lan_map=lan_map),
                style=translate("method", lan_map=lan_map),
                data=read_df(
                    root_out_dir / data_name / "log.csv",
                    methods=PREPRINT_METHODS,
                    lan_map=lan_map,
                ),
                ax=axes[i, j],
                legend=legend,
            )

    axes[0, 0].set_title(translate("artificial", lan_map=lan_map))
    axes[0, 1].set_title(translate("Dunnhumby", lan_map=lan_map))
    axes[0, 2].set_title(translate("Tafeng", lan_map=lan_map))
    for i in range(3):
        axes[0, i].set_ylabel(translate("entropy", lan_map=lan_map))
        axes[1, i].set_ylabel(translate("test-recall", lan_map=lan_map))

    sns.move_legend(axes[0, 2], "upper right")
    plt.tight_layout()
    plt.savefig(root_out_dir / "simulation.png")


@invoke.task
def create_fullpaper_figure(c: invoke.context.Context, root_out_dir: str) -> None:
    lan_map = LANGUANGE_MAP_JP
    root_out_dir = pathlib.Path(root_out_dir)
    _, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 6))
    metrics = list(
        map(
            lambda s: translate(s, lan_map=lan_map),
            [
                "test-recall",
                "entropy",
                "accuracy",
                "diversity:topic:category",
                "novelty:ip",
                "serendipity:topic:category",
            ],
        )
    )

    data_names = ["artificial", "dunnhumby", "tafeng"]
    fig, axes = plt.subplots(nrows=6, ncols=3, figsize=(16, 20))
    df = {
        data_name: read_df(
            root_out_dir / f"{data_name}/log.csv",
            methods=PREPRINT_METHODS,
            lan_map=lan_map,
        )
        for data_name in data_names
    }
    for j, data_name in enumerate(data_names):
        axes[0, j].set_title(translate(s=data_name, lan_map=lan_map))
        for i, metric in enumerate(metrics):
            legend = (i, j) == (len(metrics) - 1, 1)  # 右上だけ凡例を表示する
            if i == 3:
                y_min = [0.9, 0.99, 0.98]
                axes[i, j].set_ylim(y_min[j], 1.0)
            sns.lineplot(
                x=lan_map["round"],
                y=metric,
                hue=lan_map["method"],
                style=lan_map["method"],
                data=df[data_name],
                ax=axes[i, j],
                legend=legend,
            )

    axes[len(metrics) - 1, 1].legend(
        loc="upper center", ncol=5, bbox_to_anchor=(0.5, -0.2), fontsize=16
    )

    plt.tight_layout()
    plt.savefig(root_out_dir / "simulation-fullpaper.png")
