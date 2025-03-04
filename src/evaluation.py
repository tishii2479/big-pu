import pathlib

import classopt
import numpy as np
import pandas as pd
import tqdm

from lib.criterion import InformationGainCriterion
from lib.eval import Evaluator
from lib.ranker import get_ranker
from lib.util import Dataset, convert_to_user_history
from util import generate_psi_from_dataset


@classopt.classopt(default_long=True)
class Args:
    data_path: str = classopt.config(default=None)
    out_dir: str = classopt.config(required=True)
    methods: list[str] = classopt.config(required=True)
    mode: str = "valid"
    ranking: str = "none"
    seed: int = 0
    topic_nums: int = 50
    iter_max: int = 50
    eps: float = 0.0
    max_eval_user: int = 100
    top_k: int = 30
    num_psi: int = 1_000
    psi_strategy: str = classopt.config(
        choices=["bernoulli", "embedding"], default="bernoulli"
    )

    def __post_init__(self) -> None:
        self.out_path = pathlib.Path(self.out_dir)
        self.out_path.mkdir(parents=True, exist_ok=True)
        print(f"output to: {self.out_path}")


def output_evals(evals: dict[str, list[dict]], out_dir: pathlib.Path) -> None:
    results = {}
    for method, evals in evals.items():
        d = pd.DataFrame(evals)
        results[method] = d.mean()
    df = pd.DataFrame(results)
    df.index.name = "method"
    print(df)
    df.to_csv(out_dir / "result.csv")


def main() -> None:
    args = Args.from_args()  # type: ignore
    print("args:", args)

    dataset = Dataset.from_json(args.data_path, mode=args.mode)

    psi, weights = generate_psi_from_dataset(
        strategy=args.psi_strategy,
        dataset=dataset,
        topic_nums=args.topic_nums,
        iter_max=args.iter_max,
        rnd=np.random.RandomState(args.seed),
        num_new_psi=args.num_psi,
    )
    model = InformationGainCriterion(psi=psi, weights=weights, eps=args.eps)
    evaluator = Evaluator(
        model=model,
        self_informations=dataset.self_informations,
        embedding_distances=dataset.item_embedding_distances,
        item_topics=dataset.item_topics,
    )
    evals: dict[str, list[dict]] = {method: [] for method in args.methods}

    for u in tqdm.tqdm(range(min(args.max_eval_user, dataset.user_n))):
        user_history = convert_to_user_history(
            train_p_u=dataset.train_p_u[u],
            train_r_u=dataset.train_r_u[u],
            user_id=u,
        )
        for method in args.methods:
            ranker = get_ranker(
                method_str=method, model=model, dataset=dataset, ref_history=False
            )

            rec_list = ranker.rerank(
                user_history, cand_items=list(range(dataset.item_n))
            )[: args.top_k]
            evals[method].append(
                evaluator.eval_rec_list(
                    metrics=args.methods,
                    rec_list=rec_list,
                    train_p_u=dataset.train_p_u[u],
                    train_r_u=dataset.train_r_u[u],
                    eval_p_u=dataset.eval_p_u[u],
                    eval_r_u=dataset.eval_r_u[u],
                )
            )
    output_evals(evals=evals, out_dir=args.out_path)


if __name__ == "__main__":
    main()
