import pathlib

import classopt
import numpy as np
import pandas as pd
import tqdm

from lib.criterion import InformationGainCriterion
from lib.data import generate_artificial_dataset
from lib.eval import Evaluator, eval_recall
from lib.ranker import AccuracyBasedRanker, Ranker, get_ranker
from lib.util import Dataset, UserHistory
from util import generate_psi_from_dataset

EPS = 1e-8


@classopt.classopt(default_long=True)
class Args:
    data_path: str = classopt.config(default=None)
    out_dir: str = classopt.config(required=True)
    methods: list[str] = classopt.config(required=True)
    mode: str = "valid"
    seed: int = 0
    topic_nums: int = 50
    trial_nums: int = 5
    iter_max: int = 50
    total_round: int = 100
    num_psi: int = 1_000
    eps: float = 0.0
    psi_strategy: str = classopt.config(
        choices=["bernoulli", "embedding"], default="bernoulli"
    )

    def __post_init__(self) -> None:
        self.out_path = pathlib.Path(self.out_dir)
        self.out_path.mkdir(parents=True, exist_ok=True)
        print(f"output to: {self.out_path}")


def get_first_item_from_reranked_list(
    ranker: Ranker,
    h: UserHistory,
    cand_items: list[int],
    recommended: list[int],
) -> int:
    L = ranker.rerank(h=h, cand_items=cand_items)
    for j in L:
        if j not in recommended:
            return j
    raise RuntimeError()


def evaluate_test_recall(
    model: InformationGainCriterion,
    h: UserHistory,
    cand_items: list[int],
    targets: set[int],
) -> float:
    accuracy_based_ranker = AccuracyBasedRanker(model=model)
    test_rec_list = accuracy_based_ranker.rerank(
        h=h,
        cand_items=cand_items,
    )
    test_rec_list = test_rec_list[:50]  # Top-50で評価する
    recall = eval_recall(rec_list=test_rec_list, p_list=targets)
    assert recall is not None
    return recall


def simulation(
    method: str,
    model: InformationGainCriterion,
    dataset: Dataset,
    total_round: int,
    seed: int,
    metrics: list[str],
    verbose: bool = False,
) -> list[dict]:
    rnd = np.random.RandomState(seed)
    ranker = get_ranker(
        method_str=method, model=model, dataset=dataset, ref_history=True
    )
    evaluator = Evaluator(
        model=model,
        self_informations=dataset.self_informations,
        embedding_distances=dataset.item_embedding_distances,
        item_topics=dataset.item_topics,
    )

    targets, test_targets, cand_items, test_cand_items = split_items(
        psi=model.psi, rnd=rnd
    )

    h = UserHistory(indices=[], rs=[], p_list=[])
    rec_log: list[int] = []
    log_data = []

    for round in tqdm.tqdm(range(total_round)):
        i = get_first_item_from_reranked_list(
            ranker=ranker,
            h=h,
            cand_items=cand_items,
            recommended=rec_log,
        )
        assert i not in rec_log
        rec_log.append(i)

        # update user history
        r = 1 if i in targets else 0
        h = h.add_next(i, r)

        # add logs
        phi = model.transform(h)
        entropy = np.sum(-phi * np.log2(phi + EPS))
        recall = evaluate_test_recall(
            model=model, h=h, cand_items=test_cand_items, targets=test_targets
        )
        log = {
            "method": method,
            "seed": seed,
            "round": round + 1,
            "item": i,
            "test-recall": recall,
            "hit": np.sum(h.rs) / len(targets),
            "entropy": entropy,
        }
        log.update(
            evaluator.eval_rec_list(
                metrics=metrics,
                rec_list=h.indices,
                train_p_u=[],
                train_r_u=[],
                eval_p_u=list(targets),
                eval_r_u=h.indices,
            )
        )
        log_data.append(log)

        if verbose:
            print(f"r={round:4}: {log}")

    return log_data


def split_items(
    psi: np.ndarray, rnd: np.random.RandomState
) -> tuple[set[int], set[int], list[int], list[int]]:
    while True:
        # sample true theta for target user
        k = rnd.randint(len(psi))
        theta = psi[k]
        item_n = len(theta)

        # generate targets
        targets = set()
        for i in range(item_n):
            if rnd.random() < theta[i]:
                targets.add(i)

        # 全体の1/2をテスト用のアイテムとして分ける
        test_cand_items = rnd.choice(range(item_n), item_n // 2, replace=False).tolist()
        test_targets = set(test_cand_items) & targets
        targets = targets - test_targets
        cand_items = list(set(range(item_n)) - set(test_cand_items))

        # test_targetsの個数が0の場合はシミュレーションをせず、再度抽選する
        if len(test_targets) > 0:
            return targets, test_targets, cand_items, test_cand_items


def main() -> None:
    args = Args.from_args()  # type: ignore
    print("args:", args)

    if args.data_path is None:
        dataset = generate_artificial_dataset()
    else:
        dataset = Dataset.from_json(json_file=args.data_path, mode=args.mode)
    psi, weights = generate_psi_from_dataset(
        strategy=args.psi_strategy,
        dataset=dataset,
        topic_nums=args.topic_nums,
        iter_max=args.iter_max,
        rnd=np.random.RandomState(args.seed),
        num_new_psi=args.num_psi,
    )
    model = InformationGainCriterion(psi=psi, weights=weights, eps=args.eps)
    log_datas: list[dict] = []

    for method in args.methods:
        print(f"simulate: {method}")
        for seed in range(args.trial_nums):
            log_data = simulation(
                method=method,
                model=model,
                dataset=dataset,
                total_round=args.total_round,
                seed=args.seed + seed,
                metrics=args.methods,
                verbose=False,
            )
            print(log_data[-1])
            log_datas.extend(log_data)
    df = pd.DataFrame(log_datas)
    df.to_csv(args.out_path / "log.csv", index=False)


if __name__ == "__main__":
    main()
