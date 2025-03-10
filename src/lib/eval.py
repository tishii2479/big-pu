import itertools
from typing import Optional, Union

import numpy as np

from lib.criterion import InformationGainCriterion
from lib.util import convert_to_user_history

EPS = 1e-8


def eval_precision(
    rec_list: list[int], p_list: Union[list[int], set[int]]
) -> Optional[float]:
    return len(set(rec_list) & set(p_list)) / len(rec_list)


def eval_recall(
    rec_list: list[int], p_list: Union[list[int], set[int]]
) -> Optional[float]:
    p_set = set(p_list)
    if len(p_set) == 0:
        return None
    return len(set(rec_list) & p_set) / len(p_set)


def eval_topic_diversity(rec_list: list[int], topics: list[int]) -> Optional[float]:
    if len(rec_list) < 2:
        return None

    topic_count = [0] * len(topics)
    for i in rec_list:
        topic_count[topics[i]] += 1

    similarity = 1.0
    for c in topic_count:
        similarity += c * (c - 1) / 2
    return 1 - similarity / (len(rec_list) * (len(rec_list) - 1))


def eval_diversity(rec_list: list[int], distance_matrix: np.ndarray) -> Optional[float]:
    if len(rec_list) < 2:
        return None
    ret = 0.0
    for i in rec_list:
        for j in rec_list:
            if i == j:
                continue
            ret += distance_matrix[i, j]
    ret /= len(rec_list) * (len(rec_list) - 1)
    return ret


def eval_topic_distance_from_user(
    rec_list: list[int], train_p_u: list[int], topics: list[int]
) -> Optional[float]:
    if len(train_p_u) == 0:
        return None
    du = 0.0
    for i in rec_list:
        for j in train_p_u:
            du += 1 if topics[i] != topics[j] else 0
    du /= len(rec_list) * len(train_p_u)
    return du


def eval_distance_from_user(
    rec_list: list[int], train_p_u: list[int], distance_matrix: np.ndarray
) -> Optional[float]:
    if len(train_p_u) == 0:
        return None
    du = 0.0
    for i in rec_list:
        for j in train_p_u:
            du += distance_matrix[i, j]
    du /= len(rec_list) * len(train_p_u)
    return du


def eval_novelty(
    rec_list: list[int], self_informations: list[float]
) -> Optional[float]:
    novelty = 0.0
    for i in rec_list:
        novelty += self_informations[i]
    return novelty / len(rec_list)


def eval_information_gain(
    rec_list: list[int],
    train_p_u: list[int],
    train_r_u: list[int],
    eval_p_u: list[int],
    eval_r_u: list[int],
    model: InformationGainCriterion,
) -> Optional[float]:
    h = convert_to_user_history(train_p_u=train_p_u, train_r_u=train_r_u)
    rec_list = list(set(rec_list) & set(eval_r_u))

    if len(rec_list) == 0:
        return None

    val = 0.0
    p = model.transform(h)
    for i in rec_list:
        r = i in eval_p_u
        q = model.transform(h, additional_record=(i, r))
        val += np.sum(-p * np.log2(p + EPS)) - np.sum(-q * np.log2(q + EPS))

    return val / len(rec_list)


def eval_serendipity(
    rec_list: list[int], topics: list[int], y_list: list[int], p_list: list[int]
) -> Optional[float]:
    seen_topic = set(list(map(lambda i: topics[i], p_list)))
    serendipity = 0
    for i in rec_list:
        if i in y_list and topics[i] not in seen_topic:
            serendipity += 1

    return serendipity / len(rec_list)


def eval_prediction_coverage(rec_lists: list[list[int]], item_n: int) -> float:
    num_recommended_items = len(set(itertools.chain.from_iterable(rec_lists)))
    return num_recommended_items / item_n


class Evaluator:
    def __init__(
        self,
        model: InformationGainCriterion,
        self_informations: list[float],
        embedding_distances: np.ndarray,
        item_topics: dict[str, list[int]],
    ) -> None:
        self.model = model
        self.self_informations = self_informations
        self.embedding_distances = embedding_distances
        self.item_topics = item_topics

    def eval_rec_list(
        self,
        metrics: list[str],
        rec_list: list[int],
        train_p_u: list[int],
        train_r_u: list[int],
        eval_p_u: list[int],
        eval_r_u: list[int],
    ) -> dict[str, Optional[float]]:
        return {
            method: self.get_eval(
                method_str=method,
                rec_list=rec_list,
                train_p_u=train_p_u,
                train_r_u=train_r_u,
                eval_p_u=eval_p_u,
                eval_r_u=eval_r_u,
            )
            for method in metrics
        }

    def get_eval(
        self,
        method_str: str,
        rec_list: list[int],
        train_p_u: list[int],
        train_r_u: list[int],
        eval_p_u: list[int],
        eval_r_u: list[int],
    ) -> Optional[float]:
        def get_topics(args: list[str], item_topics: dict[str, list[int]]) -> list[int]:
            try:
                return item_topics[args[-1]]
            except (IndexError, KeyError):
                return item_topics["category"]

        args = method_str.split(":")
        method = args[0]
        topics = get_topics(args=args, item_topics=self.item_topics)

        # TODO: 定数をどこかで管理する
        if method == "accuracy":
            return eval_recall(
                rec_list=rec_list,
                p_list=eval_p_u,
            )
        elif method == "diversity":
            if args[1] == "topic":
                return eval_topic_diversity(rec_list=rec_list, topics=topics)
            elif args[1] == "embedding":
                return eval_diversity(
                    rec_list=rec_list, distance_matrix=self.embedding_distances
                )
        elif method == "novelty":
            if args[1] == "ip":
                return eval_novelty(
                    rec_list=rec_list,
                    self_informations=self.self_informations,
                )
            elif args[1] == "du":
                if args[2] == "topic":
                    return eval_topic_distance_from_user(
                        rec_list=rec_list,
                        train_p_u=train_p_u,
                        topics=topics,
                    )
                elif args[2] == "embedding":
                    return eval_distance_from_user(
                        rec_list=rec_list,
                        train_p_u=train_p_u,
                        distance_matrix=self.embedding_distances,
                    )
        elif method == "serendipity":
            return eval_serendipity(
                rec_list=rec_list,
                topics=topics,
                y_list=eval_p_u,
                p_list=train_p_u,
            )
        elif method == "information-gain":
            return eval_information_gain(
                rec_list=rec_list,
                train_p_u=train_p_u,
                train_r_u=train_r_u,
                eval_p_u=eval_p_u,
                eval_r_u=eval_r_u,
                model=self.model,
            )
        raise ValueError(method_str)
