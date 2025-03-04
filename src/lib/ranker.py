import abc

import numpy as np

from lib.criterion import InformationGainCriterion
from lib.util import Dataset, UserHistory


class DistanceMeasurer(abc.ABC):
    def dist(self, i: int, j: int) -> float:
        raise NotImplementedError()


class GeneralDistanceMeasurer(DistanceMeasurer):
    def __init__(self, distance_matrix: np.ndarray) -> None:
        self.distance_matrix = distance_matrix

    def dist(self, i: int, j: int) -> float:
        return self.distance_matrix[i, j]


class TopicDistanceMeasurer(DistanceMeasurer):
    def __init__(self, topics: list[int]) -> None:
        self.topics = topics

    def dist(self, i: int, j: int) -> float:
        if self.topics[i] != self.topics[j]:
            return 1.0
        return 0.0


class Ranker(abc.ABC):
    def rerank(
        self,
        h: UserHistory,
        cand_items: list[int],
    ) -> list[int]:
        raise NotImplementedError()


class AccuracyBasedRanker(Ranker):
    def __init__(self, model: InformationGainCriterion) -> None:
        self.model = model

    def rerank(self, h: UserHistory, cand_items: list[int]) -> list[int]:
        phi = self.model.transform(h)
        p = self.model.calc_r_ui(phi=phi)
        reranked = list(cand_items)
        reranked.sort(key=lambda i: p[i], reverse=True)
        return reranked


class DiversityBasedRanker(Ranker):
    def __init__(
        self,
        model: InformationGainCriterion,
        distance_measurer: DistanceMeasurer,
        ref_history: bool,
        lmda: float = 0.0,
    ) -> None:
        self.model = model
        self.distance_measurer = distance_measurer
        self.lmda = lmda

        # 多様性の計算をする際に、ユーザの推薦アイテムとの類似度を測るかどうか
        # シミュレーションの時に使うことを想定
        self.ref_history = ref_history

    def rerank(self, h: UserHistory, cand_items: list[int]) -> list[int]:
        if self.ref_history:
            return self.rerank_ref_history(h=h, cand_items=cand_items)
        else:
            return self.rerank_mmr(h=h, cand_items=cand_items)

    def rerank_mmr(self, h: UserHistory, cand_items: list[int]) -> list[int]:
        phi = self.model.transform(h)
        p = self.model.calc_r_ui(phi=phi)

        reranked: list[int] = []
        cand_items_set = set(cand_items)
        while (
            len(cand_items_set) > 0 and len(reranked) < 100
        ):  # 最大でもN=100まで最適化する
            max_i = 0
            max_e = -1e20
            for i in cand_items_set:
                d = 0.0
                for j in reranked:
                    d += self.distance_measurer.dist(i, j)

                d /= max(1.0, len(reranked))
                e = p[i] * self.lmda + d * (1 - self.lmda)
                if e > max_e:
                    max_i = i
                    max_e = e
            reranked.append(max_i)
            cand_items_set.remove(max_i)

        # 残りを追加する
        for i in cand_items_set:
            reranked.append(i)

        return reranked

    def rerank_ref_history(self, h: UserHistory, cand_items: list[int]) -> list[int]:
        phi = self.model.transform(h)
        p = self.model.calc_r_ui(phi=phi)

        ds = {}
        for i in cand_items:
            d = 0.0
            for j in h.indices:
                d += self.distance_measurer.dist(i, j)
            d /= max(1, len(h.indices))
            ds[i] = d

        reranked = list(cand_items)
        reranked.sort(
            key=lambda i: p[i] * self.lmda + ds[i] * (1 - self.lmda), reverse=True
        )
        return reranked


class DistanceFromUserBasedRanker(Ranker):
    def __init__(
        self,
        model: InformationGainCriterion,
        distance_measurer: DistanceMeasurer,
        lmda: float = 0.0,
    ) -> None:
        self.model = model
        self.distance_measurer = distance_measurer
        self.lmda = lmda

    def rerank(self, h: UserHistory, cand_items: list[int]) -> list[int]:
        return self.rerank_mmr(h=h, cand_items=cand_items)

    def rerank_mmr(self, h: UserHistory, cand_items: list[int]) -> list[int]:
        phi = self.model.transform(h)
        p = self.model.calc_r_ui(phi=phi)
        es = {
            i: sum(map(lambda j: self.distance_measurer.dist(i, j), h.p_list))
            for i in cand_items
        }

        reranked = list(cand_items)
        reranked.sort(
            key=lambda i: p[i] * self.lmda + es[i] * (1 - self.lmda), reverse=True
        )
        return reranked


class NoveltyBasedRanker(Ranker):
    def __init__(
        self,
        model: InformationGainCriterion,
        self_informations: list[float],
        lmda: float = 0.0,
    ) -> None:
        self.model = model
        self.self_informations = self_informations
        self.lmda = lmda

    def rerank(self, h: UserHistory, cand_items: list[int]) -> list[int]:
        phi = self.model.transform(h)
        p = self.model.calc_r_ui(phi=phi)
        ds = list(self.self_informations)

        reranked = list(cand_items)
        reranked.sort(
            key=lambda i: p[i] * self.lmda + ds[i] * (1 - self.lmda), reverse=True
        )
        return reranked


class SerendipityBasedRanker(Ranker):
    def __init__(
        self, model: InformationGainCriterion, topics: list[int], lmda: float = 0.0
    ) -> None:
        self.model = model
        self.topics = topics
        self.lmda = lmda

    def rerank(self, h: UserHistory, cand_items: list[int]) -> list[int]:
        phi = self.model.transform(h)
        p = self.model.calc_r_ui(phi=phi)
        seen_topic = set(list(map(lambda i: self.topics[i], h.p_list)))

        ds = {}
        for i in cand_items:
            if self.topics[i] not in seen_topic:
                ds[i] = p[i]
            else:
                ds[i] = 0

        reranked = list(cand_items)
        reranked.sort(
            key=lambda i: p[i] * self.lmda + ds[i] * (1 - self.lmda), reverse=True
        )
        return reranked


class InformationGainBasedRanker(Ranker):
    def __init__(self, model: InformationGainCriterion, lmda: float = 0.0) -> None:
        self.model = model
        self.lmda = lmda

    def rerank(self, h: UserHistory, cand_items: list[int]) -> list[int]:
        # \lambda * p(r_ui) + (1 - \lambda) * I(u, r_ui)
        phi = self.model.transform(h)
        p = self.model.calc_r_ui(phi=phi)
        ig = self.model.calc_conditional_entropies(i_list=cand_items, h=h, phi=phi)
        ig = {item: ig[i] for i, item in enumerate(cand_items)}
        reranked = list(cand_items)
        reranked.sort(
            key=lambda i: p[i] * self.lmda - ig[i] * (1 - self.lmda), reverse=True
        )
        return reranked


def get_ranker(
    method_str: str,
    model: InformationGainCriterion,
    dataset: Dataset,
    ref_history: bool,
) -> Ranker:
    def get_topics(args: list[str], item_topics: dict[str, list[int]]) -> list[int]:
        try:
            return item_topics[args[-1]]
        except (IndexError, KeyError):
            raise ValueError(args)

    args = method_str.split(":")
    method = args[0]

    # TODO: 定数をどこかで管理する
    if method == "accuracy":
        return AccuracyBasedRanker(model=model)
    elif method == "diversity":
        if args[1] == "topic":
            topics = get_topics(args=args, item_topics=dataset.item_topics)
            return DiversityBasedRanker(
                model=model,
                distance_measurer=TopicDistanceMeasurer(topics=topics),
                ref_history=ref_history,
            )
        elif args[1] == "embedding":
            return DiversityBasedRanker(
                model=model,
                distance_measurer=GeneralDistanceMeasurer(
                    distance_matrix=dataset.item_embedding_distances
                ),
                ref_history=ref_history,
            )
        else:
            raise ValueError(method_str)
    elif method == "novelty":
        if args[1] == "ip":
            return NoveltyBasedRanker(
                model=model,
                self_informations=dataset.self_informations,
            )
        elif args[1] == "du":
            if args[2] == "topic":
                topics = get_topics(args=args, item_topics=dataset.item_topics)
                return DistanceFromUserBasedRanker(
                    model=model,
                    distance_measurer=TopicDistanceMeasurer(topics=topics),
                )
            elif args[2] == "embedding":
                return DistanceFromUserBasedRanker(
                    model=model,
                    distance_measurer=GeneralDistanceMeasurer(
                        distance_matrix=dataset.item_embedding_distances
                    ),
                )
    elif method == "serendipity":
        topics = get_topics(args=args, item_topics=dataset.item_topics)
        return SerendipityBasedRanker(model=model, topics=topics)
    elif method == "information-gain":
        return InformationGainBasedRanker(model=model)
    raise ValueError(method_str)
