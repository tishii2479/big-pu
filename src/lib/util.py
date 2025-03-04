import json
from typing import Optional

import numpy as np
import pandas as pd
import tqdm
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF


class Dataset:
    def __init__(
        self,
        user_n: int,
        item_n: int,
        item_topics: dict[str, list[int]],
        train_p_u: dict[int, list[int]],
        train_r_u: dict[int, list[int]],
        eval_p_u: dict[int, list[int]],
        eval_r_u: dict[int, list[int]],
    ):
        self.user_n = user_n
        self.item_n = item_n
        self.item_topics = item_topics
        print("create-topic-cluster:start")
        self.item_topics["latent-cluster"], self.item_embeddings = create_topic_cluster(
            p_u=train_p_u, item_n=item_n, latent_factor=64, n_topics=50, random_state=0
        )
        print("create-topic-cluster:end")
        self.item_embedding_distances = (
            1 + cos_sim_matrix(embeddings=self.item_embeddings)
        ) / 2  # [-1, 1] -> [0, 1]に変換
        self.train_p_u = train_p_u
        self.train_r_u = train_r_u
        self.eval_p_u = eval_p_u
        self.eval_r_u = eval_r_u
        self.self_informations = calculate_self_information(
            H_list=train_p_u,
            item_n=item_n,
        )
        self.print_info()

    @classmethod
    def from_json(cls, json_file: str, mode: str) -> "Dataset":
        with open(json_file, "r") as f:
            json_data = json.load(f)

        user_n = json_data["params"]["user_n"]
        item_n = json_data["params"]["item_n"]
        item_topics = json_data["item_topics"]

        data_df = pd.DataFrame(json_data["data"][mode]).T
        data_df = data_df.map(lambda d: d if isinstance(d, list) else [])
        data_df.index = data_df.index.astype(int)

        train_p_u = data_df["train_purchased_items"].to_dict()
        train_r_u = data_df["train_recommended_items"].to_dict()
        eval_p_u = data_df["eval_purchased_items"].to_dict()
        eval_r_u = data_df["eval_recommended_items"].to_dict()

        return cls(
            user_n=user_n,
            item_n=item_n,
            item_topics=item_topics,
            train_p_u=train_p_u,
            train_r_u=train_r_u,
            eval_p_u=eval_p_u,
            eval_r_u=eval_r_u,
        )

    def print_info(self) -> None:
        print(f"{self.user_n=} {self.item_n=}")
        print("total_train_p_u:", sum(map(len, self.train_p_u.values())))
        print("total_train_r_u:", sum(map(len, self.train_r_u.values())))
        print("total_eval_p_u:", sum(map(len, self.eval_p_u.values())))
        print("total_eval_r_u:", sum(map(len, self.eval_r_u.values())))
        print("item_topics:")
        for topic_name, topics in self.item_topics.items():
            print(f"    topic: {topic_name}, count: {len(set(topics))}")


class UserHistory:
    def __init__(
        self,
        indices: list[int],
        rs: list[float],
        p_list: list[int],
        user_id: Optional[int] = None,
    ) -> None:
        self.indices: list[int] = indices
        self.rs: list[float] = rs
        self.p_list: list[int] = p_list

        self.user_id = user_id

    def add_next(self, i: int, r: int) -> "UserHistory":
        indices = list(self.indices)
        rs = list(self.rs)
        p_list = list(self.p_list)
        indices.append(i)
        rs.append(r)
        if r == 1:
            p_list.append(i)
        return UserHistory(indices=indices, rs=rs, p_list=p_list, user_id=self.user_id)

    def add_history(self, i: int, r: int) -> None:
        self.indices.append(i)
        self.rs.append(r)


def convert_to_user_history(
    train_p_u: list[int], train_r_u: list[int], user_id: Optional[int] = None
) -> UserHistory:
    h = UserHistory(indices=[], rs=[], p_list=list(train_p_u), user_id=user_id)
    a = set(train_p_u)
    for i in train_r_u:
        r = 1 if i in a else 0
        h.add_history(i, r)
    return h


def convert_to_history_array(dataset: Dataset) -> np.ma.MaskedArray:
    H = np.zeros((dataset.user_n, dataset.item_n))
    H[:] = np.nan
    for u, p_u in dataset.train_r_u.items():
        a = set(dataset.train_p_u[u])
        for i in p_u:
            r = 1 if i in a else 0
            H[u, i] = r
    return np.ma.masked_where(np.isnan(H), H)


def calculate_self_information(
    H_list: dict[int, list[int]], item_n: int, eps: float = 1e-8
) -> list[float]:
    item_count = [0] * item_n
    for p_u in tqdm.tqdm(H_list.values()):
        if p_u is None:
            continue
        for i in p_u:
            item_count[i] += 1

    return [-np.log2((item_count[i] + eps) / item_n) for i in range(item_n)]


def create_topic_cluster(
    p_u: dict[int, list[int]],
    item_n: int,
    latent_factor: int,
    n_topics: int,
    random_state: int,
) -> tuple[list[int], np.ndarray]:
    m = np.zeros((item_n, item_n)).tolist()
    for p_u in tqdm.tqdm(p_u.values()):
        p_u = set(p_u)  # uniquify
        for i in p_u:
            for j in p_u:
                if i == j:
                    continue
                m[i][j] += 1

    m = np.array(m)
    nmf = NMF(n_components=latent_factor, random_state=random_state, max_iter=1_000)
    X = nmf.fit_transform(m)
    kmeans = KMeans(n_clusters=n_topics, random_state=random_state).fit(X)
    return kmeans.labels_.tolist(), X


# https://qiita.com/fam_taro/items/dac3b1bcfc01461a0120
def cos_sim_matrix(embeddings: np.ndarray) -> np.ndarray:
    d = embeddings @ embeddings.T
    norm = np.maximum(1e-6, (embeddings * embeddings).sum(axis=1, keepdims=True) ** 0.5)
    return d / norm / norm.T


def sigmoid(a: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-a))
