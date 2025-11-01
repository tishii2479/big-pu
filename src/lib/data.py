import numpy as np

from lib.util import Dataset


def generate_artificial_psi(
    num_item: int = 5000,
    num_topic: int = 50,
    num_upper_topic: int = 10,
    num_k: int = 20,
    num_topic_per_k: int = 10,
    alpha: float = 10.0,
    random_state: int = 0,
) -> np.ndarray:
    assert num_topic % num_upper_topic == 0
    rnd = np.random.RandomState(random_state)
    s_i = num_item // num_topic
    psi = []
    for _ in range(num_k):
        u_topic = rnd.randint(num_upper_topic)
        p = np.array(
            [1.0] * u_topic + [alpha] + [1.0] * (num_upper_topic - u_topic - 1)
        ).repeat(
            num_topic // num_upper_topic
        )  # 上位トピックu_topicに所属するトピックの確率を高くする
        p /= p.sum()
        topics = rnd.choice(range(num_topic), p=p, size=num_topic_per_k, replace=False)
        topic_item_distr = [0.9 if k in topics else 0.1 for k in range(num_topic)]
        topic_item_distr = np.repeat(topic_item_distr, s_i)
        psi.append(topic_item_distr.tolist())

    return np.array(psi)


def generate_artificial_dataset(
    num_user: int = 5000,
    num_item: int = 1000,
    num_topic: int = 50,
    num_u_topic: int = 10,
    num_k: int = 20,
    num_topic_per_k: int = 10,
    num_sample_item: int = 500,
    alpha: float = 10.0,
    random_state: int = 0,
) -> Dataset:
    s_u = num_user // num_k
    rnd = np.random.RandomState(random_state)

    psi = generate_artificial_psi(
        num_item=num_item,
        num_upper_topic=num_u_topic,
        num_topic=num_topic,
        num_k=num_k,
        num_topic_per_k=num_topic_per_k,
        alpha=alpha,
        random_state=random_state,
    )

    train_p_u: dict[int, list[int]] = {u: [] for u in range(num_user)}
    train_r_u: dict[int, list[int]] = {u: [] for u in range(num_user)}
    eval_p_u: dict[int, list[int]] = {u: [] for u in range(num_user)}
    eval_r_u: dict[int, list[int]] = {u: [] for u in range(num_user)}

    for u in range(num_user):
        k = u // s_u
        topic_item_distr = psi[k]
        for i in rnd.choice(range(num_item), size=num_sample_item, replace=False):
            r = 1 if rnd.random() < topic_item_distr[i] else 0
            train_r_u[u].append(i)
            if r == 1:
                train_p_u[u].append(i)

        for i in rnd.choice(range(num_item), size=num_sample_item, replace=False):
            r = 1 if rnd.random() < topic_item_distr[i] else 0
            eval_r_u[u].append(i)
            if r == 1:
                eval_p_u[u].append(i)

    item_topics = {
        "category": [i // (num_item // num_topic) for i in range(num_item)],
        "category-u": [i // (num_item // num_u_topic) for i in range(num_item)],
    }
    dataset = Dataset(
        user_n=num_user,
        item_n=num_item,
        item_topics=item_topics,
        train_p_u=train_p_u,
        train_r_u=train_r_u,
        eval_p_u=eval_p_u,
        eval_r_u=eval_r_u,
    )
    return dataset


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.colors import LinearSegmentedColormap

    psi = generate_artificial_psi()
    colors = np.array(
        [
            [255, 255, 255, 1],  # white
            [0, 7, 50, 1],
        ],
        dtype=np.float64,
    )
    colors[:, :3] /= 256
    cmap = LinearSegmentedColormap.from_list("cmap", colors=colors)
    sns.heatmap(psi, cmap=cmap)
    plt.xlabel("i")
    plt.ylabel("k")
    plt.tight_layout()
    plt.show()

    # dataset = generate_artificial_dataset()
    # dataset.print_info()
