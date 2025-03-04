import numpy as np
import torch
import tqdm

from lib.bmd import BernoulliMixtureDistribution
from lib.util import Dataset, convert_to_history_array, convert_to_user_history, sigmoid


def augment_representive_point(
    psi: np.ndarray, weights: np.ndarray, rnd: np.random.RandomState, num_new_psi: int
) -> tuple[np.ndarray, np.ndarray]:
    new_psi = [e for e in psi]
    new_weights = [e for e in weights]

    # mixup psi to create new psi
    while len(new_psi) < num_new_psi:
        a, b = rnd.randint(len(psi)), rnd.randint(len(psi))
        w = rnd.uniform(0, 1)
        new_psi.append(w * psi[a] + (1 - w) * psi[b])
        new_weights.append(w * weights[a] + (1 - w) * weights[b])

    return np.array(new_psi), np.array(new_weights)


class EmbeddingModel(torch.nn.Module):
    def __init__(self, num_users: int, num_items: int, embedding_dim: int):
        super().__init__()
        self.user_embeddings = torch.nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = torch.nn.Embedding(num_items, embedding_dim)
        torch.nn.init.normal_(self.user_embeddings.weight, mean=0, std=0.01)
        torch.nn.init.normal_(self.item_embeddings.weight, mean=0, std=0.01)

    def forward(
        self, user_indices: torch.Tensor, item_indices: torch.Tensor
    ) -> torch.Tensor:
        user_embeds = self.user_embeddings(user_indices)
        item_embeds = self.item_embeddings(item_indices)
        scores = (user_embeds * item_embeds).sum(dim=1)  # 内積
        return scores


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data: list[tuple[int, int, float]]):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        user, item, label = self.data[idx]
        return (
            torch.tensor(user, dtype=torch.long),
            torch.tensor(item, dtype=torch.long),
            torch.tensor(label, dtype=torch.float32),
        )


def generate_user_item_embeddings(
    dataset: Dataset, iter_max: int
) -> tuple[np.ndarray, np.ndarray]:
    embedding_dim = 32
    learning_rate = 0.01
    lambda1 = embedding_dim * 1e-5
    lambda2 = embedding_dim * 1e-5
    batch_size = 64

    data: list[tuple[int, int, float]] = []
    for u in range(dataset.user_n):
        h = convert_to_user_history(
            train_p_u=dataset.train_p_u[u], train_r_u=dataset.train_r_u[u], user_id=u
        )
        for i, r in zip(h.indices, h.rs):
            data.append((u, i, r))

    dataloader = torch.utils.data.DataLoader(
        CustomDataset(data), batch_size=batch_size, shuffle=True
    )

    model = EmbeddingModel(
        num_users=dataset.user_n, num_items=dataset.item_n, embedding_dim=embedding_dim
    )
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(iter_max):
        total_bce_loss, total_reg_loss = 0.0, 0.0
        for user_batch, item_batch, label_batch in tqdm.tqdm(dataloader):
            pred = model(user_batch, item_batch)

            bce_loss = criterion(pred, label_batch)
            reg_loss = (
                lambda1 * model.user_embeddings(user_batch).norm(2) ** 2
                + lambda2 * model.item_embeddings(item_batch).norm(2) ** 2
            )
            loss = bce_loss + reg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_bce_loss += bce_loss.item()
            total_reg_loss += reg_loss.item()

        print(
            f"Epoch {epoch+1}/{iter_max}, BCE-Loss: {total_bce_loss:.4f}, "
            f"Reg-Loss: {total_reg_loss:.4f}"
        )

    user_embeddings = model.user_embeddings.weight.data.detach().numpy()
    item_embeddings = model.item_embeddings.weight.data.detach().numpy()

    return user_embeddings, item_embeddings


def generate_psi_from_dataset(
    strategy: str,
    dataset: Dataset,
    topic_nums: int,
    iter_max: int,
    rnd: np.random.RandomState,
    num_new_psi: int,
) -> tuple[np.ndarray, np.ndarray]:
    if strategy == "bernoulli":
        return generate_psi_bernoulli(
            dataset=dataset,
            topic_nums=topic_nums,
            iter_max=iter_max,
            rnd=rnd,
            num_new_psi=num_new_psi,
        )
    elif strategy == "embedding":
        return generate_psi_embedding(
            dataset=dataset,
            iter_max=iter_max,
            rnd=rnd,
            num_new_psi=num_new_psi,
        )
    else:
        raise ValueError(strategy)


def generate_psi_bernoulli(
    dataset: Dataset,
    topic_nums: int,
    iter_max: int,
    rnd: np.random.RandomState,
    num_new_psi: int,
) -> tuple[np.ndarray, np.ndarray]:
    H = convert_to_history_array(dataset=dataset)
    bmd = BernoulliMixtureDistribution(n_components=topic_nums)
    bmd.fit(H=H, rnd=rnd, iter_max=iter_max)
    psi = np.array(bmd.means.data)
    psi, weights = augment_representive_point(
        psi=psi, weights=bmd.weights, rnd=rnd, num_new_psi=num_new_psi
    )
    return psi, weights


def generate_psi_embedding(
    dataset: Dataset,
    iter_max: int,
    rnd: np.random.RandomState,
    num_new_psi: int,
) -> tuple[np.ndarray, np.ndarray]:
    e_u, e_i = generate_user_item_embeddings(dataset=dataset, iter_max=iter_max)
    psi: list[list[float]] = []
    weights = []

    # mixup to create new psi
    while len(psi) < num_new_psi:
        a, b = rnd.randint(len(e_u)), rnd.randint(len(e_u))
        w = rnd.uniform(0, 1)
        e_k = w * e_u[a] + (1 - w) * e_u[b]
        psi.append(sigmoid(np.dot(e_k, e_i.T)).tolist())
        weights.append(1.0)

    return np.array(psi), np.array(weights)
