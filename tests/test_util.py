import numpy as np
import pytest

from lib.criterion import InformationGainCriterion
from lib.data import generate_artificial_dataset
from lib.eval import eval_serendipity
from lib.util import UserHistory
from util import generate_psi_bernoulli, generate_psi_embedding

EPS = 1e-8


def calc_conditional_entropy_slow(
    model: InformationGainCriterion, i: int, h: UserHistory, phi: np.ndarray
) -> float:
    h_0, h_1 = h.add_next(i, 0), h.add_next(i, 1)
    p_1 = (phi * model.psi[:, i]).sum()
    assert 0 <= p_1 <= 1
    q_0 = model.transform(h_0)
    q_1 = model.transform(h_1)
    return p_1 * np.sum(-q_1 * np.log2(EPS + q_1)) + (1 - p_1) * np.sum(
        -q_0 * np.log2(EPS + q_0)
    )


class Test:
    def test_calc_conditional_entropies(self) -> None:
        num_item = 100
        num_topic = 10
        num_sample_item = 5
        dataset = generate_artificial_dataset(
            num_item=num_item, num_topic=num_topic, num_sample_item=num_sample_item
        )
        rnd = np.random.RandomState(0)
        psi, weights = generate_psi_bernoulli(
            dataset=dataset, topic_nums=num_topic, iter_max=20, rnd=rnd, num_new_psi=100
        )
        model = InformationGainCriterion(psi=psi, weights=weights, eps=0.0)
        h = UserHistory(indices=[0, 1, 2, 3], rs=[0, 0, 0, 1], p_list=[3])
        phi = rnd.random(len(model.psi))
        phi /= phi.sum()
        assert np.allclose(
            model.calc_conditional_entropies(
                i_list=list(range(num_item)), h=h, phi=phi
            ),
            [
                calc_conditional_entropy_slow(model=model, i=i, h=h, phi=phi)
                for i in range(num_item)
            ],
        )

    def test_eval_serendipity(self) -> None:
        rec_list = [0, 1, 2]
        topics = [0, 0, 1, 0, 1]
        y_list = [2]
        p_list = [3]
        assert eval_serendipity(
            rec_list=rec_list, topics=topics, y_list=y_list, p_list=p_list
        ) == pytest.approx(1 / 3)

    def test_generate_psi_from_embedding(self) -> None:
        num_user = 500
        num_item = 100
        num_topic = 10
        num_sample_item = 5
        num_new_psi = 100
        dataset = generate_artificial_dataset(
            num_user=num_user,
            num_item=num_item,
            num_topic=num_topic,
            num_sample_item=num_sample_item,
        )
        rnd = np.random.RandomState(0)
        psi, weights = generate_psi_embedding(
            dataset=dataset, rnd=rnd, num_new_psi=num_new_psi
        )
        assert psi.shape == (num_new_psi, num_item)
        assert weights.shape == (num_new_psi,)
