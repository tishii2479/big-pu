from typing import Optional, Union

import numpy as np
from scipy.special import logsumexp

from lib.util import UserHistory

EPS = 1e-8


class InformationGainCriterion:
    def __init__(self, psi: np.ndarray, weights: np.ndarray, eps: float):
        self.psi = psi
        self.weights = weights
        self.eps = eps

        np.clip(self.psi, EPS, 1 - EPS, out=self.psi)
        np.clip(self.weights, EPS, 1 - EPS, out=self.weights)

    def calc_r_ui(self, phi: np.ndarray) -> np.ndarray:
        return np.matmul(phi, self.psi)

    def calc_conditional_entropies(
        self,
        i_list: list[int],
        h: UserHistory,
        phi: np.ndarray,
    ) -> np.ndarray:
        _log_resps = np.log(self.weights) + self.log_bernoulli(h)
        _log_resps -= logsumexp(_log_resps, axis=-1)

        def calc_conditional_entropy_fast(i: int) -> float:
            p_1 = (phi * self.psi[:, i]).sum()
            q = []
            for r in range(2):
                log_resps = _log_resps.copy()
                log_resps += r * np.log(
                    self.eps / 2 + (1 - self.eps) * self.psi[:, i]
                ) + (1 - r) * np.log(
                    self.eps / 2 + (1 - self.eps) * (1 - self.psi[:, i])
                )
                resps = np.exp(log_resps)
                resps /= np.sum(resps)
                q.append(resps)
            return p_1 * np.sum(-q[1] * np.log2(EPS + q[1])) + (1 - p_1) * np.sum(
                -q[0] * np.log2(EPS + q[0])
            )

        ret = [calc_conditional_entropy_fast(i) for i in i_list]
        return np.array(ret)

    def transform(
        self, h: UserHistory, additional_record: Optional[tuple[int, int]] = None
    ) -> np.ndarray:
        if additional_record is not None:
            i, r = additional_record
            h = h.add_next(i, r)
        log_resps = np.log(self.weights) + self.log_bernoulli(h)
        log_resps -= logsumexp(log_resps.data, axis=-1)
        resps = np.exp(log_resps)
        return resps

    def log_bernoulli(self, h: UserHistory) -> Union[np.ndarray, float]:
        if len(h.indices) == 0:
            return 0.0
        r = np.array(h.rs)
        # P(r=1) = e/2 + (1-e)*psi
        # P(r=0) = e/2 + (1-e)*(1-psi)
        return np.sum(
            r * np.log(self.eps / 2 + (1 - self.eps) * self.psi[:, h.indices])
            + (1 - r)
            * np.log(self.eps / 2 + (1 - self.eps) * (1 - self.psi[:, h.indices])),
            axis=1,
        )
