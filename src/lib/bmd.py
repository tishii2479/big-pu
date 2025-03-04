import numpy as np
import tqdm
from scipy.special import logsumexp

EPS = 1e-8


# 混合ベルヌーイ分布
# https://qiita.com/ctgk/items/aaca9eb0438baab6c2fa
class BernoulliMixtureDistribution:
    def __init__(self, n_components: int):
        # クラスタ数
        self.n_components = n_components

    def fit(
        self, H: np.ma.MaskedArray, rnd: np.random.RandomState, iter_max: int = 100
    ) -> None:
        self.ndim = np.size(H, 1)

        self.weights = np.ones(self.n_components) / self.n_components
        self.means = rnd.uniform(0.4, 0.6, size=(self.n_components, self.ndim))
        self.means /= np.sum(self.means, axis=-1, keepdims=True)

        pbar = tqdm.tqdm(range(iter_max))
        for i in pbar:
            params = np.hstack((self.weights.ravel(), self.means.ravel()))

            stats = self._expectation(H)

            self._maximization(H, stats)
            if np.allclose(
                params, np.hstack((self.weights.ravel(), self.means.ravel()))
            ):
                print(f"early break at: {i}")
                break

            pbar.set_postfix({"log L": self._log_bernoulli(H=H).sum()})
        self.n_iter = i + 1

    def _log_bernoulli(self, H: np.ma.MaskedArray) -> np.ma.MaskedArray:
        np.clip(self.means, 1e-10, 1 - 1e-10, out=self.means)
        return np.sum(
            H[:, None, :] * np.log(self.means)
            + (1 - H[:, None, :]) * np.log(1 - self.means),
            axis=-1,
        )

    def _expectation(self, H: np.ma.MaskedArray) -> np.ma.MaskedArray:
        log_resps = np.log(self.weights) + self._log_bernoulli(H)
        log_resps -= logsumexp(log_resps.data, axis=-1)[:, None]
        resps = np.exp(log_resps)
        return resps

    def _maximization(self, H: np.ma.MaskedArray, resps: np.ma.MaskedArray) -> None:
        Nk = np.sum(resps, axis=0)

        self.weights = Nk / len(H)
        self.means = (np.ma.dot(H.T, resps) / Nk).T
