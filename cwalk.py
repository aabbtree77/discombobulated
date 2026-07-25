import numpy as np

class CWALK:
    def __init__(self, D, x0=None, sigma=1.0, sigma_multiplier=np.exp(-1e-5), lam=None, rng=None):
        self.D = D
        self.sigma_multiplier = sigma_multiplier
        self.rng = np.random.default_rng() if rng is None else rng

        if x0 is None:
            raise ValueError("x0 must be provided by the driver script.")

        self.xmean = np.asarray(x0, dtype=float).copy()
        self.sigma = float(sigma)

        self.lam = 10 * D if lam is None else int(lam)
        self.mu = self.lam // 2

        self.best_x = self.xmean.copy()
        self.best_f = np.inf

    # ------------------------------------------------------------
    # ASK
    # ------------------------------------------------------------
    def ask(self):
        Z = self.rng.standard_normal((self.lam, self.D))
        X = self.xmean + self.sigma * Z
        return X

    # ------------------------------------------------------------
    # TELL
    # ------------------------------------------------------------
    def tell(self, X, fitness):
        X = np.asarray(X, dtype=float)
        fitness = np.asarray(fitness, dtype=float)

        order = np.argsort(fitness)

        # best-so-far
        if fitness[order[0]] < self.best_f:
            self.best_f = float(fitness[order[0]])
            self.best_x = X[order[0]].copy()

        # update mean
        self.xmean = np.mean(X[order[:self.mu]], axis=0)

        # placeholder step-size schedule
        self.sigma *= self.sigma_multiplier
        #self.sigma = 1e-1 
