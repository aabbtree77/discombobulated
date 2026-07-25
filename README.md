## Derivative-Free Optimization

My favorite algorithm is (mu, lambda)-ES with an exponentially decaying (manually selected) step size (sigma). It solves what the best ones solve (BIPOP-aCMAES in [pycma](https://github.com/CMA-ES/pycma), RCMAES in [Minion](https://github.com/khoirulmuzakka/Minion), [RDEx_SOP](https://github.com/SichenTao/IEEE-CEC-2025-Competition-RDEx-Series/blob/main/RDEx_SOP/code/rdex-sop/RDEx.cpp)...) without stability problems and complexity.

The catch is that sometimes one needs to tune the decay (sigma_multiplier) per problem, but the default is often fine and the algorithm is so underrated.

Many think that covariance matrices and the CSA controller is what pushes the CMAES ahead, but in my experience the isotropic sampling discarding half of the worst candidates does it all even better. It avoids entrapment automatically, there is no need for increasing step sizes, the exponential schedule will do. This also scales much better with increasing lambda and D.

The whole algorithm is literally this code:

```python
import numpy as np

class CWALK:
    def __init__(self, D, x0=None, sigma=1.0, sigma_multiplier=np.exp(-1e-5), lam=None, rng=None):
        self.D = D
        self.sigma_multiplier = sigma_multiplier
        self.rng = np.random.default_rng() if rng is None else rng

        if x0 is None:
            raise ValueError("x0 must be provided by the script.")

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

        # step-size schedule
        self.sigma *= self.sigma_multiplier
```

No CSA-related step size blow ups anymore, 1e12 condition number warnings, cumulative paths, crazy empirical parameter hierarchies overfitting who knows what function in what paper/benchmark/decade, dsigma, hsigma, active/nonactive weights, nonuniform weights, BIPOP-like grid searches.

The quintessential solvable case is a wiggly function with a weak global trend such as the rotated Lunacek bi-Rastrigin (BBOB-2009 F24). This is where all the Newton/Powell methods fail, including the "multimodal" ones such as the MCS and Nomad.

Regarding the unsolvable mixtures such as the F12 in CEC2022, they are hopeless, but I suspect a simulated annealing (SA) wizard might crack them, it is just not worth it.

## Setup

```
git clone https://github.com/aabbtree77/cwalk.git
cd cwalk

uv venv

source .venv/bin/activate

uv pip install \
    numpy \
    scipy \
    matplotlib \
    cma \
    coco-experiment \
    ipython \
    minionpy
```

## Best Case Example: BBOB-2009 F24

```bash
python3 test_bbob2009.py
----------------------------------------
Backend : BBOB
Name    : bbob_f024_i01_d40
D       : 40
Bounds  : [-5.0, 5.0] for every coordinate
fopt    : 102.61
Created : 2026-07-25 13:41:05

Initial sigma : 1
evals=     10000 best_f=5.689207e+02 error=4.663107e+02 sigma=7.788e-01
evals=     20000 best_f=4.049768e+02 error=3.023668e+02 sigma=6.065e-01
evals=     30000 best_f=3.913395e+02 error=2.887295e+02 sigma=4.724e-01
evals=     40000 best_f=3.518843e+02 error=2.492743e+02 sigma=3.679e-01
evals=     50000 best_f=3.339904e+02 error=2.313804e+02 sigma=2.865e-01
evals=     60000 best_f=3.339904e+02 error=2.313804e+02 sigma=2.231e-01
evals=     70000 best_f=3.339904e+02 error=2.313804e+02 sigma=1.738e-01
evals=     80000 best_f=3.027807e+02 error=2.001707e+02 sigma=1.353e-01
evals=     90000 best_f=3.027807e+02 error=2.001707e+02 sigma=1.054e-01
evals=    100000 best_f=3.027807e+02 error=2.001707e+02 sigma=8.208e-02
evals=    110000 best_f=3.027807e+02 error=2.001707e+02 sigma=6.393e-02
evals=    120000 best_f=3.027807e+02 error=2.001707e+02 sigma=4.979e-02
evals=    130000 best_f=3.027807e+02 error=2.001707e+02 sigma=3.877e-02
evals=    140000 best_f=2.906895e+02 error=1.880795e+02 sigma=3.020e-02
evals=    150000 best_f=2.413436e+02 error=1.387336e+02 sigma=2.352e-02
evals=    160000 best_f=2.034952e+02 error=1.008852e+02 sigma=1.832e-02
evals=    170000 best_f=1.724560e+02 error=6.984605e+01 sigma=1.426e-02
evals=    180000 best_f=1.468839e+02 error=4.427388e+01 sigma=1.111e-02
evals=    190000 best_f=1.345976e+02 error=3.198756e+01 sigma=8.652e-03
evals=    200000 best_f=1.232769e+02 error=2.066693e+01 sigma=6.738e-03
evals=    210000 best_f=1.170949e+02 error=1.448487e+01 sigma=5.248e-03
evals=    220000 best_f=1.131380e+02 error=1.052795e+01 sigma=4.087e-03
evals=    230000 best_f=1.118096e+02 error=9.199592e+00 sigma=3.183e-03
evals=    240000 best_f=1.092467e+02 error=6.636718e+00 sigma=2.479e-03
evals=    250000 best_f=1.088372e+02 error=6.227180e+00 sigma=1.930e-03
evals=    260000 best_f=1.080927e+02 error=5.482740e+00 sigma=1.503e-03
evals=    270000 best_f=1.075881e+02 error=4.978138e+00 sigma=1.171e-03
evals=    280000 best_f=1.072953e+02 error=4.685341e+00 sigma=9.119e-04
evals=    290000 best_f=1.070397e+02 error=4.429728e+00 sigma=7.102e-04
evals=    300000 best_f=1.069117e+02 error=4.301656e+00 sigma=5.531e-04
evals=    310000 best_f=1.068279e+02 error=4.217944e+00 sigma=4.307e-04
evals=    320000 best_f=1.067653e+02 error=4.155310e+00 sigma=3.355e-04
evals=    330000 best_f=1.067249e+02 error=4.114893e+00 sigma=2.613e-04
evals=    340000 best_f=1.066891e+02 error=4.079094e+00 sigma=2.035e-04
evals=    350000 best_f=1.066718e+02 error=4.061836e+00 sigma=1.585e-04
evals=    360000 best_f=1.066545e+02 error=4.044539e+00 sigma=1.234e-04
evals=    370000 best_f=1.066461e+02 error=4.036095e+00 sigma=9.611e-05
evals=    380000 best_f=1.066389e+02 error=4.028945e+00 sigma=7.485e-05
evals=    390000 best_f=1.066342e+02 error=4.024204e+00 sigma=5.829e-05
evals=    400000 best_f=1.066311e+02 error=4.021071e+00 sigma=4.540e-05

Finished
----------------------------------------
Completed evaluations : 400000
Requested budget      : 400000
Best f                : 1.066310714024e+02
fopt                  : 1.026100000000e+02
Error                 : 4.021071402368e+00
Progress saved to     : progress_bbob2009_f24.csv

```

To get the zero error, increase lambda 10x, decrease the negative sigma_multiplier exponent 10x,
increase the budget 100x, but these are epsilon matters and tougher cases won't get optimized this easily.

## Worst Case Example: CEC2022-2022 F12

```bash
python3 test_cec2022.py
----------------------------------------
Backend : CEC2022
Name    : CEC2022 f12
D       : 20
Bounds  : [-100.0, 100.0] for every coordinate
fopt    : 2700.0
Created : 2026-07-25 13:41:13

Initial sigma : 20
evals=     10000 best_f=3.055804e+03 error=3.558041e+02 sigma=1.213e+01
evals=     20000 best_f=2.998437e+03 error=2.984373e+02 sigma=7.358e+00
evals=     30000 best_f=2.985656e+03 error=2.856557e+02 sigma=4.463e+00
evals=     40000 best_f=2.976510e+03 error=2.765102e+02 sigma=2.707e+00
evals=     50000 best_f=2.976106e+03 error=2.761062e+02 sigma=1.642e+00
evals=     60000 best_f=2.975896e+03 error=2.758959e+02 sigma=9.957e-01
evals=     70000 best_f=2.975771e+03 error=2.757711e+02 sigma=6.039e-01
evals=     80000 best_f=2.975741e+03 error=2.757411e+02 sigma=3.663e-01
evals=     90000 best_f=2.975728e+03 error=2.757285e+02 sigma=2.222e-01
evals=    100000 best_f=2.975720e+03 error=2.757196e+02 sigma=1.348e-01
evals=    110000 best_f=2.975719e+03 error=2.757187e+02 sigma=8.174e-02
evals=    120000 best_f=2.975718e+03 error=2.757183e+02 sigma=4.958e-02
evals=    130000 best_f=2.975718e+03 error=2.757178e+02 sigma=3.007e-02
evals=    140000 best_f=2.975718e+03 error=2.757178e+02 sigma=1.824e-02
evals=    150000 best_f=2.975718e+03 error=2.757177e+02 sigma=1.106e-02
evals=    160000 best_f=2.975718e+03 error=2.757177e+02 sigma=6.709e-03
evals=    170000 best_f=2.975718e+03 error=2.757177e+02 sigma=4.069e-03
evals=    180000 best_f=2.975718e+03 error=2.757177e+02 sigma=2.468e-03
evals=    190000 best_f=2.975718e+03 error=2.757177e+02 sigma=1.497e-03
evals=    200000 best_f=2.975718e+03 error=2.757177e+02 sigma=9.080e-04

Finished
----------------------------------------
Completed evaluations : 200000
Requested budget      : 200000
Best f                : 2.975717705667e+03
fopt                  : 2.700000000000e+03
Error                 : 2.757177056671e+02
Progress saved to     : progress_cec_f12.csv
```

Nothing works on this function, some PSOs get into 2860s, but still far away.

## References

1. Rechenberg, Ingo. Evolutionsstrategie: Optimierung technischer Systeme nach Prinzipien der biologischen evolution [Evolution Strategy: Optimization of Technical Systems According to the Principles of Biological Evolution]. Frommann-Holzboog Verlag, Stuttgart, 1973.

2. Schwefel, Hans-Paul. Numerische Optimierung von Computer-Modellen
   mittels der Evolutionsstrategie. Basel, Stuttgart, Birkhäuser, 1977.
