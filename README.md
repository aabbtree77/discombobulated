> «Я  
> достаю  
> из широких штанин  
> дубликатом  
> бесценного груза...»

## Derivative-Free Optimization

(mu, lambda)-ES with an exponentially decaying step size and a larger lambda (10D..100D) is a very underrated algorithm. It raises the key question:

**Do we need the CMAES at all?**

It turns out, the CMA part in the CMAES is needed to solve badly scaled variables (ill-conditioning), see e.g. [Issue 356](https://github.com/CMA-ES/pycma/issues/356). (mu, lambda)-ES completely fails on such wildly scaled rotated ellipsoids. Consider this as its adversarial attack, nothing more. If we ignore artificial ill-conditioned rotations, fixed budgets, epsilon targets, I still very much prefer (mu, lambda)-ES.

The whole (mu, lambda)-ES algorithm is literally this code:

```python
import numpy as np

class CWALK:
    def __init__(self, D, x0=None, sigma=1.0, lam=None, rng=None):
        self.D = D
        self.rng = np.random.default_rng() if rng is None else rng

        if x0 is None:
            raise ValueError("x0 must be provided by the driver script.")

        self.xmean = np.asarray(x0, dtype=float).copy()
        self.sigma = float(sigma)

        self.lam = 100 * D if lam is None else int(lam)
        self.mu = self.lam // 2

        self.best_x = self.xmean.copy()
        self.best_f = np.inf

    # ------------------------------------------------------------
    # ASK
    # ------------------------------------------------------------
    def ask(self):
        self.Z = self.rng.standard_normal((self.lam, self.D))
        X = self.xmean + self.sigma * self.Z
        return X

    # ------------------------------------------------------------
    # TELL
    # ------------------------------------------------------------
    def tell(self, X, fitness, sigma):
        X = np.asarray(X, dtype=float)
        fitness = np.asarray(fitness, dtype=float)
        order = np.argsort(fitness)

        # best-so-far
        if fitness[order[0]] < self.best_f:
            self.best_f = float(fitness[order[0]])
            self.best_x = X[order[0]].copy()

        # update mean
        self.xmean = np.mean(X[order[:self.mu]], axis=0)
        self.normz = np.linalg.norm(np.mean(self.Z[order[:self.mu]], axis=0))

        # update sigma
        if sigma is not None:
            self.sigma = sigma
```

In pure search (no iterative remappings to solve ill-conditioning), (mu=lambda/2, lambda=100D)-ES can even be on par with BIPOP-aCMAES. It won't solve new problems magically, but it removes so much complexity. Adjust the cost function evaluation budget size, sometimes the final step size multiplier's epsilon value, this is it.

One key solvable tough case is the rotated Lunacek bi-Rastrigin (BBOB-2009 F24). This is where all the Newton/Powell methods fail, including the "multimodal" ones such as the MCS and Nomad.

Regarding mixtures such as the F12 in CEC2022, these are hopeless in theory, but optimized adequately when speaking pragmatically.

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

## The Good: BBOB-2009 F24

```bash
python3 test_bbob2009.py

Problem
----------------------------------------
Backend : BBOB
Name    : bbob_f024_i01_d40
D       : 40
Bounds  : [-5.0, 5.0] for every coordinate
fopt    : 102.61
Created : 2026-07-27 00:29:20

Initial sigma : 1
evals=    100000 best_f=5.299968e+02 error=4.273868e+02 sigma=8.017e-01
evals=    200000 best_f=4.000308e+02 error=2.974208e+02 sigma=6.368e-01
evals=    300000 best_f=3.666502e+02 error=2.640402e+02 sigma=5.058e-01
evals=    400000 best_f=3.473649e+02 error=2.447549e+02 sigma=4.018e-01
evals=    500000 best_f=3.265643e+02 error=2.239543e+02 sigma=3.192e-01
evals=    600000 best_f=3.240048e+02 error=2.213948e+02 sigma=2.535e-01
evals=    700000 best_f=3.240048e+02 error=2.213948e+02 sigma=2.014e-01
evals=    800000 best_f=3.240048e+02 error=2.213948e+02 sigma=1.600e-01
evals=    900000 best_f=3.240048e+02 error=2.213948e+02 sigma=1.271e-01
evals=   1000000 best_f=3.240048e+02 error=2.213948e+02 sigma=1.009e-01
evals=   1100000 best_f=3.240048e+02 error=2.213948e+02 sigma=8.017e-02
evals=   1200000 best_f=3.240048e+02 error=2.213948e+02 sigma=6.368e-02
evals=   1300000 best_f=3.121135e+02 error=2.095035e+02 sigma=5.058e-02
evals=   1400000 best_f=3.013288e+02 error=1.987188e+02 sigma=4.018e-02
evals=   1500000 best_f=2.701156e+02 error=1.675056e+02 sigma=3.192e-02
evals=   1600000 best_f=2.091583e+02 error=1.065483e+02 sigma=2.535e-02
evals=   1700000 best_f=1.721276e+02 error=6.951763e+01 sigma=2.014e-02
evals=   1800000 best_f=1.526473e+02 error=5.003727e+01 sigma=1.600e-02
evals=   1900000 best_f=1.316534e+02 error=2.904339e+01 sigma=1.271e-02
evals=   2000000 best_f=1.206309e+02 error=1.802090e+01 sigma=1.009e-02
evals=   2100000 best_f=1.119537e+02 error=9.343675e+00 sigma=8.017e-03
evals=   2200000 best_f=1.104614e+02 error=7.851370e+00 sigma=6.368e-03
evals=   2300000 best_f=1.074570e+02 error=4.847032e+00 sigma=5.058e-03
evals=   2400000 best_f=1.054741e+02 error=2.864093e+00 sigma=4.018e-03
evals=   2500000 best_f=1.047269e+02 error=2.116855e+00 sigma=3.192e-03
evals=   2600000 best_f=1.039641e+02 error=1.354105e+00 sigma=2.535e-03
evals=   2700000 best_f=1.035237e+02 error=9.137222e-01 sigma=2.014e-03
evals=   2800000 best_f=1.032888e+02 error=6.788106e-01 sigma=1.600e-03
evals=   2900000 best_f=1.031278e+02 error=5.178318e-01 sigma=1.271e-03
evals=   3000000 best_f=1.029578e+02 error=3.478413e-01 sigma=1.009e-03
evals=   3100000 best_f=1.028953e+02 error=2.853422e-01 sigma=8.017e-04
evals=   3200000 best_f=1.028682e+02 error=2.582219e-01 sigma=6.368e-04
evals=   3300000 best_f=1.028322e+02 error=2.222083e-01 sigma=5.058e-04
evals=   3400000 best_f=1.028317e+02 error=2.216639e-01 sigma=4.018e-04
evals=   3500000 best_f=1.028190e+02 error=2.090154e-01 sigma=3.192e-04
evals=   3600000 best_f=1.028112e+02 error=2.012098e-01 sigma=2.535e-04
evals=   3700000 best_f=1.028073e+02 error=1.973345e-01 sigma=2.014e-04
evals=   3800000 best_f=1.028049e+02 error=1.948547e-01 sigma=1.600e-04
evals=   3900000 best_f=1.028034e+02 error=1.934189e-01 sigma=1.271e-04
evals=   4000000 best_f=1.028017e+02 error=1.917442e-01 sigma=1.009e-04

Finished
----------------------------------------
Completed evaluations : 4000000
Requested budget      : 4000000
Best f                : 1.028017441860e+02
fopt                  : 1.026100000000e+02
Error                 : 1.917441860190e-01
Progress saved to     : progress_bbob2009_f24.csv

```

It takes 1e4xD evals to reach 0.2% relative error. Reduce budget 10x, reduce lambda 10x, relative error will increase 10x.
Increase budget 10x, increase lambda 10x, relative error will decrease 100x!

However, to get a guaranteed convergence is not easy. For 1e7xD relative error is still O(1e-5).

For a fixed lambda, simply increasing budget does not lead to convergence. One needs to increase lambda and budget by the same factor. However, going for epsilon this way does not look viable. Better run with 1e4xD..1e5D budget and lambda=10D..100D to reveal a nonadversarial initial point and vicinity of the optimum, and then apply scipy's SLSQP or BFGS if epsilon matters.

lambda=D does not reach the global optimum at all. Anything interesting starts with lambda=10D.

Restart only to avoid adversarial initial points. Restarting does not improve precision/convergence that much. However, it is essential. Adding bursts to sigma does not allow the algorithm to escape adversarial local optima.

Normality is not essential, but other distributions do not improve optmization. One can reach relative error O(1e-5) with

```python
self.Z = self.rng.laplace(0.0, 1.0, (self.lam, self.D))
```

or even uniform distribution:

```python
self.Z = self.rng.uniform(-3.0, 3.0, (self.lam, self.D))
```

The parameters are not very critical, but they should be reasonable. Uniformity within [-5.0, 5.0] will still work, but [-1.0, 1.0] won't. The scale in the Laplace distribution can go up to 3.0..4.0, but no further.

## The Bad: CEC-2022 F12

```bash
python3 test_cec2022.py

Problem
----------------------------------------
Backend : CEC2022
Name    : CEC2022 f12
D       : 20
Bounds  : [-100.0, 100.0] for every coordinate
fopt    : 2700.0
Created : 2026-07-27 00:53:02

Initial sigma : 20
evals=    100000 best_f=3.078020e+03 error=3.780203e+02 sigma=1.274e+01
evals=    200000 best_f=2.995935e+03 error=2.959349e+02 sigma=8.036e+00
evals=    300000 best_f=2.986139e+03 error=2.861390e+02 sigma=5.070e+00
evals=    400000 best_f=2.986139e+03 error=2.861390e+02 sigma=3.199e+00
evals=    500000 best_f=2.986139e+03 error=2.861390e+02 sigma=2.019e+00
evals=    600000 best_f=2.985843e+03 error=2.858433e+02 sigma=1.274e+00
evals=    700000 best_f=2.985738e+03 error=2.857379e+02 sigma=8.036e-01
evals=    800000 best_f=2.985697e+03 error=2.856973e+02 sigma=5.070e-01
evals=    900000 best_f=2.985667e+03 error=2.856670e+02 sigma=3.199e-01
evals=   1000000 best_f=2.985649e+03 error=2.856490e+02 sigma=2.019e-01
evals=   1100000 best_f=2.985646e+03 error=2.856463e+02 sigma=1.274e-01
evals=   1200000 best_f=2.985643e+03 error=2.856425e+02 sigma=8.036e-02
evals=   1300000 best_f=2.985642e+03 error=2.856422e+02 sigma=5.070e-02
evals=   1400000 best_f=2.985642e+03 error=2.856419e+02 sigma=3.199e-02
evals=   1500000 best_f=2.985642e+03 error=2.856417e+02 sigma=2.019e-02
evals=   1600000 best_f=2.985642e+03 error=2.856417e+02 sigma=1.274e-02
evals=   1700000 best_f=2.985642e+03 error=2.856416e+02 sigma=8.036e-03
evals=   1800000 best_f=2.985642e+03 error=2.856416e+02 sigma=5.070e-03
evals=   1900000 best_f=2.985642e+03 error=2.856416e+02 sigma=3.199e-03
evals=   2000000 best_f=2.985642e+03 error=2.856416e+02 sigma=2.019e-03

Finished
----------------------------------------
Completed evaluations : 2000000
Requested budget      : 2000000
Best f                : 2.985641603543e+03
fopt                  : 2.700000000000e+03
Error                 : 2.856416035435e+02
Progress saved to     : progress_cec2022_f12.csv
```

Most of the state of the art is here around 10% relative error. I have not seen any algorithm to go below 2900. The same story with all the hybrids/composites in CEC-2017 and CEC-2022, and not only with them, for any existing algorithm.

## The Ugly: BBOB-2009 F10

"F10 is the Ellipsoidal Function (a high-conditioning, unimodal function). It is hard to optimize because it features an extreme condition number (around 1e6) combined with non-separability, meaning its axes are rotated and scale at vastly different rates."

ES does not solve it, gets into 1e5, pushes it down to 1e2, too slow to reach fopt = -5.494000e+01.

"A very rough rule of thumb is that without CMA, the number of evaluations are proportional to the condition number of the function divided by 100 when the condition number is larger than 100." - Nikolaus Hansen, [Issue 356.](https://github.com/CMA-ES/pycma/issues/356)

The proportion coefficient can be 1e4 and the ES still won't solve F10 (in 100M evals).

Solid implementations of Newton methods (scipy SLSQP/BFGS) tend to solve the ill-conditioned section of BBOB-2009 F10 - F14, see "A Global Surrogate Assisted CMA-ES" by Nikolaus Hansen (GECCO 2019).

One may fancy ideas to combine Newton with ESes, to cope with ill-conditioning, to acquire more precision, but this is not viable.

Even the smartest implementations of Newton methods face plant already on BBOB-2009 F7 (moderate ill-conditioning, but zero gradients in large portions of the domain). They do not help on multimodals even with heavy restarts. Frankly they are not good without **analytical gradients**.

## Conclusion

After some more thorough testing, see [Minion Issue 11](https://github.com/khoirulmuzakka/Minion/issues/11), I am convinced that the complexity of BIPOP-aCMAES or ARRDE is justified. These are much stronger algorithms. The ES solves a tough F24 BBOB-2009, but falters already on a rotated ill-conditioned ellipsoid F10 BBOB-2009. Endless simplifications out there also do not work as portrayed.

Regarding the two, BIPOP-aCMAES or the ARRDE, I am now slightly on the latter side as it solves F24 CEC-2017 completely, while somewhat screwing up F24 BBOB-2009. I believe it is easier to improve the performance of the ARRDE on BBOB-2009 than BIPOP-aCMAES on CEC-2017, but both are very capable, and BBOB-2009/Hansen's works remain a powerful instrument to study DFO algorithms.

The ARRDE is somewhat more linear and stable, solves ill-conditioning without matrices, but it is often 10x more evaluation-hungry. Anything below max(1e7xD, 200M) evals is BIPOP-aCMAES for me, but above that, I would take the ARRDE.

## References

Khoirul Faiq Muzakka et al. (2026) [Robust Differential Evolution via Nonlinear Population Size Reduction and Adaptive Restart: The ARRDE Algorithm](https://arxiv.org/abs/2511.18429v4), [Minion (github)](https://github.com/khoirulmuzakka/Minion)

Nikolaus Hansen (2019) [A Global Surrogate Assisted CMA-ES](https://inria.hal.science/hal-02143961v1/document), [pycma (github)](https://github.com/CMA-ES/pycma)

Aurore Blelly at al. (2018) [Stopping Criteria, Initialization, and Implementations of
BFGS and their Effect on the BBOB Test Suite](https://inria.hal.science/hal-01811588/file/workshop_paper-authorversion.pdf)

Nikolaus Hansen at al. (2019) [Real-Parameter Black-Box Optimization Benchmarking 2009: Noiseless Functions Definitions](https://inria.hal.science/inria-00362633v2/document)
