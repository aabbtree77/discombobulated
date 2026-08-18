<p align="center">
  <img src="bbob2009vscec2017.png" alt="bbob2009 vs cec2017 as Venn diagrams with ill-cond vs multimodality" style="width: 90%; height: auto;" />
</p>

## In Search of the Best Derivative-Free Optimization Algorithm

Do we need complex modern CMAESes/DEs?

The "CMA" part in "CMAES" is needed to solve badly scaled non-separable cost functions (ill-conditioning), see e.g. [Issue 356](https://github.com/CMA-ES/pycma/issues/356). However, if one's variables are proper, the ES part is literally this code:

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

Believe it or not, the code solves the rotated Lunacek bi-Rastrigin (F24 BBOB-2009). This is where all the intricate Newton/Powell methods fail, including some big and complex "multimodal" ones such as the MCS and Nomad.

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

## The Good: F24 BBOB-2009

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

For a fixed lambda, simply increasing budget does not lead to convergence. One needs to increase lambda and budget by the same factor. However, going for epsilon this way does not look viable. Better run with 1e4xD..1e5D budget and lambda=10D..100D to reveal a nonadversarial initial point and vicinity of the optimum, and then apply scipy's SLSQP/BFGS if epsilon matters.

lambda=D does not reach the global optimum at all. Anything interesting starts with lambda=10D.

**Restart to avoid adversarial initial points. Restarting does not improve precision/convergence. However, it is essential: unlike in CMAESes, the zero initial starting point won't lead the ES to the F24 optimum.**

Normality is not essential, but other distributions do not improve optmization. One can reach relative error O(1e-5) with

```python
self.Z = self.rng.laplace(0.0, 1.0, (self.lam, self.D))
```

or even uniform distribution:

```python
self.Z = self.rng.uniform(-3.0, 3.0, (self.lam, self.D))
```

Uniformity within [-5.0, 5.0] will still work, but [-1.0, 1.0] won't. The scale in the Laplace distribution can go up to 3.0..4.0, but no further.

The choice of the final sigma value at the end of the budget, be it 1e-4 or 1e-6, is not too critical on Rastrigins, but it is a parameter to adjust nonetheless. The choice of the initial sigma value is crucial and tied to the budget. For very large budgets sigma can be tiny and constant, otherwise we go with 10% of the biggest coordinate range (from box constraints).

Adding random sigma bursts during the optimization does not improve anything.

Expect to solve a good half of the whole BBOB-2009 with the ES (if not everything except F2, and F10-F14, but these can be done with the scipy BFGS).

## The Bad: F12 CEC-2022

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

Most of the state of the art is around 10% of the relative error. I have not seen any algorithm to go below 2900. The same story with most of the hybrids/composites in CEC-2017 and CEC-2022, and not only with them.

~~This is actually decent. The bad part about the ES is that it might be somewhat sensitive to the starting point and initial step size, and also the sigma schedule.~~

pycma BIPOP-aCMAES and Minion ARRDE are much more frugal and often better on F20 - F30 CEC2017 than the ES.

## The Ugly: F10 BBOB-2009

"F10 is the Ellipsoidal Function (a high-conditioning, unimodal function). It is hard to optimize because it features an extreme condition number (around 1e6) combined with non-separability, meaning its axes are rotated and scale at vastly different rates."

The ES is horrid when ill-conditioning takes place. It still solves these problems, but one needs to increase the budget 1000x, say to a billion of evals.

"A very rough rule of thumb is that without CMA, the number of evaluations are proportional to the condition number..." - Nikolaus Hansen, [Issue 356.](https://github.com/CMA-ES/pycma/issues/356)

That number can be proprotional to the condition number squared... The ES reaches f = -29.5 (when fopt = -54.94) on F10 BBOB-2009 in 1B evals with a constant step size 1e-3. After 1M evals it is still at f = 2.61e+07...

After some more thorough testing, see [Minion Issue 11](https://github.com/khoirulmuzakka/Minion/issues/11), it is tempting to resort to BIPOP-aCMAES or ARRDE.

This is not really ugly as it is still solvable by the ES, but the budget needs to be enormous. F20 - F30 CEC-2017 (except F22 and F27) is where things become ugly.

## The Rules of the Game

In derivative free optimization (DFO) there emerge two primary challenges: multimodality and ill-conditioning. The first one is handled with stochastic sampling such as the ES, the second - with Newton methods.

The figure above indicates that a large part of BBOB-2009, if not entirely the whole benchmark, can be covered by running any solid Newton (scipy SLSQP/BFGS) with the ES and choosing the better result.

CEC-2017 is a bigger challenge as there are a lot of functions which are both, multimodal and ill-contioned. The trouble here is that except for F22 and F27, most of these ill-conditioned multimodals are beyond the reach of any known method if we require an optimizer to get close to the global optimum with say 1% relative error in 1B evals.

Still, there are some cost functions which allow to differentiate various algorithms without a tediously slow massive testing.

I propose the following benchmark:

```markdown
| Algorithm    | F10 BBOB-2009 | F24 BBOB-2009 | F24 CEC-2017   |
| ------------ | ------------- | ------------- | -------------- |
| ES           | >1B           | <10M          | >200M (f=2800) |
| BIPOP-aCMAES | <50K          | <10M          | >200M (f=2500) |
| ARRDE        | <500K         | >200M         | >200M (f=2400) |
```

One could add F7 BBOB-2009 to make it very unfriendly for Newton/gradient methods, but let us assume zero gradient cost functions are rare.

The three functions above reveal a lot:

- ES: wipes the floor with Newton/Powell, MCS, Nomad... on Rastrigin-like multimodals. Sadly, works only with mild condition numbers (up to ~1000, would solve F18 BBOB-2009). It is somewhat sensitive w.r.t. starting points and the "sigma schedule", but does so much with so little. Very fast even in Python, no dependencies, predictable behavior. Too low tech, won't impress the champions.

- BIPOP-aCMAES (pycma CMAES), a brilliant most tested optimization algorithm on the planet, very frugal when it works, but fails on F24 - F30 CEC-2017 when there is no single coordinate system to unrotate.

- ARRDE: pushes the frontier, but demands C++ and budgets larger than 1e7xD to differentiate itself from pycma CMAES. It completely solves F24 CEC-2017 (!), yet cannot nail F25 CEC-2017. It is still better than CMAESes even on the F25. Notably, the ARRDE sustains ill-conditioning without matrices.

## Anything Better Out There?

### A Few Newest DEs

The ARRDE is outstanding with larger budgets. There is also a new DE called RDEx-SOP, but it is designed for tiny CEC-2025 budgets (2e4xD evals). It already has improvements, alternatives.

- Khoirul Faiq Muzakka et al. (2026) [Robust Differential Evolution via Nonlinear Population Size Reduction and Adaptive Restart: The ARRDE Algorithm](https://arxiv.org/abs/2511.18429v4), [Minion (github)](https://github.com/khoirulmuzakka/Minion), [Minion Issue 11](https://github.com/khoirulmuzakka/Minion/issues/11), [algolist](https://minion-py.readthedocs.io/en/latest/algolist.html)

- Sichen Tao et al. (2026) [RDEx-SOP: Exploitation-Biased Reconstructed Differential Evolution for Fixed-Budget Bound-Constrained Single-Objective Optimization](https://arxiv.org/abs/2603.27089)

- Dikshit Chauhan (2026) [DE-2LS: Differential Evolution with Late-Stage local-search for Unconstrained Single-Objective Numerical Optimization](https://arxiv.org/abs/2606.27762)

- Dikshant et al. (2026) [RDEx-CASK: Cauchy Mutation, Archive, and Stagnation Kick for RDEx-CSOP](https://arxiv.org/abs/2605.09652)

- Ryoji Tanabe and Alex Fukunaga (2020) [How Far Are We From an Optimal, Adaptive DE?](https://arxiv.org/abs/2010.01032)

### Some CMAES Papers

- LLMs are everywhere now. This one uses local minimal models to "explain" concrete optimization results after the run, which is not very useful per se, but might stimulate some thinking outside equations:

  Jill Baumann and Oliver Kramer (2024) [Towards Explainable Evolution Strategies with
  Large Language Models](https://arxiv.org/abs/2407.08331)

- Some theory indicating that the population size in the ES should be O(sqrt(D)xlog(D)):

  Lisa Schönenberger and Hans-Georg Beyer (2023) [On a Population Sizing Model for Evolution Strategies
  Optimizing the Highly Multimodal Rastrigin Function](https://pmc.ncbi.nlm.nih.gov/articles/PMC7615652/)

- Simplifications exist, but I would not recommend them, e.g.

  Zhenhua Li and Qingfu Zhang (2017) [A Simple Yet Efficient Rank One Update for Covariance
  Matrix Adaptation](https://arxiv.org/abs/1710.03996)

  See pycma's [Issue 356](https://github.com/CMA-ES/pycma/issues/356) for some of it in action, also consider adjusting the CSA rule according to pycma's [Issue 231](https://github.com/CMA-ES/pycma/issues/231).

  The problem is, for any such simplification, everything starts anew. For instance, the rank one algorithm is too sensitive/unreliable w.r.t. starting points and initial step sizes on F10 BBOB-2009, while pycma has no trouble here. The rank one update also does not work with larger lambdas as its simplistic CSA blows up the step sizes.

  None of this is valuable as we simply lose years of testing and tuning present in pycma. This is why I would also not recommend any custom implementation of CMAESes including the ones by [Minion](https://github.com/khoirulmuzakka/Minion).

  Consider F10 and F24 in BBOB-2009. These are two vastly different problems which pycma solves outstandingly well. No algorithm in the world will work well in both of the domains if the authors are not aware of such a dichotomy and have not tested their algorithm on these two specifically.

  F10 wants fewer iterations, but the right ones, it is a Newton/matrix territory. It does not want sampling/biology, bigger lambdas are detrimental.

  F24 is the opposite and smokes any Newton. It is an archetypical case for ESes, but a full CMAES with CSA is kind of irrelevant there. ES: start somewhere, sample for better directions, take a leap with a big step size in an averaged better direction. Continue doing that with an exponentially shrinking step size so that it is nearly zero by the end of the budget. Divide budget for a few restarts beforehand. This is brilliant on F24, but it does not work on F10 at all.

  pycma manages to do both with its own very fine precision instruments "CMA" and "CSA" which are no longer what is on wiki and are dangerous to simplify. Any simplification should at least be tested on each BBOB-2009 function one by one, with different step sizes, initial points, lambdas. Merely averaging over the whole BBOB-2009 suite a few random runs does not reveal damages and weaknesses introduced by simplification.

### Some Advances in Simulated Annealing and Memetics

scipy includes an algorithm called "dual annealing" (DA) which runs BFGS as local search. Scroll down [this code](https://github.com/sgubianpm/sdaopt/blob/master/sdaopt/_sda.py) for all the references. DA looks visible also in the R community.

I did not get anything from DAs on CEC2017 F24 - F30 in D=20.

Minion includes [one interesting comparison](https://minion-py.readthedocs.io/en/stable/l_bfgs_b_notebook.html) between the ARRDE, numerous BFGS implementations, and two DA implementations. It turns out that Minion's DA is worse than scipy DA, except on F17 and F26 (CEC-2017). The ARRDE is clearly better than anything on: F10, F12, F17 (somewhat), F21, F22, F24, F26, F28, and F30. However, in the rest of the cases DAs are close and on F25 scipy DA = 2600 (!), the ARRDE and the rest are close and only around 2900. It is the first time I see the problem where the ARRDE could be clearly worse.

I was not able to get anything with DAs on F24 BBOB-2009 in D=40, and on F24 - F30 CEC-2017 in D=20. Also tried [this code](https://github.com/DawitLam/Improvements_to_Dual_Annealing_in_SciPy) to no avail.

Minion's result in D=10 is somewhat contradictory, but it depends on the starting point and D=10 may not generalize to D=20. According to [Minion's notebook](https://minion-py.readthedocs.io/en/stable/l_bfgs_b_notebook.html), the ARRDE solves F26 CEC-2017 in D=10 in less than 100K evals (reaching 2600). In my run, for the zero starting point, SEED = 20260815, the ARRDE reaches 2800 in 2B evals (F26 CEC-2017 D=20).

A note on memetics, e.g. the use of BFGS inside some global search. About half of the problems on BBOB-2009 and CEC-2017 are completely solvable with Newton methods, and there are some multimodals which are solvable with Newton via restarts. On trully difficult problems, such as F24 - F30 CEC-2017 in higher dimensions such as D=20 this approach does not seem to lead anywhere.

In a way, CMAES is the most tuned and tested memetics, the best one can do when combining Newton/curvature with stochastics, or the most improved ES regarding adaptive step sizes and curvature exploitation.

### Some BBOB-2009 Tests

There are a lot of tests reported, but I would recommend running things yourself, as it is often much faster and there is no need to decipher various missing details and rationale in some super terse for no reason reports.

One exception is Baeysian Optimization as it is complex and annoyingly slow to run. I would avoid this domain entirely as those tiny budgets lack stability, and there is no convergence/critical mass on any existing algorithm despite the field energing already in 1970s...

**Most importantly, if an algorithm has no code used by masses, it does not exist.**

- Youssef Diouane et al. (2022) [TREGO: a Trust-Region Framework for Efficient Global Optimization](https://arxiv.org/abs/2101.06808)

- Zachary Hoffman and Steve Huntsman (2022) [Benchmarking an algorithm for expensive high-dimensional
  objectives on the bbob and bbob-largescale testbeds](https://hal.science/hal-03665291v1/file/GECCOarXiv2022.pdf)

- Ryoji Tanabe (2022) [Benchmarking the Hooke-Jeeves Method, MTS-LS1, and BSrr on
  the Large-scale BBOB Function Set](https://arxiv.org/abs/2204.13284)

- Nikolaus Hansen (2019) [A Global Surrogate Assisted CMA-ES](https://inria.hal.science/hal-02143961v1/document), [pycma (github)](https://github.com/CMA-ES/pycma), [pycma Issue 356](https://github.com/CMA-ES/pycma/issues/356)

- Nikolaus Hansen at al. (2019) [Real-Parameter Black-Box Optimization Benchmarking 2009: Noiseless Functions Definitions](https://inria.hal.science/inria-00362633v2/document)

- Konstantinos Varelas (2019) [Benchmarking Large Scale Variants of CMA-ES and L-BFGS-B
  on the bbob-largescale Testbed](https://inria.hal.science/hal-02160106/file/wksp213s2-file1.pdf)

- Aurore Blelly at al. (2018) [Stopping Criteria, Initialization, and Implementations of
  BFGS and their Effect on the BBOB Test Suite](https://inria.hal.science/hal-01811588/file/workshop_paper-authorversion.pdf)

## Classics

Most of the early algorithms did not survive the test of time. Some analysis tools, boundary handling, a few test functions did.

Modern methods won't live long either. Optimization is technology. However, we still share the same excitement as people in the 1960s when the algorithm finds the optimum.

Classics should neither be underestimated nor overestimated. A lot of elegance and insights there, but also not enough data/testing, obsession with math, proofs, belief structures. The whole BFGS and convex optimization saga stretching from Fletcher and Powell circa 1963 and continuing with Nesterov and all these endless SIAM, NIPS... reports. Practical Bayes starts already in 1970s, if not earlier.

Sadly, math is most often about writing too much about too little, but the client (government) pays per equation/lemma, so the reports tend to grow in time, and the latest Fields medals are now spanning 200 pages on arXiv due to that feedback.

**People do not read/review/test much, but they bow to complexity and belief structures.**

Add cliping to the Barzilai–Borwein method for gradient descent, then write the whole paper/thesis about the convergence of the modified version when the whole method is too niche to spend this much time, be it a clipped/stabilized step size or not.

Endless variations around Newton, stochastic sampling, surrogates, mostly without proper testing at all. These types of works continue to appear en masse even in 2020s, wonder if I am the only one who is reading them...

H. H. Rosenbrock (1960) An Automatic Method for Finding the Greatest or Least Value of a Function

R. Fletcher and M.J.D. Powell (1963) A Rapidly Convergent Descent Method for Minimization

L. A. Rastrigin (1965) Solution of inverse problems by statistical optimization methods

M. J. Box (1966) A Comparison of Several Current Optimization Methods, and the use of Transformations in Constrained Problems

M.A. Schumer and K. Steiglitz (1968) Adaptive step size random search

L.J. White and R.G. Day (1971) An Evaluation of Adaptive Step-Size Random Search

J. Mockus, V. Tiesis, A. Zilinskas (1978) The Application of Bayesian Methods for Seeking the Extremum

J. Bernussou and J. Geromel (1981) An easy way to find gradient matrix of composite matricial functions

...

[CMAES 1996 - 2014](https://cma-es.github.io/)

...

finally escaping calculus, probabilities, and linear algebra:

Khoirul Faiq Muzakka, Ahsani Hafizhu Shali, Haris Suhendar, Sören Möller, Martin Finsterbusch (2026) [Robust Differential Evolution via Nonlinear Population Size Reduction and Adaptive Restart: The ARRDE Algorithm](https://arxiv.org/abs/2511.18429)

## CMAES Evolution

CMAES is incredible when one sees how much effort surrounds the algorithm:

[https://cma-es.github.io/](https://cma-es.github.io/)

Sadly, the CMAES evolution seems to end around 2014, and with it, the whole ES story.

## P.S.

I got sidetracked. The main idea was to share a surprise pulled by the basic ES on Rastrigins, but this superpower did not generalize to ill-conditioned functions. Use pycma BIPOP-aCMAES for tiny budgets and Minion ARRDE to push the limits.
