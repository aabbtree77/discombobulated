> Three wise men from freezing North<br>
> Keep telling me and holding forth<br>
> The metal will not bring a yield<br>
> The game's not worth the candle, nor the labor's field<br> 
><br>
> But I am planting my aluminium cucumbers, ah-ah<br>
> Right on a tarpaulin field<br>
> Yes I am planting my aluminium cucumbers, ah-ah<br>
> Right on a tarpaulin field<br> 
><br> 
> [\- AI-MUSIC KANYE WEST ft. ВИКТОР ЦОЙ - ALUMINIUM CUCUMBERS](https://www.youtube.com/watch?v=980EpMVJ6Pg&list=RD980EpMVJ6Pg&start_radio=1)

<br>

<br>

<p align="center">
  <img src="bbob2009vscec2017.png" alt="bbob2009 vs cec2017 as Venn diagrams with ill-cond vs multimodality" style="width: 90%; height: auto;" />
</p>

## In Search of the Best Derivative-Free Optimization Algorithm

Do we need complex modern optimization algorithms?

The "CMA" part in "CMAES" solves badly scaled non-separable cost functions (ill-conditioning, stiffness), see e.g. [Issue 356](https://github.com/CMA-ES/pycma/issues/356). However, if one's variables are proper, the ES part is literally this code:

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

Believe it or not, the code solves the rotated Lunacek bi-Rastrigin (F24 BBOB-2009). This is where all the intricate Newton/Powell methods fail, including the MCS and Nomad.

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

It takes 1e4xD evals to reach 0.2% relative error.

- Reduce budget 10x, reduce lambda 10x, relative error will increase 10x.
- Increase budget 10x, increase lambda 10x, relative error will decrease 100x!

For 1e7xD evals the relative error is still O(1e-5).

lambda=D does not reach the global optimum at all. Anything interesting starts with lambda=10D.

Restart to avoid adversarial initial points. Restarting does not improve precision/convergence. However, it is essential: unlike in CMAES, a zero does not lead to the optimum.

Normality is not essential, but other distributions do not improve optmization.

One can reach relative error O(1e-5) with

```python
self.Z = self.rng.laplace(0.0, 1.0, (self.lam, self.D))
```

or even uniform distribution:

```python
self.Z = self.rng.uniform(-3.0, 3.0, (self.lam, self.D))
```

Uniformity within [-5.0, 5.0] will still work, but [-1.0, 1.0] won't. The scale in the Laplace distribution can go up to 3.0..4.0, but no further.

The choice of the final sigma value at the end of the budget, be it 1e-4 or 1e-6, is not too critical. The choice of the initial sigma value is. For very large budgets sigma can be tiny and constant, otherwise we go with 10% of the biggest coordinate range (from box constraints).

Adding random sigma bursts during the optimization does not improve anything.

Expect to solve a good half of the whole BBOB-2009 with the ES (if not everything except F2, F10 - F14, but these can be done with scipy BFGS).

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

I have not seen any algorithm to go below 2900.

## The Ugly: F10 BBOB-2009

"F10 is the Ellipsoidal Function (a high-conditioning, unimodal function). It is hard to optimize because it features an extreme condition number (around 1e6) combined with non-separability, meaning its axes are rotated and scale at vastly different rates."

The ES becomes brittle with ill-conditioning.

"A very rough rule of thumb is that without CMA, the number of evaluations are proportional to the condition number..." - Nikolaus Hansen, [Issue 356.](https://github.com/CMA-ES/pycma/issues/356)

That number can be proprotional to the condition number squared... The ES reaches f = -29.5 (when fopt = -54.94) on F10 BBOB-2009 in 1B evals with a constant step size 1e-3. After 1M evals it is still at f = 2.61e+07...

After some more thorough testing, see [Minion Issue 11](https://github.com/khoirulmuzakka/Minion/issues/11), it is tempting to resort to ARRDE.

## Rules of the Game

So this is all about multimodality, ill-conditioning, dimensions, and evalutation budgets.

The figure above indicates that a large part of BBOB-2009, if not entirely the whole benchmark, can be covered by running any solid Newton (scipy SLSQP/BFGS) with the ES and choosing the better result.

CEC-2017 is a bigger challenge with functions which are both: multimodal and ill-contioned. Moreover, with a few exceptions, its F21-F30 functions are not solvable by any known method.

Solvable = getting close to the global optimum within, say, 1% relative error in 1B evals in at least D=20. Everything is easy in D=10.

A preliminary view:

```markdown
| Algorithm    | F10 BBOB-2009 D=40 | F24 BBOB-2009 D=40 | F24 CEC-2017 D=20 | F25 CEC-2017 D=20 |
| ------------ | ------------------ | ------------------ | ----------------- | ----------------- |
| ES           | >1B                | <10M f=102.61      | >200M f=2800      | >1B f=2900        |
| BIPOP-aCMAES | <50K               | <10M f=102.61      | =200M f=2500      | =200M f=2899      |
| ARRDE        | <500K              | =200M f=1.4895     | =200M f=2400      | =1B f=2600        |
```

- ES: wipes the floor with Newton/Powell on Rastrigin-like multimodals. Outstanding only with mild condition numbers (up to ~1000, still solves F18 BBOB-2009). It is sensitive w.r.t. starting points, but this is nothing serious.

- BIPOP-aCMAES (pycma CMAES), used to be the best, fails on F21 - F30 CEC-2017 when there is no single coordinate system to rescale-unrotate. It does not solve F25 CEC-2017 already in D=10.

- ARRDE: pushes the frontier, but it is some hairy C++ list processing and becomes interesting only with budgets larger than 1e7xD. It completely solves F24 CEC-2017 (!), yet cannot nail F25 CEC-2017. Notably, ARRDE sustains ill-conditioning without matrices, and does it much better than CMAES.

Scroll down for more benchmarking on CEC-2017.

## Anything Better Out There?

- DEs lose their quality very rapidly w.r.t. increasing D beyond 10.

- ES/CMAES are worse at ill-conditioning than DEs.

### CMAES Mods?

- Lots of CMAES complications exist, but I could not get anything from them so far, e.g.

  Dimitar Nedanovski et al. (2026) [MSC-CMA-ES: Structure-Aware Restarts for CMA-ES via Cyclic Nearest-Better Basin Discovery](https://arxiv.org/abs/2606.15830), [Github](https://github.com/snenovgmailcom/cma_es_project/tree/main)

  It does not reach f = 2400 on F24 CEC-2017 at all and does not look any different than BIPOP-aCMAES, despite the paper hinting that it could be interesting on the CEC-2017 composites. Very slow even with the C++ acceleration.

  Default parameters, seed = 20260825, F24 CEC-2017 D=20 got precisely f = 2500 in 200M evals, which took about 5 hours to run (a single optimization) on i7 gen4 16GB RAM. The C++ acceleration is only for clustering, pycma CMAES runs inside MSC-CMA-ES.

- Another one bites the dust:

  Khoirul Faiq Muzakka et al. (2026) [RCMAES: A Robust CMA-ES Variant for CEC2026 Competition](https://arxiv.org/abs/2604.27138)

  No difference, except that it is much faster to test than pycma and MSC-CMA-ES and is integrated into [Minion](https://github.com/khoirulmuzakka/Minion), though we did have libcmaes before.

- Simplifications exist, but I am not sure what to do with them, e.g.

  Zhenhua Li and Qingfu Zhang (2017) [A Simple Yet Efficient Rank One Update for Covariance
  Matrix Adaptation](https://arxiv.org/abs/1710.03996)

  See pycma's [Issue 356](https://github.com/CMA-ES/pycma/issues/356) for some of it in action, also consider adjusting the CSA according to pycma [Issue 231](https://github.com/CMA-ES/pycma/issues/231).

  In D up to 10, ARRDE is vastly superior on tougher challenges with very large budgets, but it fails on the CEC-2017 composites in D=20 despite a few exceptions.

### Dual Annealing?

scipy includes an algorithm called "dual annealing" (DA) which runs BFGS as local search. Scroll down [this code](https://github.com/sgubianpm/sdaopt/blob/master/sdaopt/_sda.py) for all the references. DA got visible first in the R community.

I did not get anything from DAs on CEC2017 F21 - F30 in D=20. Also tried [this code](https://github.com/DawitLam/Improvements_to_Dual_Annealing_in_SciPy) to no avail.

Minion includes [one interesting comparison](https://minion-py.readthedocs.io/en/stable/l_bfgs_b_notebook.html) between the ARRDE, numerous BFGS implementations, and two DA implementations. It turns out that Minion's DA is worse than scipy DA, except on F17 and F26 (CEC-2017). ARRDE is clearly better than anything on: F10, F12, F17 (somewhat), F21, F22, F24, F26, F28, and F30. However, in the rest of the cases DAs are close and on F25 scipy DA = 2600 (!), ARRDE and the rest are close and only around 2900. It is the first time I see the problem where the ARRDE could be clearly worse. WTF?!

D=10 does not generalize to D=20 at all. According to [Minion's notebook](https://minion-py.readthedocs.io/en/stable/l_bfgs_b_notebook.html), ARRDE solves F26 CEC-2017 in D=10 in fewer than 100K evals (reaching 2600). In my runs, for the zero starting point, seed = 20260815, ARRDE reaches only 2800 in 2B evals (F26 CEC-2017 D=20). Night and day.



### Testing ARRDE

F25 CEC-2017 D=20, 1B evals: ARRDE f=2700, seed=20260829, single run takes 4.68 hours on i7 gen 4 16GB RAM.

F28 CEC-2017 D=20, <=200M evals: ARRDE f=3000, BIPOP-aCMAES f=3100; fopt = 2800.

F24 CEC-2017:

- restarts are critical, at least O(10) are needed,

- 200M evals are sufficient to solve the problem completely (f=fopt=2400) when the seed is good.

F25 CEC-2017:

- restarts may not be needed,

- 1B evals still do not solve the problem (f=2700, fopt=2500).

Making significant progress on F24 does not imply its transfer on F25 and vice versa.

ARRDE can be inconsistent w.r.t. increasing budgets, e.g. F25 CEC-2017 D=20 seed=20260829:

- 500M evals: f=2800,
- 1B evals: f=2700,
- 2B evals: f=2800.

Also, when setting a budget say to 200M, the first 10M evals can lead to a better result than rerunning
the whole thing with 10M evals or 20M evals, and this is not so predictable due to population size reduction and global phases.

On F24 CEC-2017, ARRDE can be very picky with seeding or whether zero is included in the initial population, unless it is used in the special restart mode, read below.

ARRDE adds (to the jSO algorithm) global phases with some intricate refinement machinery via merged local intervals acting as an implementation of [Tabu Search](https://github.com/zarankumar/tabu-search).

Decent up to D=10, afterwards every problem becomes a special case. Might work spectacularly (F24 CEC-2017 D=20), but may also lead to nowhere (F25 CEC-2017 D=20). Generally very bad with increasing D>10 as the experiments with F24 BBOB-2009 for D=10, 20,40, 100 would show.

ARRDE also becomes extremely slow to run beyond 200M evals.

## CEC-2017 Composites

CEC-2017 was a step forward compared to CEC-2014 and BBOB-2009. It added multiple ill-conditioned matrices and revealed the simplest problems not amenable to any modern technology. They rule out any existing ES and DE.

The composites F21-F30 are the hardest cost functions of the benchmark. Do not run anything on F25 in D=20 without being prepared to spend years going nowhere. All the algorithms of [Minion](https://github.com/khoirulmuzakka/Minion) fail there. ARRDE is the only exception, but also needs tinkering and solves only very few cases in D=20. Forget about D=100.

What are these challenges?

Firstly, the subsets of already deceptive functions (in each given list) are mixed into hybrids.

In turn, these hybrids are rotated and scaled with different matrices and further mixed with some distance based weights.

Any single function is often already deceptive: multimodal, sometimes non-differentiable. It can already be ill-conditioned before being mixed into a hybrid. The latter in turn will get their own ill-conditioning. The key is ill-conditioning with multiple matrices in higher D>10. This is what kills modern ESes and DEs.

There are separate research works with a deep focus on some components, see e.g. [Happy Cat Function](https://www.researchgate.net/publication/234024034_HappyCat_-_A_Simple_Function_Class_Where_Well-Known_Direct_Search_Algorithms_Do_Fail) which is a deceptive ridge generator designed to obfuscate ES and DE searches. 

Deceptiveness is less severe than multiple ill-condtioned matrices and increasing D>10.

Initial functions/components extracted from multilayer mixing (which is still only 2 layers, more or less, in CEC-2017):

F30:

1. Rastrigin's Function
2. Griewank's Function
3. Schaffer’s F6 Function
4. Rosenbrock's Function
5. Katsuura Function
6. Ackley's Function
7. Expanded Griewank's plus Rosenbrock's Function
8. Modified Schwefel's Function

F29:

1. Rastrigin's Function
2. Griewank's Function
3. Schaffer’s F6 Function
4. Rosenbrock's Function
5. High Conditioned Elliptic Function
6. Ackley's Function
7. HGBat Function
8. Discus Function
9. Bent Cigar Function
10. Expanded Griewank's plus Rosenbrock's Function
11. Weierstrass Function

F28:

1. Rastrigin's Function
2. Griewank's Function
3. Rosenbrock's Function
4. Schaffer’s F6 Function
5. Katsuura Function
6. Ackley's Function

F27:

1. HGBat Function
2. Rastrigin's Function
3. Modified Schwefel's Function
4. Bent-Cigar Function
5. High Conditioned Elliptic Function
6. Expanded Schaffer's F6 Function

F26:

1. Expanded Schaffer's F6 Function
2. Modified Schwefel's Function
3. Griewank's Function
4. Rosenbrock's Function
5. Rastrigin's Function

F25:

1. Rastrigin's Function
2. Happy Cat Function
3. Ackley's Function
4. Discus Function
5. Rosenbrock's Function

F24:

1. Ackley's Function
2. High Conditioned Elliptic Function
3. Griewank's Function
4. Rastrigin's Function

F23:

1. Rosenbrock's Function
2. Ackley's Function
3. Modified Schwefel's Function
4. Rastrigin's Function

F22:

1. Rastrigin's Function
2. Griewank's Function
3. Modified Schwefel's Function

F21:

1. Rosenbrock's Function
2. High Conditioned Elliptic Function
3. Rastrigin's Function

### Results with Selected CEC-2017 Composites

D=20, seed=20260829, 200M evals.

| Place | Algorithm    | F22  | F24  | F25  | F28  |
| ----- | ------------ | ---- | ---- | ---- | ---- |
| 1     | R6           | 2251 | 2438 | 2600 | 2804 |
| 2     | ARRDE        | 2243 | 2400 | 2899 | 3000 |
| 3     | BIPOP-aCMAES | 2300 | 2800 | 2910 | 3100 |

- R6 (my own ARRDE mod): solves F22, F24, F28, makes progress on F25 (in just 10M..50M evals).

- ARRDE: solves F22 and F24. Can be pushed to 2700 on F25 with 500M..2B evals.

- BIPOP-aCMAES lags already on F22 (also stalls on F21 at ~2300 when the other two get into ~2200).

F21, F23, F26, F27, F29, F30 remained unsolved.

A month later:

Vanilla ARRDE also solves F28, but one needs to use tiny popsize=50, and run restarts with 20M eval budget. This mode also solves F24 much faster, 10M eval budget is enough with about 5 restarts, and also F25 in D=10 is solvable.

It can also get into 2600 on F25 CEC-2017 via restarts or tunneling heuristics.

I put R6 on hold for now. It runs 5x faster in real time, but it does not solve anything new compared to ARRDE.

## Why Some Composites are not Solvable

F25 CEC-2017 in D=20 turns out to be a needle in haystack.

The box is [-100, 100]^20.

Around the global minimum, the sphere of radius 2.6 already produces points above f=2600, 
but these are the best f-values of a wider deceptive region/attractor. 

It is still hard to get even into that attractor, but most powerful algorithms (ARRDE with tinkering, not vanilla ARRDE) find it. The global minimum region is effectively of volume zero.

```bash
==============================================================================
CEC2017 F25 / D=20
==============================================================================
Benchmark f_opt : 2.500000000000000e+03
Evaluated f(CENTER): 2.500000000000000e+03
Difference from benchmark optimum: 0.000000000000000e+00

Global optimum coordinates:
x[ 1] =  9.5857011347942525e+00
x[ 2] =  7.3284069743605357e+01
x[ 3] = -2.0094577798663906e+01
x[ 4] =  1.7371653140229942e+01
x[ 5] =  5.9574127940262727e+01
x[ 6] = -7.3501078371657567e+00
x[ 7] = -7.2252707276610991e+01
x[ 8] =  7.1523160505817813e+01
x[ 9] = -3.0593826926589863e+01
x[10] =  2.9168098455838908e+01
x[11] = -4.8827527560679016e+00
x[12] = -2.0297992276166561e+01
x[13] = -2.9676773673704254e+01
x[14] =  6.9616639408104419e+01
x[15] = -2.0525248447677704e+01
x[16] =  6.3380734488079668e+01
x[17] =  5.0897438177325498e+01
x[18] = -2.7404310164695751e+01
x[19] =  4.2375220020476362e+01
x[20] = -6.8372003363618632e+01

Sphere radius requested : 2.600000000000
Number of sphere points : 20
Random seed             : 20260926

idx   distance from CENTER          f(x)
---   ----------------------   ----------------------
  1        2.599999999999998    2.629636086737374e+03
  2        2.600000000000004    2.625406875661240e+03
  3        2.599999999999998    2.622794897119958e+03
  4        2.600000000000001    2.619765718635950e+03
  5        2.599999999999998    2.630442065951148e+03
  6        2.599999999999994    2.633805380061843e+03
  7        2.600000000000004    2.609663671141000e+03
  8        2.599999999999995    2.623780417555373e+03
  9        2.599999999999998    2.631237490944540e+03
 10        2.599999999999995    2.610352600002223e+03
 11        2.600000000000003    2.615985974975741e+03
 12        2.600000000000000    2.614363551126151e+03
 13        2.600000000000002    2.637206051053545e+03
 14        2.600000000000001    2.604314325506584e+03
 15        2.600000000000001    2.618162146817209e+03
 16        2.600000000000000    2.620541351031640e+03
 17        2.600000000000000    2.632102145035525e+03
 18        2.600000000000005    2.619249925266242e+03
 19        2.600000000000000    2.622497903425905e+03
 20        2.600000000000000    2.632773260194739e+03

Sphere min  f = 2.604314325506584e+03
Sphere max  f = 2.637206051053545e+03
Sphere mean f = 2.622704091912197e+03
```

Even at 2.5 it is still 1/20 chance to see the direction below 2600, any mu-averaging would lose it:

```bash
Sphere radius requested : 2.500000000000
Number of sphere points : 20
Random seed             : 20260926

idx   distance from CENTER          f(x)
---   ----------------------   ----------------------
  1        2.500000000000000    2.622540334827559e+03
  2        2.499999999999998    2.618613571225004e+03
  3        2.500000000000000    2.616154304786483e+03
  4        2.500000000000000    2.613354887936021e+03
  5        2.500000000000000    2.623250035207675e+03
  6        2.499999999999998    2.626366569770741e+03
  7        2.500000000000002    2.604007963855981e+03
  8        2.500000000000002    2.617063556581270e+03
  9        2.499999999999996    2.623973395079520e+03
 10        2.500000000000002    2.604645806623607e+03
 11        2.500000000000000    2.609856338658577e+03
 12        2.500000000000003    2.608341155895762e+03
 13        2.499999999999999    2.629555808086666e+03
 14        2.500000000000001    2.599056596278414e+03
 15        2.499999999999998    2.611868607211285e+03
 16        2.500000000000004    2.614081095829475e+03
 17        2.499999999999998    2.624817280430716e+03
 18        2.499999999999997    2.612874689563205e+03
 19        2.500000000000000    2.615887632205078e+03
 20        2.499999999999999    2.625497713046359e+03

Sphere min  f = 2.599056596278414e+03
Sphere max  f = 2.629555808086666e+03
Sphere mean f = 2.616090367154970e+03
```

A sphere of radius 2.5 in D=20 has a volume 2.3471e6. The whole search space is 200^200 ~ 1.048576e46. The volume ratio is 1e-40.

Assume the same r=2.5, but D=10. A sphere now has a volume 2.43202594745e4. The whole box is 200^100 ~ 1.024e23. The volume ratio is 1e-19. Still tiny, but already searchable by the ARRDE with tinkering and budgets of O(1e8..1e9) evals. 

In a way, current intelligence squares eval budgets which is already enough in D=10. For D=20, we need 10 extra orders, which might be doable with GPU clouds (in theory), but D=40 is beyond anything.

When someone says that "It works in D=10, but it will work in D=20, 40... I just don't want to waste time on longer runs", that means they are dealing with easy problems and are content with local optima.

### Personal Notes

- ARRDE is the first algorithm to solve a CEC-2017 composite in D=20. No matrices, think about it.

- Three composites are already solvable in D=20: F22, F24, and F28.

- F25 is solvable in D=10 (R6, <50M evals, vanilla ARRDE will get there too). BIPOP-aCMAES does not solve it.

- For larger budgets it might make sense to shrink default ARRDE popsize to 50 and wrap ARRDE inside restarts. Say, 50 restarts with 20M budget per run instead of a single run with 1B evals.

- Tunneling/filling (see Aimo Törn and Antanas Žilinskas (1987) Global Optimization) do not improve the state of the art (ARRDE). They can reduce evals and help reaching 2600 on F25 CEC-2017 in D=20, but this is not solving it (fopt=2500).

- ARRDE/R6 suffer in D>10 and are pale on [Lunacek's bi-Rastrigin](https://coco-platform.org/testsuites/bbob/functions/f24.html) already in D=20, while (mu, lambda)-ES and BIPOP-aCMAES solve the problem in D=40 very rapidly, in <10M evals.

- jSO improves tiny bit LSHADE, nothing as dramatic as advertised. j2020 is significantly better than LSHADE/jSO on F24 and F25 in CEC-2017 D=10, but still not good enough. ARRDE is much better, but also complex, overtuned, and hopeless on tougher cases in D=20.

- One pretty moment here is that simple (mu, lambda)-ES solves [Lunacek's bi-Rastrigin](https://coco-platform.org/testsuites/bbob/functions/f24.html) in D=40. This cost function mixes quadrics with harmonics via sum and min operators. It is vital in physics, but normally not a black box, which rules out DFO.

- F25 CEC-2017 in D=20 is a tough nut to crack. This problem defies mechanisms to escape local minima. It seems to be impossible to identify and exclude regions that drop anything to f=2600 instead of fopt=2500, at least not with lists, boxes, and ellipsoids.

- The difficulty is not in multimodality, stiffness, limited budgets per se. These can lead to solvable problems. Solvability depends a lot on how narrow the global optimum is for D>10. If a well is such that its size in a single dimension is about 0.1% or 1% of the searchable coordinate range, this is still solvable in D=10, but not in D=20 as the F25 CEC-2017 challenge indicates.

- Strive not to mix variables of different nature and scale, this complicates DFO enormously and nothing really works beyond D=10. Notice that CEC-2017 is only a two-layer mixing and generally non-solvable already in D=20. We can complicate this much further and no algorithm will ever catch up.

- I do not expect much progress in DFO in the nearest decade. RL/AI won't solve fundamental difficulties. CMAES halts at ill-conditioning/stiffness. DEs do not scale beyond D>10 and are already very ugly codes with so many parameters to tune.

- Instead of ES or DE, better focus more on what is actually being optimized.

## Selected References

- M.J. Box (1966) A Comparison of Several Current Optimization Methods, and the use of Transformations in Constrained Problems

- J. Bernussou and J. Geromel (1981) An easy way to find gradient matrix of composite matricial functions

- [CMAES 1996 - 2014](https://cma-es.github.io/)

- Aurore Blelly at al. (2018) [Stopping Criteria, Initialization, and Implementations of
  BFGS and their Effect on the BBOB Test Suite](https://inria.hal.science/hal-01811588/file/workshop_paper-authorversion.pdf)

- Nikolaus Hansen (2019) [A Global Surrogate Assisted CMA-ES](https://inria.hal.science/hal-02143961v1/document), [pycma (github)](https://github.com/CMA-ES/pycma), [pycma Issue 356](https://github.com/CMA-ES/pycma/issues/356)

- Nikolaus Hansen at al. (2019) [Real-Parameter Black-Box Optimization Benchmarking 2009: Noiseless Functions Definitions](https://inria.hal.science/inria-00362633v2/document)

- Zachary Hoffman and Steve Huntsman (2022) [Benchmarking an algorithm for expensive high-dimensional
  objectives on the BBOB and BBOB-largescale testbeds](https://hal.science/hal-03665291v1/file/GECCOarXiv2022.pdf)

- Khoirul Faiq Muzakka, Ahsani Hafizhu Shali, Haris Suhendar, Sören Möller, Martin Finsterbusch (2026) [Robust Differential Evolution via Nonlinear Population Size Reduction and Adaptive Restart: The ARRDE Algorithm](https://arxiv.org/abs/2511.18429v4), [Minion (github)](https://github.com/khoirulmuzakka/Minion), [Minion Issue 11](https://github.com/khoirulmuzakka/Minion/issues/11), [algolist](https://minion-py.readthedocs.io/en/latest/algolist.html)

Farewell to matrices and convergence proofs: [356](https://github.com/CMA-ES/pycma/issues/356), [367](https://github.com/CMA-ES/pycma/discussions/367), but honestly farewell to list processing (DEs) too, and I do not think AI or RL add much here.
