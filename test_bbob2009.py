#!/usr/bin/env python3

import numpy as np

from cwalk import CWALK
from problems import make_bbob2009

# ============================================================
# Settings
# ============================================================

# To run with random init point each time, 
# comment out rng in es = CWALK(... below.
# SEED is for reproducibility.
# Restart to get the best results.

#SEED = 20260723
SEED = 20250306

FUNCTION = 10
DIMENSION = 40
# Note: BBOB2009 and CECs have very limited dimension sets, one cannot use arbitrary
# values. These benchmarks use C++ code which builds rotation matrices to mangle cost funcs
# to make them harder, the matrices are stored as dense arrays with messy static bounds
# which limit DIMENSION to 40 on BBOB-2009 and 20 on CEC2022.

BUDGET = 100_000*DIMENSION
PROGRESS_EVERY = 100_000

#Default lambda is 100D, set it with None or use a number:
LAMBDA = None

INIT_AT_ZERO = True

USE_MANUAL_SIGMA = False
MANUAL_SIGMA = 0.1

LOGFILE = "progress_bbob2009_f10.csv"

# ============================================================

def sigma_multiplier(completed_evals: int, budget: int) -> float:
    """
    Exponential decay:
        0%   -> 1.0
        100% -> 1e-4
    """
    t = np.clip(completed_evals / budget, 0.0, 1.0)
    return 10.0 ** (-4.0 * t)    


# ============================================================

def main():

    rng = np.random.default_rng(SEED)

    problem = make_bbob2009(
        function=FUNCTION,
        dimension=DIMENSION,
    )

    problem.print()

    # --------------------------------------------------------
    # initial point
    # --------------------------------------------------------

    if INIT_AT_ZERO:
        x0 = np.zeros(problem.D)
    else:
        x0 = rng.uniform(
            problem.bounds[:, 0],
            problem.bounds[:, 1],
        )

    # --------------------------------------------------------
    # sigma
    # --------------------------------------------------------

    if USE_MANUAL_SIGMA:
        sigma0 = MANUAL_SIGMA
    else:
        largest_range = np.max(problem.bounds[:, 1] - problem.bounds[:, 0])
        sigma0 = 0.10 * largest_range

    print(f"Initial sigma : {sigma0:.6g}")

    # --------------------------------------------------------
    # optimizer
    # --------------------------------------------------------

    es = CWALK(
        D=problem.D,
        x0=x0,
        sigma=sigma0,
        lam=LAMBDA,
        rng=rng, #comment out to get random init point every time this file runs
    )

    # --------------------------------------------------------
    # logging
    # --------------------------------------------------------

    history = []

    best_f = np.inf

    completed_evals = 0
    next_progress = PROGRESS_EVERY

    # ========================================================
    # optimization
    # ========================================================

    while completed_evals + es.lam <= BUDGET:

        #
        # Report everything whose evaluation count lies inside
        # the completed interval [completed_evals].
        #
        while next_progress <= completed_evals:

            history.append((next_progress, best_f))

            print(
                f"evals={next_progress:10d} "
                f"best_f={best_f:.6e} "
                f"error={abs(best_f-problem.fopt):.6e} "
                f"sigma={es.sigma:.3e} "
           #    f"normz={es.normz:.3e} "
            )

            next_progress += PROGRESS_EVERY
            
        X = es.ask()

        Xeval = problem.project(X)

        F = problem.evaluate(Xeval)

        sigma = sigma0 * sigma_multiplier(completed_evals, BUDGET)
        es.tell(X, F, sigma)

        best_f = min(best_f, float(np.min(F)))

        completed_evals += es.lam

    #
    # Flush remaining progress points.
    #
    while next_progress <= BUDGET:

        history.append((next_progress, best_f))

        print(
            f"evals={next_progress:10d} "
            f"best_f={best_f:.6e} "
            f"error={abs(best_f-problem.fopt):.6e} "
            f"sigma={es.sigma:.3e}"
        )

        next_progress += PROGRESS_EVERY

    #
    # Exact budget endpoint.
    #
    history.append((BUDGET, best_f))

    # --------------------------------------------------------
    # save
    # --------------------------------------------------------

    np.savetxt(
        LOGFILE,
        np.asarray(history),
        delimiter=",",
        header="evals,best_f",
        comments="",
    )

    # --------------------------------------------------------
    # summary
    # --------------------------------------------------------

    print()
    print("Finished")
    print("----------------------------------------")
    print(f"Completed evaluations : {completed_evals}")
    print(f"Requested budget      : {BUDGET}")
    print(f"Best f                : {best_f:.12e}")
    print(f"fopt                  : {problem.fopt:.12e}")
    print(f"Error                 : {abs(best_f-problem.fopt):.12e}")
    print(f"Progress saved to     : {LOGFILE}")


if __name__ == "__main__":
    main()
