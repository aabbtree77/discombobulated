#!/usr/bin/env python3

import numpy as np

from cwalk import CWALK
from problems import make_cec2022


# ============================================================
# Settings
# ============================================================

SEED = 20260723

# This function is a hopeless mixture of multimodals with rotations.
# Currently beyond any modern tech, but might not be solvable
# in principle with DFOs due to lack of structure and vast space in
# 10^D with coord ranges -100.0... 100.0. Anything decent gets into
# fopt = 2970 or so rapidly, say in 100K evals, and then stalls.
# The best I saw some PSO 2026 methods that get into 2860s, but this
# is still not 2700s and not worthy, people overfit their algorithms
# in those competitions like crazy, I would not go there.
# Suffered enough on MNIST digits at their time, burning PC for the
# entire night for better params, overfitting, it's not worth it.
# Nobody remembers MNIST digits anymore, but how much time was
# wasted there! And on CIFAR10 and the rest.

FUNCTION = 12
DIMENSION = 20

BUDGET = 10_000*DIMENSION
PROGRESS_EVERY = 10_000

LAMBDA = 10*DIMENSION

INIT_AT_ZERO = False

USE_MANUAL_SIGMA = False
MANUAL_SIGMA = 20.0

SIGMA_MULTIPLIER = np.exp(-1e-2)

LOGFILE = "progress_cec_f12.csv"


# ============================================================

def main():

    rng = np.random.default_rng(SEED)

    problem = make_cec2022(
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
        sigma = MANUAL_SIGMA
    else:
        largest_range = np.max(problem.bounds[:, 1] - problem.bounds[:, 0])
        sigma = 0.10 * largest_range

    print(f"Initial sigma : {sigma:.6g}")

    # --------------------------------------------------------
    # optimizer
    # --------------------------------------------------------

    es = CWALK(
        D=problem.D,
        x0=x0,
        sigma=sigma,
        sigma_multiplier=SIGMA_MULTIPLIER,
        lam=LAMBDA,
        rng=rng,
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
                f"sigma={es.sigma:.3e}"
            )

            next_progress += PROGRESS_EVERY
        
        '''    
        if completed_evals < 2e6:
            es.sigma = 100
        elif completed_evals < 4e6:
            es.sigma = 50
        elif completed_evals < 6e6:
            es.sigma = 20     
        '''
        
        X = es.ask()

        Xeval = problem.project(X)

        F = problem.evaluate(Xeval)

        es.tell(X, F)

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
