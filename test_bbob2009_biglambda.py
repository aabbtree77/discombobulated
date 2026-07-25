#!/usr/bin/env python3

import numpy as np

from cwalk import CWALK
from problems import make_bbob2009


# ============================================================
# Settings
# ============================================================

# To run with random init point each time, 
# comment out rng in es = CWALK(... below.
# SEED is for reproducibility of results
# What is incredible is that everything works with 20260723,
# it gets to percentages within fopt of F24, then you wake up
# next day, set the seed to 20260724, and the optimizer gets stuck
# at local minimum around fopt = 1.43... rather than needed 1.026...
# The answer is simply restart, you will get there in say 50 runs,
# or earlier. Restarts are essential, 
# and lambda = 100D is an overkill to show off finesse in the end.
SEED = 20260723

FUNCTION = 24
DIMENSION = 40 #change to 20, but need to adjust sigma_multiplier and do restarts again
# Note that BBOB2009 and CECs have very limited dimension sets, you cannot use arbitrary
# values. These benchmarks use C++ code which builds rotation matrices to mangle cost funcs
# to make them harder, the matrices are stored as dense arrays with archaic messy tiny bounds.

BUDGET = 10_000_000
PROGRESS_EVERY = 1000_000

#Default lambda is 10D set with
#LAMBDA = None

# For finesse and epsilon improvements, use 100D,
# but the runs will take more time and a new play with step sizes
# to improve the run on 10D. Not recommended unless you are into bs competitions.
# I add it here just to show off a bit against CMAES which will struggle with such
# large lambdas.
 
LAMBDA = 4000

INIT_AT_ZERO = False

USE_MANUAL_SIGMA = False
MANUAL_SIGMA = 1.0

SIGMA_MULTIPLIER = np.exp(-2e-3) # This is still pain in the ass, need to adjust per problem

LOGFILE = "progress_bbob2009_f24_biglambda.csv"


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
                f"sigma={es.sigma:.3e}"
            )

            next_progress += PROGRESS_EVERY

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
