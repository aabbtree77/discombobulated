#!/usr/bin/env python3

import numpy as np

from cwalk import CWALK
from problems import make_bbob2009


# ============================================================
# Settings
# ============================================================

# To run with random init point each time, 
# comment out rng in es = CWALK(... below.
# SEED is for reproducibility of results.
# What is incredible is that everything works with 20260723
# Restart a dozen of times to get the best results.

SEED = 20260723

FUNCTION = 24
DIMENSION = 40 #change to 20, but need to adjust sigma_multiplier and do restarts again
# Note that BBOB2009 and CECs have very limited dimension sets, you cannot use arbitrary
# values. These benchmarks use C++ code which builds rotation matrices to mangle cost funcs
# to make them harder, the matrices are stored as dense arrays with archaic messy tiny bounds.

BUDGET = 10_000*DIMENSION
PROGRESS_EVERY = 10_000

#Default lambda is 10D set with
LAMBDA = None

# For finesse and epsilon improvements, use 100D,
# but the runs will take more time and a new play with step sizes
# to improve the run on 10D. Not recommended unless you are into bs competitions.
# I add it here just to show off a bit against CMAES which will struggle with such
# large lambdas.
 
#LAMBDA = 4000

INIT_AT_ZERO = False

USE_MANUAL_SIGMA = False
MANUAL_SIGMA = 1.0

SIGMA_MULTIPLIER = np.exp(-1e-2)

# SIGMA_MULTIPLIER is still a pain in the ass, might be automatable, but I prefer to adjust per problem
# manually as I am more into whether it solves a particular problem rather than nonsense tables.
# Automation matters for black box uses though, inside EGO/TREGO/BO and such, or to solve Rubik's cube... 
# Not sure a single exponent is the best way, only one parameter, but rather sensitive.
# Might be better to split into very short opening phase, long middle game, and the end game with fixed sigmas
# like in the early deep nets before Adam and all.
#  
# On BBOB2009 F24 the best cases are when the algo reaches the first good minimum around f = 1.4
# with step size 1e-3...1e-4, but the result may vary depending on init random point (not critically, 
# but may need to restart up to 10...100 restarts). On other problems everything is new again.
# The adjustment is not very hard, just takes time. Start with 1e-2, 1e-3, 1e-4. Do a lot of restarts
# per value, then fine tune a bit deciding whether it's 2e-3, 3e-3... No need to go too precisely,
# restarts matter a lot. A single success in 30, 50, 100 restarts is a success and reveals possibilities.
# I would not bang my head against CEC2022 F12 though, we do not know if these composites are searchable per se.
# It's like looking for a needle in 10^20, might still be doable, but I would not go there.
# Better look into bandits, RL, game theory and such, more interesting trade offs and optimization problems.

LOGFILE = "progress_bbob2009_f24.csv"


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

        '''
        if completed_evals < 1e5:
            es.sigma = 1e-1
        elif completed_evals < 2e5:
            es.sigma = 1e-2
        elif completed_evals < 3e5:
            es.sigma = 1e-3
        elif completed_evals < 4e5:
            es.sigma = 1e-4       
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
