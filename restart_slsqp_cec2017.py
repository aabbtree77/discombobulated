#!/usr/bin/env python3
"""
Pure multi-start SLSQP / L-BFGS-B on CEC-2017.
python restart_slsqp_cec2017.py --function 25 --dim 10 --algo slsqp
python restart_slsqp_cec2017.py --function 25 --dim 10 --algo bfgs
"""

import argparse
import time
import warnings
import numpy as np
from scipy.optimize import minimize
import minionpy as mpy

# ------------------------------------------------------------
MAX_EVALS      = 1_000_000_000
PROGRESS_EVERY = 10_000_000
MAXITER        = 100
FTOL           = 1e-14
EPS            = 1e-8
# ------------------------------------------------------------

CEC2017_BIASES = {
    1: 100.0, 3: 300.0, 4: 400.0, 5: 500.0, 6: 600.0, 7: 700.0,
    8: 800.0, 9: 900.0, 10: 1000.0, 11: 1100.0, 12: 1200.0,
    13: 1300.0, 14: 1400.0, 15: 1500.0, 16: 1600.0, 17: 1700.0,
    18: 1800.0, 19: 1900.0, 20: 2000.0, 21: 2100.0, 22: 2200.0,
    23: 2300.0, 24: 2400.0, 25: 2500.0, 26: 2600.0, 27: 2700.0,
    28: 2800.0, 29: 2900.0, 30: 3000.0,
}

class BudgetExceeded(Exception):
    pass

class State:
    def __init__(self, problem, lower, upper):
        self.problem = problem
        self.lower   = lower
        self.upper   = upper
        self.evals   = 0
        self.best_f  = np.inf
        self.best_x  = None
        self.next_progress = PROGRESS_EVERY
        self.t0 = time.perf_counter()

    def evaluate(self, x):
        if self.evals >= MAX_EVALS:
            raise BudgetExceeded
        x = np.clip(np.asarray(x, dtype=float).ravel(), self.lower, self.upper)
        f = float(self.problem(x.reshape(1, -1))[0])
        if not np.isfinite(f):
            f = 1e300
        self.evals += 1
        if f < self.best_f:
            self.best_f = f
            self.best_x = x.copy()
        self._progress()
        return f

    def _progress(self):
        while self.evals >= self.next_progress:
            elapsed = time.perf_counter() - self.t0
            nrm = 0.0 if self.best_x is None else float(np.linalg.norm(self.best_x))
            print(
                f"evals={self.evals:12d}  "
                f"best_f={self.best_f:.12e}  "
                f"||xbest||={nrm:.6e}  "
                f"time={elapsed:10.1f}s",
                flush=True,
            )
            self.next_progress += PROGRESS_EVERY

def random_point(lower, upper, dim):
    return np.random.uniform(lower, upper, dim)

def run(args):
    np.random.seed(args.seed)
    dim = args.dim
    lower, upper = -100.0, 100.0
    bounds = [(lower, upper)] * dim

    problem = mpy.CEC2017Functions(function_number=args.function, dimension=dim)
    fopt = CEC2017_BIASES.get(args.function, 0.0)

    print(f"CEC2017 F{args.function}  D={dim}  algo={args.algo}  "
          f"budget={MAX_EVALS}  maxiter={MAXITER}")
    print(f"fopt ≈ {fopt:.6e}\n")

    state = State(problem, lower, upper)
    method = "SLSQP" if args.algo.lower() == "slsqp" else "L-BFGS-B"

    options = {
        "maxiter": MAXITER,
        "ftol": FTOL,
        "disp": False,
    }
    if method == "SLSQP":
        options["eps"] = EPS
        options["iprint"] = 0
    else:  # L-BFGS-B
        options["gtol"] = 1e-14
        options["maxfun"] = MAXITER * (2 * dim + 1)  # generous

    while state.evals < MAX_EVALS:
        x0 = random_point(lower, upper, dim)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                minimize(
                    fun=state.evaluate,
                    x0=x0,
                    method=method,
                    jac="2-point",
                    bounds=bounds,
                    options=options,
                )
        except BudgetExceeded:
            break

    elapsed = time.perf_counter() - state.t0
    print("\n==============================")
    print("Finished")
    print("==============================")
    print(f"Best f   : {state.best_f:.12e}")
    print(f"fopt     : {fopt:.12e}")
    print(f"Error    : {abs(state.best_f - fopt):.12e}")
    if state.best_x is not None:
        print(f"||xbest||: {np.linalg.norm(state.best_x):.6e}")
    print(f"Evals    : {state.evals}")
    print(f"Time     : {elapsed:.1f}s")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--function", type=int, default=25)
    p.add_argument("--dim", type=int, default=10)
    p.add_argument("--algo", choices=["slsqp", "bfgs"], default="slsqp")
    p.add_argument("--seed", type=int, default=20261002)
    args = p.parse_args()
    run(args)
