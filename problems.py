# problems.py

from datetime import datetime

import numpy as np
import cocoex
import minionpy as mpy


CEC2022_BIASES = {
    1: 300.0,
    2: 400.0,
    3: 600.0,
    4: 800.0,
    5: 900.0,
    6: 1800.0,
    7: 2000.0,
    8: 2200.0,
    9: 2300.0,
    10: 2400.0,
    11: 2600.0,
    12: 2700.0,
}


class Problem:
    """
    Generic optimization problem wrapper.

    Common interface independent of benchmark backend.
    """

    def __init__(
        self,
        backend,
        dimension,
        evaluator,
        bounds,
        fopt,
        description,
    ):
        self.backend = backend
        self.D = int(dimension)

        self.bounds = np.asarray(bounds, dtype=float)
        if self.bounds.shape != (self.D, 2):
            raise ValueError("bounds must have shape (D,2)")

        self.fopt = float(fopt)
        self.description = description

        self._evaluate = evaluator

        self.evals = 0

        self.created = datetime.now()

    # ------------------------------------------------------------
    # evaluation
    # ------------------------------------------------------------

    def evaluate(self, X):
        """
        Evaluate one or many points.

        Input
            (D,)
            (N,D)

        Output
            (N,)
        """

        X = np.asarray(X, dtype=float)

        single = (X.ndim == 1)

        if single:
            X = X[None, :]

        F = np.asarray(self._evaluate(X), dtype=float)

        self.evals += len(F)

        if single:
            return float(F[0])

        return F

    # ------------------------------------------------------------
    # utilities
    # ------------------------------------------------------------

    def project(self, X):
        lo = self.bounds[:, 0]
        hi = self.bounds[:, 1]
        return np.clip(X, lo, hi)

    # ------------------------------------------------------------
    # information
    # ------------------------------------------------------------

    def print(self):
        print()
        print("Problem")
        print("----------------------------------------")
        print("Backend :", self.backend)
        print("Name    :", self.description)
        print("D       :", self.D)

        lo = self.bounds[:, 0]
        hi = self.bounds[:, 1]

        if np.all(lo == lo[0]) and np.all(hi == hi[0]):
            print(f"Bounds  : [{lo[0]}, {hi[0]}] for every coordinate")
        else:
            print("Bounds")
            for i, (a, b) in enumerate(self.bounds):
                print(f"  x[{i:2d}] : [{a}, {b}]")

        print("fopt    :", self.fopt)
        print("Created :", self.created.strftime("%Y-%m-%d %H:%M:%S"))
        print()


# ======================================================================
# BBOB
# ======================================================================

def make_bbob2009(function, dimension, instance=1):
    problem = cocoex.BareProblem(
        suite_name="bbob",
        function=function,
        dimension=dimension,
        instance=instance,
    )

    bounds = np.tile(
        np.array([-5.0, 5.0]),
        (dimension, 1),
    )

    def evaluator(X):
        return [float(problem(x)) for x in X]

    return Problem(
        backend="BBOB",
        dimension=dimension,
        evaluator=evaluator,
        bounds=bounds,
        fopt=problem.best_value(),
        description=problem.id,
    )


# ======================================================================
# CEC2022
# ======================================================================

def make_cec2022(function, dimension):
    problem = mpy.CEC2022Functions(
        function_number=function,
        dimension=dimension,
    )

    bounds = np.tile(
        np.array([-100.0, 100.0]),
        (dimension, 1),
    )

    def evaluator(X):
        return problem(np.asarray(X))

    return Problem(
        backend="CEC2022",
        dimension=dimension,
        evaluator=evaluator,
        bounds=bounds,
        fopt=CEC2022_BIASES[function],
        description=f"CEC2022 f{function}",
    )
