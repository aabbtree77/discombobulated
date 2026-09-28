#!/usr/bin/env python3

import numpy as np
import minionpy as mpy

# ============================================================
# CONFIGURATION
# ============================================================

D = 20
FUNCTION = 25

# Number of starting points / independent ARRDE runs.
N = 20

# Evaluation budget for EACH run.
#
# Example:
#   RUN_BUDGET = 1_000_000
#   N = 20
#   => total = 20,000,000 evaluations
#
RUN_BUDGET = 1_000_000

# Exact same sphere used by the point-generation script.
RADIUS = 8.0
START_SEED = 20260928

# ARRDE seeds:
# run_seed = BASE_SEED + run_index
BASE_SEED = 20261001

LOWER = -100.0
UPPER = 100.0

PROGRESS_STEP = 1_000_000


# ============================================================
# CEC2017 GLOBAL OPTIMUMS
# ============================================================

CEC2017_BIASES = {
    1: 100.0,
    2: 200.0,
    3: 300.0,
    4: 400.0,
    5: 500.0,
    6: 600.0,
    7: 700.0,
    8: 800.0,
    9: 900.0,
    10: 1000.0,
    11: 1100.0,
    12: 1200.0,
    13: 1300.0,
    14: 1400.0,
    15: 1500.0,
    16: 1600.0,
    17: 1700.0,
    18: 1800.0,
    19: 1900.0,
    20: 2000.0,
    21: 2100.0,
    22: 2200.0,
    23: 2300.0,
    24: 2400.0,
    25: 2500.0,
    26: 2600.0,
    27: 2700.0,
    28: 2800.0,
    29: 2900.0,
    30: 3000.0,
}


# ============================================================
# EXACT CEC2017 F25 GLOBAL OPTIMUM
# ============================================================

CENTER_FULL = np.array(
    [
         9.5857011347942525e+00,
        73.284069743605357e+00,
       -20.094577798663906e+00,
        17.371653140229942e+00,
        59.574127940262727e+00,
        -7.3501078371657567e+00,
       -72.252707276610991e+00,
        71.523160505817813e+00,
       -30.593826926589863e+00,
        29.168098455838909e+00,
        -4.8827527560679016e+00,
       -20.297992276166561e+00,
       -29.676773673704254e+00,
        69.616639408104419e+00,
       -20.525248447677704e+00,
        63.380734488079668e+00,
        50.897438177325498e+00,
       -27.404310164695751e+00,
        42.375220020476362e+00,
       -68.372003363618632e+00,
        61.120696923286047e+00,
        29.826898722828787e+00,
       -36.080297971181707e+00,
         4.0526012350118723e+00,
       -51.488403582707235e+00,
       -30.635065577288653e+00,
         0.12455935036975860e+00,
        31.959388017635508e+00,
       -69.112390274699962e+00,
       -26.759882931951381e+00,
        16.901054032566517e+00,
         8.1119407229646772e+00,
        -9.3765833570512864e+00,
        59.902308447240998e+00,
        42.611492035263915e+00,
        18.664839589346592e+00,
       -19.584820637603378e+00,
        78.828522830656041e+00,
       -74.943542120486285e+00,
         9.7592905656245019e+00,
       -21.136839004762862e+00,
       -74.208980982289177e+00,
       -46.370276882485612e+00,
       -67.076217041852061e+00,
       -63.663971349440502e+00,
         3.3534800127863327e+00,
       -13.108804358685763e+00,
        39.317858080993339e+00,
        79.883666872924039e+00,
        31.900771669997262e+00,
       -22.370670597241570e+00,
        18.403252699600184e+00,
       -51.306587094118157e+00,
        15.869611672996362e+00,
        54.958916406593715e+00,
       -77.040715722851786e+00,
       -47.561628944967701e+00,
        74.170228351918155e+00,
        10.955505085622661e+00,
         7.7695604524433426e+00,
       -20.151089267875577e+00,
       -42.119840306285695e+00,
         0.35786972956632718e+00,
       -64.740255594598253e+00,
        -7.0201047095563744e+00,
        49.993096322749324e+00,
       -34.000076637603378e+00,
       -19.006704316943278e+00,
        51.502756075702031e+00,
       -13.368971190121478e+00,
       -50.494241916804690e+00,
        -2.9046729146059018e+00,
        24.443312537131511e+00,
       -37.150883252337053e+00,
        71.908756028818317e+00,
        20.890545340669597e+00,
       -13.765837571474648e+00,
        13.271049133162503e+00,
        56.633032623973222e+00,
        72.636335742024869e+00,
       -32.651322588627842e+00,
       -64.526773974544724e+00,
       -61.846743888150314e+00,
       -57.400132161285789e+00,
       -16.447940424293250e+00,
       -61.702561057034210e+00,
        48.609598145172747e+00,
       -43.304817860622336e+00,
       -43.897982602891503e+00,
       -10.929345575594038e+00,
       -11.371411738624523e+00,
        53.977914820414931e+00,
       -78.190067983320944e+00,
       -49.629093036814169e+00,
       -10.571151715969272e+00,
         8.8990752216889870e+00,
       -79.636308625171111e+00,
       -57.846916352327412e+00,
       -37.796678821461590e+00,
       -59.867612166268572e+00,
    ],
    dtype=float,
)

SUPPORTED_DIMS = (2, 10, 20, 30, 50, 100)

if D not in SUPPORTED_DIMS:
    raise ValueError(
        f"D={D} is not a standard CEC2017 dimension. "
        f"Supported values: {SUPPORTED_DIMS}"
    )

CENTER = CENTER_FULL[:D].copy()


# ============================================================
# GENERATE THE SAME TYPE OF SPHERE POINT LIST
# ============================================================

def generate_sphere_points():
    rng = np.random.default_rng(START_SEED)

    points = []
    attempts = 0
    max_attempts = max(1_000, N * 1_000_000)

    while len(points) < N:
        attempts += 1

        if attempts > max_attempts:
            raise RuntimeError(
                "Could not obtain enough in-bounds sphere points. "
                "Try a smaller RADIUS."
            )

        direction = rng.normal(size=D)
        norm = np.linalg.norm(direction)

        if norm == 0.0:
            continue

        direction /= norm

        point = CENTER + RADIUS * direction

        # Do not clip. Clipping would change the physical radius.
        if np.all((point >= LOWER) & (point <= UPPER)):
            points.append(point)

    return np.asarray(points, dtype=float)


# ============================================================
# RUN ONE ARRDE OPTIMIZATION
# ============================================================

def run_arrde(problem, start_point, run_index, fopt):
    run_seed = BASE_SEED + run_index

    evals = 0
    best_f = np.inf
    next_progress = PROGRESS_STEP

    def tracked_batch(X):
        nonlocal evals, best_f, next_progress

        fs = np.asarray(problem(X), dtype=float).reshape(-1)

        evals += len(X)

        gen_best = float(np.min(fs))
        if gen_best < best_f:
            best_f = gen_best

        while evals >= next_progress:
            error = abs(best_f - fopt)

            print(
                f"[run {run_index + 1:2d}/{N:2d}] "
                f"evals={next_progress:10d} "
                f"best={best_f:.12e} "
                f"error={error:.12e}"
            )

            next_progress += PROGRESS_STEP

        return fs

    bounds = [(LOWER, UPPER)] * D

    options = {
        "population_size": 0,
        "maxiters": -1,
        "x_tol": -1.0,
        "f_tol": -1.0,
    }

    optimizer = mpy.Minimizer(
        func=tracked_batch,
        x0=[start_point.tolist()],
        bounds=bounds,
        algo="ARRDE",
        maxevals=RUN_BUDGET,
        seed=run_seed,
        options=options,
    )

    result = optimizer.optimize()

    return best_f, result, run_seed


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 88)
    print(f"CEC2017 F{FUNCTION} / D={D}")
    print("=" * 88)

    fopt = CEC2017_BIASES[FUNCTION]

    problem = mpy.CEC2017Functions(
        function_number=FUNCTION,
        dimension=D,
    )

    benchmark_f_opt = float(problem.get_f_opt())

    print(f"Dimension             : {D}")
    print(f"Function              : F{FUNCTION}")
    print(f"Number of runs        : {N}")
    print(f"Budget per run        : {RUN_BUDGET:,}")
    print(f"Total evaluation bud. : {N * RUN_BUDGET:,}")
    print(f"Sphere radius         : {RADIUS}")
    print(f"Starting-point seed   : {START_SEED}")
    print(f"ARRDE base seed       : {BASE_SEED}")
    print(f"Benchmark f_opt       : {benchmark_f_opt:.15e}")
    print()

    if abs(benchmark_f_opt - fopt) > 1e-9:
        raise RuntimeError(
            f"Unexpected benchmark optimum: "
            f"library={benchmark_f_opt:.15e}, "
            f"expected={fopt:.15e}"
        )

    # --------------------------------------------------------
    # Generate starting-point list
    # --------------------------------------------------------

    start_points = generate_sphere_points()

    distances = np.linalg.norm(
        start_points - CENTER,
        axis=1,
    )

    print("Starting points")
    print("-" * 88)
    print("run   distance from optimum")
    print("---   --------------------")

    for i, distance in enumerate(distances, start=1):
        print(
            f"{i:3d}   {distance:.15f}"
        )

    print()

    # --------------------------------------------------------
    # Run ARRDE independently from every starting point
    # --------------------------------------------------------

    results = []

    global_best = np.inf
    global_best_run = None

    print("Starting independent ARRDE runs...")
    print()

    for run_index, start_point in enumerate(start_points):

        distance = float(
            np.linalg.norm(start_point - CENTER)
        )

        print("=" * 88)
        print(
            f"RUN {run_index + 1}/{N}   "
            f"seed={BASE_SEED + run_index}   "
            f"start_distance={distance:.15f}"
        )
        print("=" * 88)

        best_f, result, run_seed = run_arrde(
            problem=problem,
            start_point=start_point,
            run_index=run_index,
            fopt=fopt,
        )

        error = abs(best_f - fopt)

        results.append(
            {
                "run": run_index + 1,
                "seed": run_seed,
                "start_distance": distance,
                "best_f": best_f,
                "error": error,
                "message": result.message,
            }
        )

        print()
        print(
            f"RUN {run_index + 1:2d} FINISHED: "
            f"best={best_f:.15e} "
            f"error={error:.15e}"
        )
        print()

        if best_f < global_best:
            global_best = best_f
            global_best_run = run_index + 1

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 88)
    print("FINAL SUMMARY")
    print("=" * 88)

    print(
        f"Global best : {global_best:.15e}"
    )
    print(
        f"Global error: {abs(global_best - fopt):.15e}"
    )
    print(
        f"Found in run: {global_best_run}"
    )

    print()
    print(
        "run   start_distance            best_f                  error"
    )
    print(
        "---   --------------------   ----------------------   ----------------------"
    )

    for r in results:
        print(
            f"{r['run']:3d}   "
            f"{r['start_distance']:20.15f}   "
            f"{r['best_f']:22.15e}   "
            f"{r['error']:22.15e}"
        )

    print("=" * 88)


if __name__ == "__main__":
    main()
