#!/usr/bin/env python3

import numpy as np
import minionpy as mpy

# ============================================================
# CONFIGURATION
# ============================================================

D = 20
FUNCTION = 24

# Number of independent ARRDE runs.
N = 20

# Evaluation budget for EACH run.
#
# 20 runs * 1,000,000 evaluations = 20,000,000 total evaluations.
RUN_BUDGET = 1_000_000

# Sphere from which the starting points are taken.
RADIUS = 80.0
START_SEED = 20260926

# Each ARRDE run gets a different seed:
# run_seed = BASE_SEED + run_index
BASE_SEED = 20261001

LOWER = -100.0
UPPER = 100.0

PROGRESS_STEP = 1_000_000


# ============================================================
# CEC2017 GLOBAL OPTIMUM
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
# EXACT CEC2017 F24 GLOBAL OPTIMUM
# ============================================================

CENTER_FULL = np.array(
    [
         6.1770164235388009e+01,
        -8.2403550782428852e+00,
        -2.1511735494021202e+01,
         1.0662928429740914e+01,
        -2.8853127679253522e+01,
         3.1242073274795931e+01,
         7.1837517429067532e+01,
        -6.7539761135140324e+01,
        -7.9194126903635507e+01,
         2.7532766213422995e+01,
        -4.6393239169564673e+01,
        -1.9189660498495055e+01,
         4.2161826637648900e+01,
        -2.4089553091716745e+01,
        -6.4419553958074914e+01,
         5.9118560349158273e+01,
         5.8197810408751394e+01,
        -4.5515033236478935e+01,
        -3.6031000228613152e+01,
        -5.2275229828560370e+01,
         3.0729226305489068e+01,
        -4.1726632659196220e+01,
        -7.1204381215946924e+01,
        -5.5797429587597691e+01,
         4.6560273162350185e+01,
         6.5534575947343640e+01,
         6.6593357951656618e-01,
        -2.2531432177187611e+01,
         4.7623115204099150e+01,
         7.4113636684656910e+01,
        -3.0848276622057089e+01,
         7.3336770434555021e+01,
         7.9820403352296168e+01,
         4.8545729883269281e+01,
         1.8537409388710469e+01,
         1.9368824125904972e+01,
        -8.5018770957840033e+00,
        -2.0949201630870441e+01,
        -7.5333142613681559e+01,
        -7.6936358822945678e+01,
         5.1806098284109538e+01,
        -3.7236727389151568e+01,
         7.9298277623624671e+01,
         5.1738037498545850e+01,
         3.0917797892781813e+01,
        -5.3456878366627208e+01,
        -4.3763568771892956e+01,
         5.3379421624066040e+01,
        -2.4116707328761326e+01,
        -2.6809943435159486e+01,
         7.7399355961661485e+01,
         4.1532722075252195e+01,
         5.2649695987184387e+01,
         4.0280538249891499e+01,
         6.8113197753258603e+01,
        -2.2450515586161295e+01,
         2.2442679058035221e+01,
        -3.9878093700799205e+01,
        -1.0428626892356494e+01,
         1.6027250185621554e+01,
        -3.3161047861520689e+00,
         6.5025596961856635e+01,
         6.6764165328240097e+01,
         6.8854940565701082e+00,
        -7.9864721767541354e+01,
         3.7355499754359712e+01,
        -4.5822966909884926e+01,
        -5.9619434429155397e+01,
        -6.3923694968188038e+01,
        -4.8424968093831545e+01,
        -2.8381091518505048e+00,
        -6.6200193912815394e+01,
         6.8434223072805551e+01,
         7.4061174211120772e+01,
         7.7531597873909215e+00,
         5.5798711996431791e+01,
        -5.6793104242902764e+01,
        -1.9907628158064384e+01,
         1.2368268763817971e+01,
        -5.4005064736122762e+01,
        -6.8553665267561399e+01,
         7.9767891622971689e+01,
         7.7294275997809265e+01,
        -7.2658773135700628e+01,
        -1.5513902239275126e+00,
        -4.8027295485151605e+01,
         2.3076480056660472e+01,
         5.1329865501881514e+01,
         4.2295219180311648e+01,
        -7.0992919433393013e+01,
        -6.3346493288570315e+01,
        -4.0895449478596191e+01,
         6.7474103765317281e+01,
        -7.8014576955705223e+01,
         5.6589882131940627e+01,
         3.3534003146707079e+01,
         1.9492026807503787e+01,
        -5.5680453691482754e+01,
         6.6958470278865875e+01,
        -3.5320927614029927e+01,
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

if CENTER.shape != (D,):
    raise RuntimeError(
        f"CENTER has shape {CENTER.shape}, expected ({D},)"
    )

if not np.all(
    (CENTER >= LOWER)
    & (CENTER <= UPPER)
):
    raise RuntimeError(
        "CENTER is outside the CEC2017 bounds"
    )


# ============================================================
# MINIONPY CEC2017 F24
# ============================================================

cec = mpy.CEC2017Functions(
    function_number=FUNCTION,
    dimension=D,
)

benchmark_f_opt = float(
    cec.get_f_opt()
)

fopt = CEC2017_BIASES[FUNCTION]

if abs(benchmark_f_opt - fopt) > 1e-9:
    raise RuntimeError(
        f"Unexpected benchmark optimum: "
        f"library={benchmark_f_opt:.15e}, "
        f"expected={fopt:.15e}"
    )


# ============================================================
# GENERATE SPHERE STARTING POINTS
# ============================================================

def generate_sphere_points():
    rng = np.random.default_rng(START_SEED)

    sphere_points = []
    attempts = 0
    max_attempts = max(
        1_000,
        N * 1_000_000,
    )

    while len(sphere_points) < N:
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
        if np.all(
            (point >= LOWER)
            & (point <= UPPER)
        ):
            sphere_points.append(point)

    return np.asarray(
        sphere_points,
        dtype=float,
    )


# ============================================================
# RUN ONE INDEPENDENT ARRDE
# ============================================================

def run_arrde(
    start_point,
    run_index,
):
    run_seed = BASE_SEED + run_index

    evals = 0
    best_f = np.inf
    next_progress = PROGRESS_STEP

    def tracked_batch(X):
        nonlocal evals, best_f, next_progress

        fs = np.asarray(
            cec(X),
            dtype=float,
        ).reshape(-1)

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

    bounds = [
        (LOWER, UPPER)
    ] * D

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

    print(
        f"Number of runs        : {N}"
    )
    print(
        f"Budget per run        : {RUN_BUDGET:,}"
    )
    print(
        f"Total evaluation bud. : {N * RUN_BUDGET:,}"
    )
    print(
        f"Sphere radius         : {RADIUS}"
    )
    print(
        f"Starting-point seed   : {START_SEED}"
    )
    print(
        f"ARRDE base seed       : {BASE_SEED}"
    )
    print(
        f"Benchmark f_opt       : "
        f"{benchmark_f_opt:.15e}"
    )
    print()

    # --------------------------------------------------------
    # Generate exactly N starting points.
    # --------------------------------------------------------

    start_points = generate_sphere_points()

    distances = np.linalg.norm(
        start_points - CENTER,
        axis=1,
    )

    print("Starting points")
    print("-" * 88)
    print(
        "run   distance from optimum"
    )
    print(
        "---   --------------------"
    )

    for i, distance in enumerate(
        distances,
        start=1,
    ):
        print(
            f"{i:3d}   "
            f"{distance:.15f}"
        )

    print()

    # --------------------------------------------------------
    # Run ARRDE independently from every point.
    # --------------------------------------------------------

    results = []

    global_best = np.inf
    global_best_run = None

    print(
        "Starting independent ARRDE runs..."
    )
    print()

    for run_index, start_point in enumerate(
        start_points
    ):

        distance = float(
            np.linalg.norm(
                start_point - CENTER
            )
        )

        print("=" * 88)
        print(
            f"RUN {run_index + 1}/{N}   "
            f"seed={BASE_SEED + run_index}   "
            f"start_distance={distance:.15f}"
        )
        print("=" * 88)

        best_f, result, run_seed = run_arrde(
            start_point=start_point,
            run_index=run_index,
        )

        error = abs(
            best_f - fopt
        )

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
        f"Global fbest : {global_best:.15e}"
    )
    print(
        f"Global error : "
        f"{abs(global_best - fopt):.15e}"
    )
    print(
        f"Found in run : {global_best_run}"
    )

    print()
    print(
        "run   start_distance            best_f                  error"
    )
    print(
        "---   --------------------   ----------------------   "
        "----------------------"
    )

    for r in results:
        print(
            f"{r['run']:3d}   "
            f"{r['start_distance']:20.15f}   "
            f"{r['best_f']:22.15e}   "
            f"{r['error']:22.15e}"
        )

    # --------------------------------------------------------
    # Requested global-best line.
    # --------------------------------------------------------

    print()
    print(
        f"GLOBAL FBEST   {global_best:.15e}"
    )

    print("=" * 88)


if __name__ == "__main__":
    main()
