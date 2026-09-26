#!/usr/bin/env python3
import numpy as np
import minionpy as mpy

# ============================================================
# SETTINGS
# ============================================================
D = 30                     # any of: 2, 10, 20, 30, 50, 100
FUNCTION = 24

# Physical Euclidean radius of the sphere.

# To get vicinity, examine various values manually by a quick bisection 
# so that the list printed in the terminal starts showing values 2499,
# which is how we define vicinity, the algorithms gets trapped in 2500,
# and we are interested in the maximal radius away from the optimum with
# at least one value below 2500.

# Similarly with F25 in the other script, see README.md on how this is used
# in worst case analysis of the problem complexity and how F24 compares to
# F25.

RADIUS = 10.0

# Number of valid points sampled on the sphere.
N = 20
SEED = 20260926
LOWER = -100.0
UPPER = 100.0

# ============================================================
# EXACT CEC2017 F24 GLOBAL OPTIMUM (first shift vector)
# ============================================================
#
# Official first row of shift_data_24.txt (length 100).
# For any supported dimension D the optimum is simply
# CENTER_FULL[:D].
#
# This is the location of the global optimum of the
# composition function (the component with bias 0).
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

# ============================================================
# VALIDATE CENTER
# ============================================================
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

# ============================================================
# SPHERE SAMPLING
# ============================================================
#
# Random directions are drawn uniformly from the D-dimensional
# unit sphere.
#
# A point is accepted only if the resulting physical coordinate
# remains inside [-100,100]^D. It is NOT clipped, because clipping
# would change its radius and therefore no longer be a sphere point.
# ============================================================
rng = np.random.default_rng(
    SEED
)
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
    direction = rng.normal(
        size=D
    )
    norm = np.linalg.norm(
        direction
    )
    if norm == 0.0:
        continue
    direction /= norm
    point = (
        CENTER
        + RADIUS * direction
    )
    if np.all(
        (point >= LOWER)
        & (point <= UPPER)
    ):
        sphere_points.append(
            point
        )
sphere_points = np.asarray(
    sphere_points,
    dtype=float,
)

# ============================================================
# EVALUATE CENTER + SPHERE
# ============================================================
all_points = np.vstack(
    [
        CENTER,
        sphere_points,
    ]
)
all_f = np.asarray(
    cec(all_points),
    dtype=float,
).reshape(-1)
center_f = float(
    all_f[0]
)
sphere_f = all_f[1:]

# ============================================================
# VERIFY DISTANCES
# ============================================================
sphere_distances = np.linalg.norm(
    sphere_points - CENTER,
    axis=1,
)

# ============================================================
# OUTPUT
# ============================================================
print()
print("=" * 78)
print(f"CEC2017 F24 / D={D}")
print("=" * 78)
print(
    f"Benchmark f_opt : {benchmark_f_opt:.15e}"
)
print(
    f"Evaluated f(CENTER): {center_f:.15e}"
)
print(
    f"Difference from benchmark optimum: "
    f"{center_f - benchmark_f_opt:.15e}"
)
print()
print("Global optimum coordinates:")
for i, value in enumerate(
    CENTER,
    start=1,
):
    print(
        f"x[{i:2d}] = {value: .16e}"
    )
print()
print(
    f"Sphere radius requested : {RADIUS:.12f}"
)
print(
    f"Number of sphere points : {N}"
)
print(
    f"Random seed             : {SEED}"
)
print()
print(
    "idx   distance from CENTER          f(x)"
)
print(
    "---   ----------------------   ----------------------"
)
for i, (distance, f) in enumerate(
    zip(
        sphere_distances,
        sphere_f,
    ),
    start=1,
):
    print(
        f"{i:3d}   "
        f"{distance:22.15f}   "
        f"{f:22.15e}"
    )
print()
print(
    f"Sphere min  f = {np.min(sphere_f):.15e}"
)
print(
    f"Sphere max  f = {np.max(sphere_f):.15e}"
)
print(
    f"Sphere mean f = {np.mean(sphere_f):.15e}"
)
print(
    f"Best sphere point is "
    f"{np.argmin(sphere_f) + 1}"
    f" with f = {np.min(sphere_f):.15e}"
)
print()
print(
    "Maximum radius error: "
    f"{np.max(np.abs(sphere_distances - RADIUS)):.3e}"
)
print("=" * 78)
