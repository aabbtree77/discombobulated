#!/usr/bin/env python3
import numpy as np
import minionpy as mpy

# ============================================================
# SETTINGS
# ============================================================
D = 30                       # any of: 2, 10, 20, 30, 50, 100
FUNCTION = 25

# Physical Euclidean radius of the sphere.

# To get vicinity, examine various values manually by a quick bisection 
# so that the list printed in the terminal starts showing values 2499,
# which is how we define vicinity, the algorithms gets trapped in 2500,
# and we are interested in the maximal radius away from the optimum with
# at least one value below 2500.

# Similarly with F24 in the other script, see README.md on how this is used
# in worst case analysis of the problem complexity and how F24 compares to
# F25.

RADIUS = 1.6

# Number of valid points sampled on the sphere.
N = 20
SEED = 20260926
LOWER = -100.0
UPPER = 100.0

# ============================================================
# EXACT CEC2017 F25 GLOBAL OPTIMUM (first shift vector)
# ============================================================
#
# Official first row of shift_data_25.txt (length 100).
# For any supported dimension D the optimum is simply
# CENTER_FULL[:D].
#
# This is the location of the global optimum of the
# composition function (the component with bias 0).
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
       -19.584820637286093e+00,
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
# MINIONPY CEC2017 F25
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
print(f"CEC2017 F25 / D={D}")
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
