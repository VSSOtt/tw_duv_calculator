#!/usr/bin/env python3
"""
Ermittelt Worst‑Case‑Δuv‑Werte für Tunable‑White‑Paare
und speichert das Ergebnis als JSON‑Datei.

• Vektorisiert + multiprozessiert (ProcessPoolExecutor)
• Konstanten ELLIPSE_N und MIX_STEPS können
  auf kleinere Werte gesetzt werden, um schnelle
  Tests zu erlauben.

Getestet mit Python 3.12 / NumPy 1.26  (8‑Core CPU).
Berechnungsdauer ca. 24s ohne colour-Bibliothek, ca. 90s mit colour-Bibliothek.
"""

from __future__ import annotations
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
import argparse, os, time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "MKL_NUM_THREADS", "BLIS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

# Auf True setzen, um die colour library für CCT-Berechnungen zu verwenden
# Dies ist genauer, aber langsamer und erfordert die Installation der Bibliothek.
USE_COLOUR_LIB = False  
if USE_COLOUR_LIB:
    try:
        import colour
    except ImportError:
        print("colour library not found. Install it with `pip install colour-science`.")
        USE_COLOUR_LIB = False

# ────────── Konfiguration ──────────
ELLIPSE_N      = 40         # Randpunkte pro Ellipse (nur unterer Halbkreis) - Performance!
MIX_STEPS      = 400        # Anzahl Blendkoeffizienten 0…1 - Performance!
PLANCK_MIN_K   = 1700       # Planck‑Lokus minimale Temperatur
PLANCK_MAX_K   = 7000       # Planck‑Lokus maximale Temperatur
PLANCK_N       = 5300       # Anzahl Planck‑Lokus‑Punkte (1K-Schritte)
TARGET_PRESET  = 2800       # Preset‑Temperatur
TARGET_TOL     = 10         # Toleranz ± K für Preset (Findet worst-Case Δuv in der Nähe von TARGET_PRESET)

BIN_TABLE = {
#   CCT    x       y       a        b        θ
    1800: (0.5496, 0.4081, 0.00417, 0.00741, 80.00),
    2000: (0.5251, 0.4120, 0.00723, 0.00400, 51.00),
    2200: (0.5054, 0.4012, 0.01250, 0.00700, 49.64),
    2400: (0.4890, 0.4182, 0.00810, 0.00420, 53.70),
    2700: (0.4578, 0.4101, 0.00810, 0.00420, 53.70),
    3000: (0.4338, 0.4030, 0.00834, 0.00408, 53.22),
    4000: (0.3818, 0.3797, 0.00939, 0.00402, 53.72),
    5000: (0.3447, 0.3553, 0.00822, 0.00354, 59.62),
    6500: (0.3123, 0.3282, 0.00690, 0.00285, 58.57),
}

# ────────── LED pairs ──────────
PAIR_LIST = [
    (1800, 3000),
    (1800, 4000),
    (2400, 5000),
    (2700, 5000),
    (2700, 6500),
    (2200, 5000),
    (2000, 5000),
]



# ───────── Global variables ──────────
temps = None  # Planckian locus temperatures in Kelvin
planck_uv = None  # Planckian locus points in uv coordinates

# ───────── Basic functions ──────────
def ellipse_lower_points(xc, yc, a, b, theta_deg, n, k=4):
    """
    Calculate lower half points of a rotated ellipse.

    This function computes the points that form the lower half of an ellipse
    given its center coordinates, semi-major and semi-minor axes, rotation 
    angle, and the number of points to generate. The ellipse is considered 
    in the x-y plane and is rotated by a specified angle.

    Parameters:
    xc (float): x-coordinate of the ellipse center.
    yc (float): y-coordinate of the ellipse center.
    a (float): Semi-major axis length of the ellipse.
    b (float): Semi-minor axis length of the ellipse.
    theta_deg (float): Rotation angle of the ellipse in degrees.
    n (int): Number of points to generate along the lower half of the ellipse.

    Returns:
    np.ndarray: An array containing the x and y coordinates of the points 
    forming the lower half of the ellipse.
    """

    # Oversample by a factor of k to obtain n points
    t = np.linspace(np.pi, 2*np.pi, n * k, endpoint=False)
    c, s = np.cos(np.deg2rad(theta_deg)), np.sin(np.deg2rad(theta_deg))
    ct, st = np.cos(t), np.sin(t)

    x = xc + a*ct*c - b*st*s
    y = yc + a*ct*s + b*st*c

    # Only keep points of the lower half of the ellipse
    msk = y <= yc
    x, y = x[msk], y[msk]

    # If not enough points were obtained, double the oversampling
    if x.size < n:
        return ellipse_lower_points(xc, yc, a, b, theta_deg, n, k*2)

    # Select n points from the oversampled array
    idx = np.linspace(0, x.size - 1, n).round().astype(int)
    return np.column_stack((x[idx], y[idx]))

def cct_to_xy(T):
    """
    Approximates the xy chromaticity coordinates from a given correlated color
    temperature (CCT) in Kelvin using the Kim/Kang-2002 model.

    Parameters:
    T (float): Correlated color temperature (CCT) in Kelvin.

    Returns:
    tuple: A tuple containing the x and y chromaticity coordinates, both in the
    range [0, 1].
    """
    if USE_COLOUR_LIB:
        xy = colour.temperature.CCT_to_xy(T)
        return xy[0], xy[1]

    T = np.clip(np.asarray(T, float), 1667.0, 25000.0)

    x = np.where(
        T <= 4000.0,
        -0.2661239e9 / T**3 - 0.2343580e6 / T**2 + 0.8776956e3 / T + 0.179910,
        -3.0258469e9 / T**3 + 2.1070379e6 / T**2 + 0.2226347e3 / T + 0.240390,
    )

    y = np.select(
        [T <= 2222.0, T <= 4000.0],
        [
            -1.1063814*x**3 - 1.3481102*x**2 + 2.18555832*x - 0.20219683,
            -0.9549476*x**3 - 1.37418593*x**2 + 2.09137015*x - 0.16748867,
        ],
        3.0817580*x**3 - 5.8733867*x**2 + 3.75112997*x - 0.37001483,
    )
    return x, y

def xy_to_uv(x, y):
    """
    Converts xy chromaticity coordinates to uv coordinates.

    This function calculates the corresponding u and v coordinates
    from given x and y chromaticity coordinates using CIE 1960 UCS
    transformation formula.

    Parameters:
    x (array_like or float): x chromaticity coordinate or array of coordinates.
    y (array_like or float): y chromaticity coordinate or array of coordinates.

    Returns:
    tuple: A tuple containing the u and v chromaticity coordinates.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    d = -2.0 * x + 12.0 * y + 3.0
    return 4.0 * x / d, 6.0 * y / d

def cct_mccamy(x, y, *, use_colour=USE_COLOUR_LIB):
    """
    Approximates the correlated color temperature (CCT) from a given xy
    chromaticity coordinate pair using the McCamy equation.
    The error of this approximation is less than 2 K in the range 2856 K to
    6504 K. For lower and higher temperatures, the error increases significantly.

    Parameters:
    x (float): x chromaticity coordinate.
    y (float): y chromaticity coordinate.

    Returns:
    float: The correlated color temperature in Kelvin.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if use_colour:
        xy = np.column_stack([x.ravel(), y.ravel()])
        CCT = colour.xy_to_CCT(xy, method="McCamy 1992")
        return CCT.reshape(x.shape)

    n = (x - 0.3320) / (y - 0.1858)
    CCT = -449 * np.power(n, 3) + 3525 * np.power(n, 2) - 6823.3 * n + 5520.33

    return CCT

def cct_hernandez(x, y, *, use_colour=USE_COLOUR_LIB):
    """
    Approximates the correlated color temperature (CCT) from a given xy chromaticity
    coordinate pair using the Hernandez equation.
    This equation is more accurate for lower temperatures, but uses more complex
    calculations than the McCamy equation. It is approximately 5 times slower.

    Parameters:
    x (float): x chromaticity coordinate.
    y (float): y chromaticity coordinate.

    Returns:
    float: The correlated color temperature in Kelvin.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if use_colour:
        xy = np.column_stack([x.ravel(), y.ravel()])
        CCT = colour.xy_to_CCT(xy, method="Hernandez 1999")
        return CCT.reshape(x.shape)
    
    xe, ye = 0.3366, 0.1735          # „low-T epicenter“
    n = (x - xe) / (y - ye)
    return (-949.86315
            + 6253.80338*np.exp(-n/0.92159)
            +   28.70599*np.exp(-n/0.20039)
            +    0.00004*np.exp(-n/0.07125))

def xy_to_cct(x, y, *, use_colour=USE_COLOUR_LIB):
    """
    Approximates the correlated color temperature (CCT) from xy chromaticity
    coordinates using a hybrid approach combining McCamy and Hernández models.
    
    The McCamy model is used for the range 2856–6504 K, where its error is
    less than or equal to ±2 K. For values outside this range, the Hernández
    model is used for improved accuracy.

    Parameters:
    x (float or array_like): x chromaticity coordinate.
    y (float or array_like): y chromaticity coordinate.

    Returns:
    float or ndarray: The correlated color temperature(s) in Kelvin.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Calculate CCT using McCamy's approximation for speed
    cct_m = cct_mccamy(x, y, use_colour=use_colour)

    # Identify where McCamy's model is inaccurate and apply Hernández's model
    mask = (cct_m < 2856.0) | (cct_m > 6504.0)
    if np.any(mask):
        cct_h = cct_hernandez(x, y, use_colour=use_colour)
        cct   = np.where(mask, cct_h, cct_m)
    else:
        cct = cct_m

    # Return Python float if the result is scalar, else return ndarray
    return float(cct) if cct.ndim == 0 else cct

def signed_duv(u, v):
    """
    Signed Δuv to the Planck locus (Colour definition):
    +Δuv above (towards green), -Δuv below (magenta).
    
    Parameters:
    x (float): x chromaticity coordinate.
    y (float): y chromaticity coordinate.
    
    Globals:
    planck_uv : (N, 2) array_like
        Precomputed Planck locus points (u, v). Must be strictly monotonic
        in the CCT sequence.  Must be sorted in the same order as `temps`.
    temps : (N,) array_like
        Temperature in Kelvin. Must be sorted in the same order as `planck_uv`.

    Returns:
    float: duv_signed
        Signed Δuv (typical range ±0.06).
    temp: float
        Temperature in Kelvin.
    """
    # Find the closest point on the locus (O(N); KD-Tree for many calls later)
    du = u - planck_uv[:, 0]
    dv = v - planck_uv[:, 1]
    dist2 = du*du + dv*dv
    idx = int(np.argmin(dist2))

    # Calculate the local tangent at the locus (forward/backward difference at the edges)
    if idx == 0:
        t = planck_uv[1]   - planck_uv[0]
    elif idx == len(planck_uv) - 1:
        t = planck_uv[-1]  - planck_uv[-2]
    else:
        t = planck_uv[idx + 1] - planck_uv[idx - 1]

    # Rotate the normal vector **counter-clockwise**
    n = np.array([t[1], -t[0]], float)
    n /= np.hypot(*n)

    # Signed Δuv = scalar product to the normal vector
    duv_signed = np.dot([u - planck_uv[idx, 0], v - planck_uv[idx, 1]], n)
    return duv_signed, temps[idx]  # Δuv, Index→Temperature

def signed_duv_vec(u, v, *, block=512):
    """
    Vectorised signed Δuv + Planck temp, but memory-friendly.

    u, v : (N,)  – 1-D arrays
    block       – number of samples processed per chunk
    Returns     – two (N,) arrays: duv_signed, temps
    """
    # flatten input ---------------------------------------------------
    u = np.asarray(u, float).ravel()
    v = np.asarray(v, float).ravel()
    N = u.size

    # one-time cache of normals --------------------------------------
    global _plk_normals
    if '_plk_normals' not in globals():
        t = np.empty_like(planck_uv)
        t[1:-1] = planck_uv[2:] - planck_uv[:-2]
        t[0]    = planck_uv[1]  - planck_uv[0]
        t[-1]   = planck_uv[-1] - planck_uv[-2]
        n = np.empty_like(t)
        n[:, 0] =  t[:, 1]          # CCW normal
        n[:, 1] = -t[:, 0]
        n /= np.linalg.norm(n, axis=1)[:, None]
        _plk_normals = n            # cached for all calls

    # output buffers --------------------------------------------------
    idx   = np.empty(N, np.int32)
    duv   = np.empty(N, float)
    temp = np.empty(N, float)

    # chunked distance search ----------------------------------------
    for s in range(0, N, block):
        e   = min(s + block, N)
        uu  = u[s:e, None]                 # (b,1)
        vv  = v[s:e, None]
        d2  = (uu - planck_uv[:, 0])**2 + (vv - planck_uv[:, 1])**2
        i   = d2.argmin(axis=1)            # (b,)
        idx[s:e] = i

        vec   = np.column_stack([uu[:, 0], vv[:, 0]]) - planck_uv[i]
        duv[s:e] = np.einsum('ij,ij->i', vec, _plk_normals[i])
        temp[s:e] = temps[i]

    return duv, temp

def evaluate_pair_fast(warm_pts, cold_pts, alphas):
    # shape shortcuts
    nw = len(warm_pts)
    nc = len(cold_pts)
    na = len(alphas)

    # warm, cold, alpha as broadcast-ready tensors
    # (nw,1,1,2)  (1,nc,1,2)  (1,1,na,1)
    wp = warm_pts[:, None, None, :]
    cp = cold_pts[None, :, None, :]
    aa = alphas[None, None, :, None]

    # mix xy for the full grid → (nw,nc,na,2)
    mix_xy = (1.0 - aa) * wp + aa * cp
    mix_xy2 = mix_xy.reshape(-1, 2)

    # xy → uv
    u, v = xy_to_uv(mix_xy2[:, 0], mix_xy2[:, 1])

    # Δuv + CCT
    duv, temps_mix = signed_duv_vec(u, v)

    # reshape back to (nw,nc,na)
    duv   = duv.reshape(nw, nc, na)
    temps = temps_mix.reshape(nw, nc, na)

    # worst Δuv over the whole grid
    idx_flat = np.abs(duv).argmax()
    iw, ic, ia = np.unravel_index(idx_flat, duv.shape)

    worst = {
        "warm_point": warm_pts[iw].tolist(),
        "cold_point": cold_pts[ic].tolist(),
        "duv": float(duv[iw, ic, ia]),
        "duv_point": mix_xy[iw, ic, ia].tolist(),
        "duv_temp": int(np.rint(temps[iw, ic, ia])),
    }

    # CCT for all mix points
    ccts = xy_to_cct(mix_xy2[:, 0], mix_xy2[:, 1]).reshape(nw, nc, na)

    # mask for 2 800 K ± TOL
    mask = np.abs(ccts - TARGET_PRESET) <= TARGET_TOL
    preset = None
    if mask.any():
        idx_p = np.abs(duv[mask]).argmax()
        iw, ic, ia = np.transpose(np.nonzero(mask))[idx_p]

        preset = {
            "warm_point": warm_pts[iw].tolist(),
            "cold_point": cold_pts[ic].tolist(),
            "duv": float(duv[iw, ic, ia]),
            "duv_point": mix_xy[iw, ic, ia].tolist(),
            "duv_temp": int(np.rint(ccts[iw, ic, ia])),
        }

    return worst, preset

# ────────── Worker (vektorisiert für 1 Randpunkt‑Paar) ──────────
def worker(task):
    """
    Worker function to evaluate a pair of warm and cold points and a given alpha vector. (multiple blend steps [0, 1])
    This function computes the mixed xy coordinates for the given warm and cold points,
    converts them to uv coordinates, and calculates the signed Δuv and corresponding
    temperatures for each blend step. It then finds the worst-case Δuv and the 2800-K preset Δuv.

    Parameters:
    task (tuple): A tuple containing the warm point, cold point, and alpha vector.

    Returns:
    tuple: A tuple containing the worst-case Δuv data and the 2800-K preset Δuv data.
    """
    wp, cp, alphas = task
    # Calculate the mixed xy coordinates for the given warm and cold points
    # (1 - alphas)[:, None] creates a column vector for blending
    # alphas[:, None] creates a column vector for blending
    # The result is a 2D array where each row corresponds to a blend step
    # and each column corresponds to the x and y coordinates of the mixed point.
    mix_xy: np.ndarray = (1 - alphas)[:, None]*wp + alphas[:, None]*cp

    # Convert xy to uv coordinates
    u, v = xy_to_uv(mix_xy[:, 0], mix_xy[:, 1])
    # Calculate signed Δuv and corresponding temperatures
    # (vectorized for all blend steps)
    duvs, temps = np.vectorize(signed_duv)(u, v)

    # Find worst-case Δuv
    j = int(np.abs(duvs).argmax())
    worst = dict(warm_point=wp.tolist(), cold_point=cp.tolist(),
                 duv=float(duvs[j]), duv_point=mix_xy[j].tolist(),
                 duv_temp=int(round(temps[j])))
    
    # 2800‑K‑Preset
    # Calculate CCTs for the mixed xy points
    ccts = np.vectorize(xy_to_cct)(mix_xy[:, 0], mix_xy[:, 1])
    # Find the index of the maximum DUV within the target tolerance
    # around the target preset temperature
    mask = np.abs(ccts - TARGET_PRESET) <= TARGET_TOL
    preset = None
    if mask.any():
        # Find the index of the maximum DUV within the target tolerance
        k = int(np.abs(duvs[mask]).argmax())
        # Get the real index in the original array
        real = np.where(mask)[0][k]
        preset = dict(warm_point=wp.tolist(), cold_point=cp.tolist(),
                      duv=float(duvs[real]), duv_point=mix_xy[real].tolist(),
                      duv_temp=int(round(ccts[real])))
    return worst, preset

def init_worker_globals():
    """Build Planck-locus uv, temperature array and normal cache once."""
    global temps, planck_uv, _plk_normals

    if planck_uv is not None:
        return                      # already initialised – nothing to do

    # Planckian temps
    temps = np.linspace(PLANCK_MIN_K, PLANCK_MAX_K, PLANCK_N)

    # xy of Planck locus → uv
    if USE_COLOUR_LIB:
        CCT_D_uv = np.column_stack([temps, np.zeros_like(temps)])
        planck_uv = colour.temperature.CCT_to_uv_Ohno2013(CCT_D_uv)
    else:
        planck_uv = np.column_stack(xy_to_uv(*cct_to_xy(temps)))

    # one-time normal vectors (CCW)
    t = np.empty_like(planck_uv)
    t[1:-1] = planck_uv[2:] - planck_uv[:-2]
    t[0]    = planck_uv[1] - planck_uv[0]
    t[-1]   = planck_uv[-1] - planck_uv[-2]
    n = np.empty_like(t)
    n[:, 0] =  t[:, 1]
    n[:, 1] = -t[:, 0]
    _plk_normals = n / np.linalg.norm(n, axis=1)[:, None]

# ────────── Processing an LED pair (multiprocessing) ──────────
def evaluate_pair(warm_pts, cold_pts, alphas, max_workers):
    """
    Evaluates a pair of warm and cold points by creating a linear combination
    of both using the given alpha vector. The worst-case DUV and the 2800-K
    preset are calculated for each combination and the pair with the highest
    worst-case DUV is returned.

    Parameters:
    warm_pts (list): A list of warm points as [x, y] lists.
    cold_pts (list): A list of cold points as [x, y] lists.
    alphas (array): An array of alpha values in the range [0, 1] to evaluate.
    max_workers (int): The maximum number of worker processes to use.

    Returns:
    tuple: A tuple containing the worst-case DUV data and the 2800-K preset
    data. Each entry is a dictionary with the keys "warm_point", "cold_point",
    "duv", "duv_point", and "duv_temp".
    """

    best_worst = {"duv": 0.0}
    best_preset = {"duv": 0.0, "_f": False}
    # Create tasks for each combination of warm and cold points
    # Each task is a tuple of (warm_point, cold_point, alphas)
    tasks = ((wp, cp, alphas) for wp in warm_pts for cp in cold_pts)
    # Initialize worker globals in each worker process
    with ProcessPoolExecutor(max_workers=max_workers, initializer=init_worker_globals) as pool:
        # Map the worker function to the tasks and collect results
        # chunksize=40 is a performance optimization to reduce overhead
        for w, p in pool.map(worker, tasks, chunksize=40):
            # Update the best worst-case DUV and preset if found
            if abs(w["duv"]) > abs(best_worst["duv"]):
                best_worst = w
            if p is not None and (not best_preset["_f"] or abs(p["duv"]) > abs(best_preset["duv"])):
                best_preset = {**p, "_f": True}

    # If no preset was found, use the best worst-case DUV as the preset
    if not best_preset["_f"]:
        best_preset = {k: best_worst[k] for k in ("warm_point", "cold_point", "duv", "duv_point", "duv_temp")}
    best_preset.pop("_f", None)
    return best_worst, best_preset

# ────────── Complete Computation + JSON Export ──────────
def main(out_file: str, n_ellipse=ELLIPSE_N, n_steps=MIX_STEPS, workers=None):
    """
    Performs the complete computation of worst-case DUVs and 2800-K presets for all
    specified LED pairs and saves the results as a JSON file.

    Parameters:
    out_file (str): The path to the output JSON file.
    n_ellipse (int): The number of points to use for the ellipse approximation.
                    Defaults to ELLIPSE_N.
    n_steps (int): The number of alpha values to evaluate. Defaults to MIX_STEPS.
    workers (int): The maximum number of worker processes to use. Defaults to None
                   (i.e., the number of CPU cores available).

    Returns:
    None
    """
    t0 = time.perf_counter()

    # precalculate the alpha values and ellipse points
    alphas = np.linspace(0, 1, n_steps)
    ellipses = {c: ellipse_lower_points(*BIN_TABLE[c], n=n_ellipse) for c in BIN_TABLE}
    #avg_points = sum(len(v) for v in ellipses.values())/len(ellipses)
    #print(f"Ellipses calculated for {len(ellipses)} bins with an average of {avg_points:.1f} points each.")

    result = {"bins": {str(c): {"center": BIN_TABLE[c][:2], "a_b_angle": BIN_TABLE[c][2:]} for c in BIN_TABLE},
              "pairs": {}}
    
    # Print header
    print("─" * 70)
    print("            Worst-case Δuv              2 800 K preset")
    print(" LED pair   Δuv      CCT   x     y      Δuv      CCT   x     y")
    print("─" * 70)

    for warm, cold in PAIR_LIST:
        # Evaluate the pair of warm and cold points
        # warm and cold are the CCTs of the LED pair
        worst, preset = evaluate_pair(ellipses[warm], ellipses[cold], alphas, workers)
        # Store the results in the result dictionary
        result["pairs"][f"{warm}-{cold}"] = {"worst-case": worst, "2800": preset}
        # Print the results in a tyble formatted way
        print(f"{warm:4}-{cold:4}K  {worst['duv']:+1.5f} {worst['duv_temp']:4}K "
              f"{worst['duv_point'][0]:1.3f} {worst['duv_point'][1]:1.3f}  "
              f"{preset['duv']:+1.5f} {preset['duv_temp']:4}K "
              f"{preset['duv_point'][0]:1.3f} {preset['duv_point'][1]:1.3f}")
        
    # Write the result to the output file as JSON
    Path(out_file).write_text(json.dumps(result, indent=2))
    print(f"\n✓ JSON geschrieben → {out_file}  (Gesamt {time.perf_counter()-t0:0.1f}s)")

def main_multithreaded(out_file: str = "tunable_white_data.json",
         n_ellipse:   int = ELLIPSE_N,
         n_steps:     int = MIX_STEPS,
         workers:     int | None = None):
    """Full run – thread-parallel, JSON output, console table."""
    init_worker_globals()

    t0 = time.perf_counter()

    # alpha vector and ellipse point clouds
    alphas   = np.linspace(0, 1, n_steps)
    ellipses = {c: ellipse_lower_points(*BIN_TABLE[c], n=n_ellipse)
                for c in BIN_TABLE}

    # result container (metadata + per-pair data)
    result = {
        "bins": {str(c): {"center": BIN_TABLE[c][:2],
                          "a_b_angle": BIN_TABLE[c][2:]}
                 for c in BIN_TABLE},
        "pairs": {}
    }

    # build task list once; keeps the order of PAIR_LIST
    tasks = [(ellipses[w], ellipses[c], alphas) for w, c in PAIR_LIST]

    # console header
    print("─" * 70)
    print("            Worst-case Δuv              2 800 K preset")
    print(" LED pair   Δuv      CCT   x     y      Δuv      CCT   x     y")
    print("─" * 70)

    # default: all logical cores
    if workers is None:
        workers = os.cpu_count() or 1

    # thread pool ⇒ no pickle overhead, NumPy releases the GIL
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for (worst, preset), (warm, cold) in zip(
                pool.map(lambda t: evaluate_pair_fast(*t), tasks),
                PAIR_LIST):

            # store JSON data
            result["pairs"][f"{warm}-{cold}"] = {
                "worst-case": worst,
                "2800": preset or worst          # fallback if None
            }

            # console table line
            p = preset or worst
            print(f"{warm:4}-{cold:4}K "
                  f"{worst['duv']:+1.5f} {worst['duv_temp']:4}K "
                  f"{worst['duv_point'][0]:1.3f} {worst['duv_point'][1]:1.3f}  "
                  f"{p['duv']:+1.5f} {p['duv_temp']:4}K "
                  f"{p['duv_point'][0]:1.3f} {p['duv_point'][1]:1.3f}")

    # write JSON
    Path(out_file).write_text(json.dumps(result, indent=2))
    print(f"\n✓ JSON written → {out_file}  ({time.perf_counter()-t0:0.1f}s)")

# ────────── CLI ──────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Parallel Δuv‑Berechnung")
    p.add_argument("-o", "--output", default="tunable_white_data.json")
    p.add_argument("--ellipse", type=int, default=ELLIPSE_N, help="Randpunkte pro Ellipse")
    p.add_argument("--steps",   type=int, default=MIX_STEPS,   help="Blendsteps 0–1")
    p.add_argument("--workers", type=int, default=os.cpu_count(), help="Prozesse (Default: alle Kerne)")
    args = p.parse_args()
    main(args.output, args.ellipse, args.steps, args.workers)
    #main_multithreaded(args.output, args.ellipse, args.steps, args.workers)
