# plot_tunable_white.py
#!/usr/bin/env python3
"""
Erzeugt ein CIE‑1931‑Diagramm mit Beschriftung der Bin‑Farbtemperaturen,
Schwarzkörperkurve, SDCM‑Ellipsen und Worst‑Case‑Linien.
"""

import argparse
import json
from math import pi
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Parameter
# ──────────────────────────────────────────────────────────────────────────────
ELLIPSE_N = 120
IMG_RES = 300
X_MIN, X_MAX = 0.28, 0.56
Y_MIN, Y_MAX = 0.30, 0.45
PLANCK_MIN_K = 1500
PLANCK_MAX_K = 10000
PLANCK_N = 600


# ──────────────────────────────────────────────────────────────────────────────
# Hilfsfunktionen
# ──────────────────────────────────────────────────────────────────────────────
def ellipse_points(xc, yc, a, b, theta_deg, n=ELLIPSE_N, factor=1):
    theta = np.deg2rad(theta_deg)
    t = np.linspace(0, 2 * pi, n, endpoint=False)
    cos_t, sin_t = np.cos(t), np.sin(t)
    a *= factor
    b *= factor
    x = xc + (a * cos_t) * np.cos(theta) - (b * sin_t) * np.sin(theta)
    y = yc + (a * cos_t) * np.sin(theta) + (b * sin_t) * np.cos(theta)
    return x, y


def cct_to_xy(T):
    T = np.clip(T, 1667, 25000)
    if T <= 4000:
        x = -0.2661239e9 / T ** 3 - 0.2343580e6 / T ** 2 + 0.8776956e3 / T + 0.179910
    else:
        x = -3.0258469e9 / T ** 3 + 2.1070379e6 / T ** 2 + 0.2226347e3 / T + 0.240390

    if T <= 2222:
        y = -1.1063814 * x ** 3 - 1.34811020 * x ** 2 + 2.18555832 * x - 0.20219683
    elif T <= 4000:
        y = -0.9549476 * x ** 3 - 1.37418593 * x ** 2 + 2.09137015 * x - 0.16748867
    else:
        y = 3.0817580 * x ** 3 - 5.87338670 * x ** 2 + 3.75112997 * x - 0.37001483
    return x, y


def xy_to_xyz(x, y, Y_val=1.0):
    X = (x / y) * Y_val
    Z = ((1 - x - y) / y) * Y_val
    return X, Y_val, Z


def xyz_to_srgb(X, Y, Z):
    M = np.array(
        [
            [3.2406, -1.5372, -0.4986],
            [-0.9689, 1.8758, 0.0415],
            [0.0557, -0.2040, 1.0570],
        ]
    )
    R, G, B = M @ np.array([X, Y, Z])
    rgb = np.maximum([R, G, B], 0)
    mask = rgb <= 0.0031308
    rgb[mask] *= 12.92
    rgb[~mask] = 1.055 * np.power(rgb[~mask], 1 / 2.4) - 0.055
    return np.clip(rgb, 0, 1)

def generate_srgb_background(res, x_range, y_range):
    img = np.ones((res, res, 3))
    for i in range(res):
        for j in range(res):
            x = x_range[0] + (j / (res - 1)) * (x_range[1] - x_range[0])
            y = y_range[0] + (i / (res - 1)) * (y_range[1] - y_range[0])
            if x < 0 or y <= 0 or x + y > 1:
                continue
            X, Y, Z = xy_to_xyz(x, y)
            rgb = xyz_to_srgb(X, Y, Z)
            img[i, j] = rgb
    return img


# ──────────────────────────────────────────────────────────────────────────────
# Plot
# ──────────────────────────────────────────────────────────────────────────────
def main(json_path):
    data = json.loads(Path(json_path).read_text())

    # Hintergrundbild generieren (sRGB‑Gamut)
    bg_img = generate_srgb_background(IMG_RES, (X_MIN, X_MAX), (Y_MIN, Y_MAX))

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(bg_img, extent=[X_MIN, X_MAX, Y_MIN, Y_MAX], origin="lower", aspect="auto")

    # Planck‑Lokus
    temps = np.linspace(PLANCK_MIN_K, PLANCK_MAX_K, PLANCK_N)
    planck_xy = np.array([cct_to_xy(T) for T in temps])
    ax.plot(planck_xy[:, 0], planck_xy[:, 1], c="black", lw=1.4, label="Schwarzkörperkurve")

    # Ellipsen + Temperatur‑Beschriftung
    for cct, info in data["bins"].items():
        xc, yc = info["center"]
        a, b, ang = info["a_b_angle"]
        ex, ey = ellipse_points(xc, yc, a, b, ang)
        ax.plot(ex, ey, c="black", lw=0.8)
        ax.text(
            xc,
            yc + 0.01,
            f"{cct} K",
            color="black",
            fontsize=8,
            ha="center",
            va="center",
            zorder=5,
        )

    # Worst‑Case‑Punkte & Linien
    for pair_data in data["pairs"].values():
        wp = pair_data["worst-case"]["warm_point"]
        cp = pair_data["worst-case"]["cold_point"]
        dp = pair_data["worst-case"]["duv_point"]

        #ax.plot(*wp, marker="x", c="black")
        #ax.plot(*cp, marker="x", c="black")
        ax.plot(*dp, marker="o", ms=4, c="black")
        ax.plot([wp[0], cp[0]], [wp[1], cp[1]], ls="--", lw=1, c="black")

    # Diagrammgestaltung
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_xlabel("x (CIE 1931)")
    ax.set_ylabel("y (CIE 1931)")
    ax.set_aspect("equal", "box")
    ax.grid(which="both", linestyle=":", linewidth=0.5, color="white", alpha=0.5)
    ax.set_title("CIE 1931 – Schwarzkörperkurve & Worst‑Case Randmischungen", pad=12)
    ax.legend(loc="lower right")

    first_pair = list(data["pairs"].values())[0]
    target_k = next((k for k in first_pair if k != "worst-case"), None)

    if not target_k:
        print("Keine Zieltemperatur gefunden.")
        return

    table_data = [
        ["Temperatur", "Worst-Case Δuv", "Worst-Case K", f"{target_k} K Δuv"],
    ]
    for pair_name, pair_data in data["pairs"].items():
        duv = round(pair_data["worst-case"]["duv"], 4)
        worst_temp = pair_data["worst-case"]["duv_temp"]
        target_duv = round(pair_data.get(target_k, {}).get("duv", 0), 4)
        table_data.append(
            [
                f"{pair_name.replace('-', ' → ')}",
                f"{duv}",
                f"{worst_temp} K",
                f"{target_duv}",
            ]
        )

    table = plt.table(
        cellText=table_data,
        colLabels=None,
        loc="lower center",
        bbox=[0.1, -0.5, 0.8, 0.3],
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot CIE‑Diagramm mit Beschriftung.")
    parser.add_argument("--json", default="tunable_white_data.json", help="Eingabe‑Datei")
    main(parser.parse_args().json)
