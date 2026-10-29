from dataclasses import replace
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cam_kernel import Params, Model, fit_shapes, gauss, expo

CK, CN = "#228B22", "#C71585"   # CaMKII, CaN (same hues as the original notebook)
SIGMA_K_FIT, SIGMA_N_FIT = 1.0, 1.19   # push-pull fit (control parameters)

# representative case: literature constants, strong stimulation
p = replace(Params(), D_c=13.0, tau_c=0.05, influx=1000.0)
r = Model(p).run()
x = r["x"]
rows = pickle.load(open("sweep.pkl", "rb"))

fig, axs = plt.subplots(1, 3, figsize=(12, 3.6), dpi=200, constrained_layout=True)

# (a) profiles + fits
ax = axs[0]
for key, col, lab in (("int_k", CK, "CaMKII"), ("int_n", CN, "CaN")):
    y = r[key] / r[key].max()
    f = fit_shapes(x, y)
    ax.plot(x, y, color=col, lw=2.5, label=f"{lab} (simulation)")
    ax.plot(x, gauss(x, f["gauss"]["width"], f["gauss"]["amp"]), color=col, lw=1,
            ls="--")
    ax.plot(x, expo(x, f["exp"]["width"], f["exp"]["amp"]), color=col, lw=1,
            ls=":")
    ax.text(0.98, 0.93 if key == "int_k" else 0.80,
            f"{lab}: σ = {f['gauss']['width']:.2f} µm", color="0.2",
            transform=ax.transAxes, ha="right", fontsize=8)
ax.plot([], [], color="0.4", lw=1, ls="--", label="Gaussian fit")
ax.plot([], [], color="0.4", lw=1, ls=":", label="exponential fit")
for s in (-2, -1, 0, 1, 2):
    ax.axvline(s, color="0.85", lw=0.8, zorder=0)
ax.set_xlim(-6, 6)
ax.set_xlabel("distance from stimulated spine (µm)")
ax.set_ylabel("integrated activation (norm.)")
ax.set_title("a  activation profiles", loc="left", fontweight="bold", fontsize=10)
ax.legend(frameon=False, fontsize=7, loc="upper left")

# (b) widths vs sqrt(D_m / koff_m)
ax = axs[1]
for lab, mk in (("weak", "o"), ("strong", "s")):
    sel = [rw for rw in rows if rw[2] == lab]
    L = np.array([rw[3] for rw in sel])
    ax.scatter(L, [rw[5] for rw in sel], color=CK, marker=mk, s=28,
               label=f"CaMKII, {lab}", zorder=3)
    ax.scatter(L, [rw[8] for rw in sel], facecolor="none", edgecolor=CN,
               marker=mk, s=28, label=f"CaN, {lab}", zorder=3)
ax.axhline(SIGMA_K_FIT, color=CK, lw=1, ls="-.")
ax.axhline(SIGMA_N_FIT, color=CN, lw=1, ls="-.")
ax.text(2.6, SIGMA_K_FIT - 0.35, "push-pull fit σ_K, σ_N", fontsize=7, color="0.3",
        ha="right")
ax.set_xlabel(r"$\sqrt{D_m / k_{off}^m}$  (µm)")
ax.set_ylabel("Gaussian width σ (µm)")
ax.set_title("b  width set by Ca–CaM mobility", loc="left", fontweight="bold",
             fontsize=10)
ax.legend(frameon=False, fontsize=7, loc="upper left")

# (c) goodness of fit, Gaussian vs exponential (CaMKII)
ax = axs[2]
for lab, mk in (("weak", "o"), ("strong", "s")):
    sel = [rw for rw in rows if rw[2] == lab]
    ax.scatter([rw[7] for rw in sel], [rw[6] for rw in sel], color=CK, marker=mk,
               s=28, label=f"CaMKII, {lab}", zorder=3)
lim = (3e-4, 1e-1)
ax.plot(lim, lim, color="0.5", lw=1)
ax.text(2e-2, 6e-4, "Gaussian fits better", fontsize=7, color="0.3", ha="center")
ax.text(1.5e-3, 1.0e-2, "exponential\nfits better", fontsize=7, color="0.3",
        ha="center")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("1 − R², exponential fit")
ax.set_ylabel("1 − R², Gaussian fit")
ax.set_title("c  shape", loc="left", fontweight="bold", fontsize=10)
ax.legend(frameon=False, fontsize=7, loc="upper left")

for ax in axs:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, lw=0.5)
fig.savefig("cam_kernel_profiles.png", bbox_inches="tight")
print("saved")
