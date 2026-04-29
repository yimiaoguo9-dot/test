import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# ── 1. Load ────────────────────────────────────────────────────────────────
# Nion Swift CSV 1D has a few header lines; adjust skiprows if needed
file_path = r"C:\Users\yimiaog2\Downloads\17SpimEELS_2meV-ch_300ms_1mmEA_off-axis_top-bot-WSe2_aligned-Pick-Sum_2026-04-27T010206_366160_1028_0.csv"

data = np.genfromtxt(file_path, delimiter=",", skip_header=5)
energy = data[:, 0]   # eV axis
intensity = data[:, 1]

# ── 2. Antisymmetrize: I(ω) - I(-ω) ───────────────────────────────────────
# Interpolate onto a symmetric, uniformly spaced grid first
e_max = min(abs(energy[0]), abs(energy[-1]))  # largest symmetric range
e_sym = np.linspace(-e_max, e_max, len(energy))

interp = interp1d(energy, intensity, kind="cubic", bounds_error=False, fill_value=0.0)
I_sym    = interp(e_sym)
I_flipped = interp(-e_sym)   # I(-ω)

antisym = I_sym - I_flipped   # I(ω) - I(-ω)

# ── 3. Plot ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

axes[0].plot(energy, intensity, color="steelblue", lw=1.2, label="Raw")
axes[0].set_ylabel("Intensity (arb. units)")
axes[0].set_title("Raw EELS spectrum")
axes[0].legend()
axes[0].axvline(0, color="gray", lw=0.7, ls="--")

axes[1].plot(e_sym, antisym, color="firebrick", lw=1.2, label=r"$I(\omega) - I(-\omega)$")
axes[1].set_ylabel("Antisymmetrized intensity")
axes[1].set_xlabel("Energy loss (eV)")
axes[1].set_title("Antisymmetrized spectrum")
axes[1].legend()
axes[1].axvline(0, color="gray", lw=0.7, ls="--")
axes[1].axhline(0, color="gray", lw=0.7, ls="--")

plt.tight_layout()
# plt.savefig("antisymmetrized.png", dpi=150)
plt.show()
# print("Saved → antisymmetrized.png")