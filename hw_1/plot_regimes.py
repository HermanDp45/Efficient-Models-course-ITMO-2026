import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from pathlib import Path

from equations import layer_costs

OUT_DIR = Path("/results")
FIG_DIR = OUT_DIR / "figures"

with open(OUT_DIR / "theta.json") as f:
    theta = json.load(f)["latency"]

df = pd.read_csv(OUT_DIR / "measurements.csv")
df = df[~df["oom"]].copy()
df["workload"] = df["B"] * df["S"] ** 2
df = df.sort_values("workload")

names = [x[0] for x in layer_costs(32, 1)]
matrix, launch, memory, compute = [], [], [], []

for _, row in df.iterrows():
    regs = []
    for _, flops, nbytes in layer_costs(row["S"], row["B"]):
        times = [theta["t_launch"], nbytes / theta["BW"], flops / theta["P"]]
        regs.append(int(np.argmax(times)))
    matrix.append(regs)
    launch.append(regs.count(0))
    memory.append(regs.count(1))
    compute.append(regs.count(2))

matrix = np.array(matrix).T
w = df["workload"].to_numpy()

fig, ax = plt.subplots(2, 1, figsize=(13, 9), gridspec_kw={"height_ratios": [1, 2]})

ax[0].plot(w, launch, label="Launch-bound")
ax[0].plot(w, memory, label="Memory-bound")
ax[0].plot(w, compute, label="Compute-bound")
ax[0].set_xscale("log")
ax[0].set_ylabel("Number of operations")
ax[0].set_title("Latency regimes")
ax[0].grid(alpha=0.3)
ax[0].legend()

logw = np.log10(w)
edges = np.empty(len(w) + 1)
edges[1:-1] = (logw[:-1] + logw[1:]) / 2
edges[0] = logw[0] - (edges[1] - logw[0])
edges[-1] = logw[-1] + (logw[-1] - edges[-2])

cmap = ListedColormap(["tab:blue", "tab:orange", "tab:green"])
ax[1].pcolormesh(
    10**edges, np.arange(len(names) + 1), matrix, cmap=cmap, vmin=-0.5, vmax=2.5
)
ax[1].set_xscale("log")
ax[1].set_xlabel(r"Workload $BS^2$")
ax[1].set_ylabel("Operation")
ax[1].set_yticks(np.arange(len(names)) + 0.5)
ax[1].set_yticklabels(names)
ax[1].legend(
    handles=[
        Patch(color="tab:blue", label="Launch-bound"),
        Patch(color="tab:orange", label="Memory-bound"),
        Patch(color="tab:green", label="Compute-bound"),
    ],
    loc="upper left",
    bbox_to_anchor=(1.01, 1),
)

plt.tight_layout()
plt.savefig(FIG_DIR / "latency_regimes.png", dpi=200, bbox_inches="tight")
plt.close()
