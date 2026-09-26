# calibrate.py
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares, lsq_linear
from equations import latency, energy, flops, memory, bytes_moved

OUT_DIR = Path("results")
THETA_PATH = OUT_DIR / "theta.json"
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def fit_latency(df):
    S = df["S"].to_numpy(np.float64)
    B = df["B"].to_numpy(np.float64)
    measured = df["latency"].to_numpy(np.float64)

    params = {
        "t_launch": (5e-6, 1e-8, 1e-2),
        "BW": (2e11, 1e8, 1e13),
        "P": (5e12, 1e9, 1e15),
    }
    keys = tuple(params)

    def unpack(q):
        return dict(zip(keys, np.exp(q)))

    def residuals(q):
        return np.log(latency(S, B, unpack(q))) - np.log(measured)

    q0 = np.log([params[k][0] for k in keys])
    lo = np.log([params[k][1] for k in keys])
    hi = np.log([params[k][2] for k in keys])

    result = least_squares(residuals, x0=q0, bounds=(lo, hi))
    return unpack(result.x), result


def fit_energy(df, latency_theta):
    S = df["S"].to_numpy(np.float64)
    B = df["B"].to_numpy(np.float64)
    measured = df["energy"].to_numpy(np.float64)

    X = np.column_stack(
        [
            latency(S, B, latency_theta),
            bytes_moved(S, B) / 1e9,
            flops(S, B) / 1e9,
        ]
    )
    result = lsq_linear(X, measured, bounds=(0, np.inf))

    theta = {
        "p_0": float(result.x[0]),
        "energy_per_byte": float(result.x[1] / 1e9),
        "energy_per_flop": float(result.x[2] / 1e9),
        "latency_theta": {k: float(v) for k, v in latency_theta.items()},
    }
    return theta, result

def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, np.float64)
    y_pred = np.asarray(y_pred, np.float64)
    err = y_true - y_pred
    rel = np.abs(err) / np.maximum(np.abs(y_true), 1e-12)
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mape_percent": float(np.mean(rel) * 100),
        "median_ape_percent": float(np.median(rel) * 100),
    }


def _evaluate(df, col, predict):
    if len(df) == 0:
        return None
    S = df["S"].to_numpy(np.float64)
    B = df["B"].to_numpy(np.float64)
    return regression_metrics(df[col].to_numpy(np.float64), predict(S, B))


def _both_splits(calib, val, col, predict):
    return {
        "calibration": _evaluate(calib, col, predict),
        "validation": _evaluate(val, col, predict),
    }

def plot_surface_and_curves(
    df_calib, df_val, column, predict, zlabel, filename, scale=1.0, title=""
):
    df_all = pd.concat([df_calib, df_val])

    S_grid, B_grid = np.meshgrid(
        np.linspace(df_all["S"].min(), df_all["S"].max(), 40),
        np.linspace(df_all["B"].min(), df_all["B"].max(), 40),
    )
    Z = predict(S_grid, B_grid) * scale

    fig = plt.figure(figsize=(14, 11))
    fig.suptitle(title, fontsize=18)

    for i, (sub, split_name, marker) in enumerate(
        [(df_calib, "Calibration", "o"), (df_val, "Validation", "^")], 1
    ):
        ax = fig.add_subplot(2, 2, i, projection="3d")
        S = sub["S"].to_numpy(np.float64)
        B = sub["B"].to_numpy(np.float64)
        measured = sub[column].to_numpy(np.float64) * scale
        predicted = predict(S, B) * scale

        ax.plot_surface(S_grid, B_grid, Z, alpha=0.22, color="blue")
        ax.scatter(
            S, B, measured, color="tab:blue", marker=marker, s=35, label="Measured"
        )
        ax.scatter(
            S, B, predicted, color="tab:orange", marker="x", s=45, label="Predicted"
        )

        ax.set_xlabel("Image size S, px")
        ax.set_ylabel("Batch size B")
        ax.set_zlabel(zlabel)
        ax.set_title(split_name)
        ax.legend()

    for i, (sub, split_name) in enumerate(
        [(df_calib, "Calibration"), (df_val, "Validation")], 3
    ):
        ax = fig.add_subplot(2, 2, i)

        S = sub["S"].to_numpy(np.float64)
        B = sub["B"].to_numpy(np.float64)
        workload = B * S**2
        measured = sub[column].to_numpy(np.float64) * scale
        predicted = predict(S, B) * scale

        order = np.argsort(workload)
        workload, measured, predicted = (
            workload[order],
            measured[order],
            predicted[order],
        )

        m = regression_metrics(measured, predicted)

        ax.plot(
            workload, measured, "o-", color="tab:blue", markersize=4, label="Measured"
        )
        ax.plot(
            workload, predicted, "x-",
            color="tab:orange", markersize=4,
            label="Predicted",
        )

        ax.set_xscale("log")
        ax.set_xlabel(r"Workload $BS^2$")
        ax.set_ylabel(zlabel)
        ax.set_title(
            f"{split_name}: measured vs predicted\n"
            f"MAPE={m['mape_percent']:.1f}%, "
            f"MedianAPE={m['median_ape_percent']:.1f}%"
        )
        ax.grid(alpha=0.3)
        ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(FIG_DIR / filename, dpi=200)
    plt.close()


def plot_parity_and_errors(
    df_calib, df_val, column, predict, label, filename, scale=1.0, log=False, title=""
):
    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    fig.suptitle(title, fontsize=18)

    splits = [
        (df_calib, "Calibration", "o", "tab:blue"),
        (df_val, "Validation", "^", "tab:orange"),
    ]

    for col_id, (sub, split_name, marker, point_color) in enumerate(splits):
        S = sub["S"].to_numpy(np.float64)
        B = sub["B"].to_numpy(np.float64)
        measured = sub[column].to_numpy(np.float64) * scale
        predicted = predict(S, B) * scale
        workload = B * S**2

        abs_err = np.abs(predicted - measured)
        ape = abs_err / np.maximum(np.abs(measured), 1e-12) * 100
        m = regression_metrics(measured, predicted)

        # parity
        ax = axes[0, col_id]
        x, y = measured.copy(), predicted.copy()
        if log:
            mask = (x > 0) & (y > 0)
            x, y = x[mask], y[mask]

        lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
        ax.scatter(x, y, color=point_color, marker=marker, alpha=0.75, label=split_name)
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.5, label="Ideal")

        if log:
            ax.set_xscale("log")
            ax.set_yscale("log")

        ax.set_xlabel(f"Measured {label}")
        ax.set_ylabel(f"Predicted {label}")
        ax.set_title(f"{split_name}: predicted vs measured")
        ax.grid(alpha=0.3)
        ax.legend()

        # APE
        ax = axes[1, col_id]
        order = np.argsort(workload)
        ax.plot(
            workload[order], ape[order], "o-",
            color="tab:purple", markersize=4,
            label="APE",
        )
        ax.axhline(
            m["mape_percent"], color="tab:red", ls="--",
            label=f"MAPE = {m['mape_percent']:.1f}%",
        )
        ax.axhline(
            m["median_ape_percent"], color="black", ls=":",
            label=f"Median APE = {m['median_ape_percent']:.1f}%",
        )
        ax.set_xscale("log")
        ax.set_xlabel(r"Workload $BS^2$")
        ax.set_ylabel("Absolute percentage error, %")
        ax.set_title(f"{split_name}: relative prediction error")
        ax.grid(alpha=0.3)
        ax.legend()

        # pointwise absolute error
        ax = axes[2, col_id]
        ax.plot(
            workload[order], abs_err[order],
            "o-", color="tab:green",
            markersize=4, label="Absolute error",
        )
        ax.axhline(m["mae"], color="tab:red", ls="--", label=f"MAE = {m['mae']:.3g}")
        ax.set_xscale("log")
        ax.set_xlabel(r"Workload $BS^2$")
        ax.set_ylabel(f"Absolute error, {label.split(',')[-1].strip()}")
        ax.set_title(f"{split_name}: absolute prediction error")
        ax.grid(alpha=0.3)
        ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(FIG_DIR / filename, dpi=200)
    plt.close()


def main():
    df = pd.read_csv(OUT_DIR / "measurements.csv")

    valid = df[
        (~df["oom"])
        & np.isfinite(df["latency"])
        & np.isfinite(df["energy"])
        & np.isfinite(df["memory"])
    ].copy()
    calib = valid[~valid["is_validation"]]
    val = valid[valid["is_validation"]]

    if len(calib) == 0:
        raise RuntimeError("No calibration points available.")

    latency_theta, _ = fit_latency(calib)
    energy_theta, _ = fit_energy(calib, latency_theta)

    mem_predict = lambda S, B: memory(S, B)
    lat_predict = lambda S, B: latency(S, B, latency_theta)
    en_predict = lambda S, B: energy(S, B, energy_theta)

    metrics = {
        "memory": _both_splits(calib, val, "memory", mem_predict),
        "latency": _both_splits(calib, val, "latency", lat_predict),
        "energy": _both_splits(calib, val, "energy", en_predict),
    }

    output = {
        "latency": {k: float(v) for k, v in latency_theta.items()},
        "energy": energy_theta,
        "metrics": metrics,
    }
    with open(THETA_PATH, "w") as f:
        json.dump(output, f, indent=4)

    plot_surface_and_curves(
        calib, val, "memory", mem_predict, "Peak memory, MiB",
        "memory_surface.png", scale=1 / 1024**2,
        title="Memory: measured values vs analytical prediction",
    )
    plot_parity_and_errors(
        calib, val, "memory", mem_predict, "memory, MiB",
        "memory_parity.png", scale=1 / 1024**2,
        title="Memory: prediction quality",
    )

    plot_surface_and_curves(
        calib, val, "latency", lat_predict, "Latency, ms", 
        "latency_surface.png", scale=1000,
        title="Latency: measured values vs calibrated analytical prediction",
    )
    plot_parity_and_errors(
        calib, val, "latency", lat_predict, "latency, ms", 
        "latency_parity.png", scale=1000, log=True,
        title="Latency: prediction quality",
    )

    plot_surface_and_curves(
        calib, val, "energy", en_predict, "Energy, J", 
        "energy_surface.png",
        title="Energy: measured values vs calibrated analytical prediction",
    )
    plot_parity_and_errors(
        calib, val, "energy", en_predict, "energy, J", 
        "energy_parity.png", log=True,
        title="Energy: prediction quality",
    )


if __name__ == "__main__":
    main()
