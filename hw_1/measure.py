# measure.py
import gc
import math
import time

import pandas as pd
import numpy as np
from pathlib import Path
import torch
import pynvml

from models import SimpleModel

BASE_IMAGE_SIZES = [32, 64, 128, 224, 256, 384, 512, 1024]
BASE_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256]

EXTRA_IMAGE_SIZES = [48, 192, 304, 352]
EXTRA_BATCH_SIZES = [7, 33, 67]

IMAGE_SIZES = sorted(BASE_IMAGE_SIZES + EXTRA_IMAGE_SIZES)
BATCH_SIZES = sorted(BASE_BATCH_SIZES + EXTRA_BATCH_SIZES)

N_WARMAP = 10
N_LATENCY_RUNS = 50
N_ENERGY_RUNS = 100

OUT_DIR = Path("results")
OUT_DIR.mkdir(exist_ok=True)
OUT_PATH = OUT_DIR / "measurements.csv"

torch.backends.cudnn.benchmark = (
    False  # main results: PyTorch's default heuristics choose the kernel
)
torch.backends.cudnn.allow_tf32 = (
    False  # so that FP32 means FP32 on Ampere and newer GPUs
)
torch.backends.cuda.matmul.allow_tf32 = False

device = torch.device("cuda")


def warmup(model, x):
    with torch.inference_mode():
        for _ in range(N_WARMAP):
            y = model(x)

    torch.cuda.synchronize()

    del y


def measure_latency(model, x):
    times = []

    with torch.inference_mode():
        for _ in range(N_LATENCY_RUNS):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            y = model(x)
            end.record()

            end.synchronize()

            # time in seconds
            elapsed_time = start.elapsed_time(end) / 1000
            times.append(elapsed_time)

            del y

    # median
    latency = np.median(times)
    return float(latency)


def measure_memory(model, x):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    with torch.inference_mode():
        y = model(x)

    torch.cuda.synchronize()
    memory = int(torch.cuda.max_memory_allocated())

    del y
    return memory


def init_nvml():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    return handle


def read_total_energy_joules(handle):
    energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle) / 1000
    return energy


def measure_energy(model, x, handle, n_runs):
    torch.cuda.synchronize()
    energy_before = read_total_energy_joules(handle)

    with torch.inference_mode():
        for _ in range(n_runs):
            y = model(x)

    torch.cuda.synchronize()

    energy_after = read_total_energy_joules(handle)

    del y
    return float((energy_after - energy_before) / n_runs)


def is_validation_point(image_size, batch_size):
    return (image_size in EXTRA_IMAGE_SIZES) or (batch_size in EXTRA_BATCH_SIZES)


def measure_configuration(model, image_size, batch_size, nvml_handle):
    S = image_size
    B = batch_size
    x = None

    try:
        x = torch.randn(B, 3, S, S, device="cuda", dtype=torch.float32)

        warmup(model, x)

        latency = measure_latency(model, x)
        memory = measure_memory(model, x)
        n_energy_runs = max(10, int(np.ceil(0.5 / latency)))
        energy = measure_energy(model, x, nvml_handle, n_energy_runs)

        row = {
            "B": B,
            "S": S,
            "latency": latency,
            "memory": memory,
            "energy": energy,
            "is_validation": is_validation_point(S, B),
            "oom": False,
        }
        return row

    except torch.cuda.OutOfMemoryError:
        print("OOM")
        if x is not None:
            del x
            x = None

        gc.collect()
        torch.cuda.empty_cache()

        row = {
            "B": B,
            "S": S,
            "latency": np.nan,
            "memory": np.nan,
            "energy": np.nan,
            "is_validation": is_validation_point(S, B),
            "oom": True,
        }

        return row

    finally:
        if x is not None:
            del x
        gc.collect()


def main():

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this homework.")

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    model = SimpleModel().to(device)
    model = model.cuda().eval()

    nvml_handle = init_nvml()

    # ==== INFO
    print("GPU" + str(torch.cuda.get_device_name(0)))
    total_memory = torch.cuda.get_device_properties(0).total_memory

    print(
        "GPU memory:",
        round(total_memory / 1024**3, 2),
        "GiB",
    )

    print()

    print("PyTorch:", torch.__version__)
    print("CUDA:", torch.version.cuda)
    print("cuDNN:", torch.backends.cudnn.version())

    print()

    print("cudnn.benchmark =", torch.backends.cudnn.benchmark)
    print(
        "cudnn.allow_tf32 =",
        torch.backends.cudnn.allow_tf32,
    )
    print(
        "matmul.allow_tf32 =",
        torch.backends.cuda.matmul.allow_tf32,
    )

    print()

    rows = []

    total = len(IMAGE_SIZES) * len(BATCH_SIZES)
    current = 0
    for S in IMAGE_SIZES:
        for B in BATCH_SIZES:
            current += 1

            print(
                f"[{current:3d}/{total}] " f"S={S:3d}, B={B:3d}",
                end=" | ",
                flush=True,
            )

            row = measure_configuration(
                model=model,
                image_size=S,
                batch_size=B,
                nvml_handle=nvml_handle,
            )

            rows.append(row)

            if row["oom"]:
                print()

            else:
                print(
                    f"latency = "
                    f"{row['latency'] * 1000:.3f} ms | "
                    f"memory = "
                    f"{row['memory'] / 1024**2:.2f} MiB | "
                    f"energy = "
                    f"{row['energy']:.6f} J"
                )

    df = pd.DataFrame(rows)

    df.to_csv(
        OUT_PATH,
        index=False,
    )

    pynvml.nvmlShutdown()

    print()
    print("==============================")
    print("Measurement finished")
    print("==============================")

    print("Total configurations:", len(df))
    print("OOM:", int(df["oom"].sum()))
    print(
        "Validation:",
        int(df["is_validation"].sum()),
    )

    print()
    print("Saved to:")
    print(OUT_PATH)


if __name__ == "__main__":
    main()
