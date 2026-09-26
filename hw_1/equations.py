# equations.py
import numpy as np

N_PARAMS = 1040324
FP32_BYTES = 4


def layer_costs(image_size, batch):
    S = np.asarray(image_size, dtype=np.float64)
    B = np.asarray(batch, dtype=np.float64)

    return [
        # name, FLOPs, bytes
        ("conv1", 2352 * B * S**2, 4 * (11 * B * S**2 + 4704)),
        ("relu1", 0, 4 * 16 * B * S**2),
        ("pool", 0, 4 * 10 * B * S**2),
        ("conv2", 6400 * B * S**2, 4 * (6 * B * S**2 + 51200)),
        ("relu2", 0, 4 * 8 * B * S**2),
        ("conv3", 2304 * B * S**2, 4 * (6 * B * S**2 + 73728)),
        ("relu3", 0, 4 * 4 * B * S**2),
        ("conv4", 1024 * B * S**2, 4 * (6 * B * S**2 + 32768)),
        ("relu4", 0, 4 * 8 * B * S**2),
        ("conv5", 4608 * B * S**2, 4 * (5 * B * S**2 + 589824)),
        ("relu5", 0, 4 * 2 * B * S**2),
        ("conv6", 1024 * B * S**2, 4 * (3 * B * S**2 + 131072)),
        ("relu6", 0, 4 * 4 * B * S**2),
        ("gap", 0, 4 * (2 * B * S**2 + 512 * B)),
        ("linear1", 2 * 512 * 256 * B, 4 * (768 * B + 131328)),
        ("relu_head", 0, 4 * 512 * B),
        ("linear2", 2 * 256 * 100 * B, 4 * (356 * B + 25700)),
    ]


def flops(image_size, batch):
    S = np.asarray(image_size, dtype=np.float64)
    B = np.asarray(batch, dtype=np.float64)

    return 17712 * B * S**2 + 313344 * B


def memory(image_size, batch):
    S = np.asarray(image_size, dtype=np.float64)
    B = np.asarray(batch, dtype=np.float64)

    # 1040324 = count params
    # 4 = fp32
    return 4 * (1040324 + 11 * B * S**2)


def bytes_moved(image_size, batch):
    S = np.asarray(image_size, dtype=np.float64)
    B = np.asarray(batch, dtype=np.float64)

    # 1040324 = count params
    # 4 = fp32
    return 4 * (91 * B * S**2 + 2148 * B + 1040324)


def latency(image_size, batch, theta):
    t_launch = theta["t_launch"]
    BW = theta["BW"]
    P = theta["P"]

    result = 0.0

    for _, f, nbytes in layer_costs(image_size, batch):
        result = result + np.maximum(
            t_launch,
            np.maximum(
                nbytes / BW,
                f / P,
            ),
        )

    return result


def energy(image_size, batch, theta_energy):
    T = latency(
        image_size,
        batch,
        theta_energy["latency_theta"],
    )

    return (
        theta_energy["p_0"] * T
        + theta_energy["energy_per_byte"] * bytes_moved(image_size, batch)
        + theta_energy["energy_per_flop"] * flops(image_size, batch)
    )
