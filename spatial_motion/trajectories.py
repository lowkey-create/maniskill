"""Cartesian references in metres in the Panda base frame."""
import numpy as np

NAMES = ("line", "diagonal", "circle", "helix")

def reference(name, start, steps):
    u = np.linspace(0, 1, steps, dtype=np.float32)
    zeros = np.zeros_like(u)
    if name == "line":
        offset = np.stack([zeros, 0.12 * u, zeros], axis=-1)
    elif name == "diagonal":
        offset = np.stack([-0.10 * u, 0.10 * u, 0.11 * u], axis=-1)
    elif name in ("circle", "helix"):
        angle = (2 if name == "circle" else 4) * np.pi * u
        height = zeros if name == "circle" else 0.14 * u
        offset = np.stack([0.055 * (np.cos(angle) - 1), 0.055 * np.sin(angle), height], axis=-1)
    else:
        raise ValueError(f"Unknown trajectory: {name}")
    return start[None] + offset[:, None]

def load_waypoints(path, num_envs):
    points = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
    if points.ndim == 2:
        points = points[:, None, :]
    if (points.ndim != 3 or points.shape[0] < 2 or points.shape[1] not in (1, num_envs)
            or points.shape[2] != 3 or not np.isfinite(points).all()):
        raise ValueError("Expected finite waypoints shaped (T, 3) or (T, num_envs, 3), T >= 2")
    return np.broadcast_to(points, (len(points), num_envs, 3)).copy()
