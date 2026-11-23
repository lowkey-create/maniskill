"""Plot commanded and measured Panda TCP paths from a SpatialMotion run."""
import argparse
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def plot_run(run, output):
    episodes = json.loads((run / "summary.json").read_text())
    traces = []
    for episode in episodes:
        name = episode["trajectory"]
        with h5py.File(run / name / "expert.h5") as h5:
            if not h5.attrs["complete"]:
                raise ValueError(f"Incomplete episode: {name}")
            count = episode["tracking_steps"]
            target = h5["target_xyz_base_m"][:count]
            actual = h5["state/tcp_pose_base"][1:count + 1, :, :3]
            error = h5["error_norm_m"][:] * 1000
            hz = h5.attrs["control_hz"]
            for slot in range(target.shape[1]):
                traces.append((f"{name} · env {slot}", target[:, slot] * 1000,
                               actual[:, slot] * 1000, error[:, slot], hz))

    figure = plt.figure(figsize=(11, 3.2 * len(traces)), layout="constrained")
    figure.suptitle("Panda TCP tracking in the robot base frame", fontweight="bold")
    for index, (label, target, actual, error, hz) in enumerate(traces):
        path = figure.add_subplot(len(traces), 2, 2 * index + 1, projection="3d")
        path.plot(*target.T, "--", color="#d77722", lw=2.2, label="Target")
        path.plot(*actual.T, color="#135aa0", lw=1.1, label="Measured TCP")
        path.scatter(*actual[0], color="#16805a", s=25, label="Start")
        path.scatter(*actual[-1], color="#7627a5", s=25, marker="x", label="End")
        path.set(xlabel="X (mm)", ylabel="Y (mm)", zlabel="Z (mm)", title=label)
        points = np.concatenate((target, actual), axis=0)
        center = (points.min(axis=0) + points.max(axis=0)) / 2
        radius = max(np.ptp(points, axis=0).max() * 0.55, 1)
        path.set(xlim=(center[0] - radius, center[0] + radius),
                 ylim=(center[1] - radius, center[1] + radius),
                 zlim=(center[2] - radius, center[2] + radius))
        path.set_box_aspect((1, 1, 1))
        if index == 0:
            path.legend(loc="upper left", fontsize=8)

        residual = figure.add_subplot(len(traces), 2, 2 * index + 2)
        time_s = (np.arange(len(error)) + 1) / hz
        residual.plot(time_s, error, color="#135aa0", lw=1.3)
        if len(error) > len(target):
            residual.axvspan(len(target) / hz, time_s[-1], color="#e3e8ee", label="Endpoint hold")
            if index == 0:
                residual.legend(loc="upper right", fontsize=8)
        residual.set(xlabel="Time (s)", ylabel="Error (mm)",
                     title=f"Tracking max {error[:len(target)].max():.2f} mm")
        residual.set_ylim(bottom=0)
        residual.grid(alpha=0.25)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    print(f"Wrote {output} from {len(traces)} recorded trajectories")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    plot_run(args.run, args.output)
