"""Plot representative recorded trajectories from a generated corpus."""
import argparse
import json
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D


def plot_corpus(root, output):
    entries = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    groups = defaultdict(list)
    for entry in entries:
        groups[entry["kind"]].append(entry)
    if not groups:
        raise ValueError("The corpus manifest contains no trajectories")
    columns = min(4, len(groups))
    rows = (len(groups) + columns - 1) // columns
    figure = plt.figure(figsize=(4.5 * columns, 4.3 * rows + 0.8), layout="constrained")
    figure.suptitle("Panda TCP paths in the base frame  |  equal XYZ scale")
    colors = ("#0e7c72", "#264da5", "#c05d17")
    labels = ("Small training size", "Large training size", "Held-out test size")
    with ExitStack() as stack:
        shards = {}
        for index, (kind, group) in enumerate(groups.items(), start=1):
            train = sorted((e for e in group if e["split"] == "train"), key=lambda e: e["size_mm"])
            test = sorted((e for e in group if e["split"] == "test"), key=lambda e: e["size_mm"])
            samples = (train[0] if train else None, train[-1] if len(train) > 1 else None,
                       test[len(test) // 2] if test else None)
            ax = figure.add_subplot(rows, columns, index, projection="3d")
            ax.set_proj_type("ortho")
            all_points = []
            for entry, color in zip(samples, colors):
                if entry is None:
                    continue
                shard = entry["shard"]
                if shard not in shards:
                    shards[shard] = stack.enter_context(h5py.File(root / shard, "r"))
                h5 = shards[shard]
                count = h5["action"].shape[0] - int(h5.attrs["hold_steps"])
                slot = entry["slot"]
                target = 1000 * h5["target_xyz_base_m"][:count, slot]
                measured = 1000 * h5["state/tcp_pose_base"][1:count + 1, slot, :3]
                all_points.extend((target, measured))
                ax.plot(*target.T, "--", color=color, alpha=0.55, linewidth=2.1)
                ax.plot(*measured.T, color=color, linewidth=1.3)
                ax.scatter(*measured[0], s=12, color=color)
            points = np.concatenate(all_points)
            center = 0.5 * (points.max(axis=0) + points.min(axis=0))
            radius = max(1, 0.58 * np.ptp(points, axis=0).max())
            ax.set(xlim=(center[0] - radius, center[0] + radius),
                   ylim=(center[1] - radius, center[1] + radius),
                   zlim=(center[2] - radius, center[2] + radius),
                   xlabel="X (mm)", ylabel="Y (mm)", zlabel="Z (mm)",
                   title=kind.replace("_", " + "))
            ax.set_box_aspect((1, 1, 1))
    legend = [Line2D([], [], color=color, label=label) for color, label in zip(colors, labels)]
    legend.extend([Line2D([], [], color="#444444", linestyle="--", label="Target"),
                   Line2D([], [], color="#444444", label="Measured")])
    figure.legend(handles=legend, loc="outside lower center", ncol=5, frameon=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot_corpus(args.corpus, args.output)
