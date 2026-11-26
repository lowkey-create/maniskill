"""Plot every path in each shard, action family, and dataset split."""
import argparse
import json
from collections import defaultdict
from itertools import groupby
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection


def draw(ax, paths, views=None):
    points = np.concatenate([p for _, target, measured in paths
                             for p in ((target,) if measured is None else (target, measured))])
    center = (points.max(axis=0) + points.min(axis=0)) / 2
    radius = 0.55 * max(1, np.ptp(points, axis=0).max())
    colors = dict(zip(dict.fromkeys(plane for plane, _, _ in paths), plt.get_cmap("tab20").colors))
    alpha = min(0.85, max(0.12, 12 / len(paths)))
    for target_only in (True, False):
        selected = [(colors[plane], target if target_only else measured)
                    for plane, target, measured in paths if target_only or measured is not None]
        if not selected:
            continue
        palette, curves = zip(*selected)
        dashed = target_only and paths[0][2] is not None
        style = dict(colors=palette, linewidths=0.9, alpha=alpha, linestyles="--" if dashed else "-")
        if views:
            i, j = views
            ax.add_collection(LineCollection([p[:, views] for p in curves], **style))
        else:
            ax.add_collection3d(Line3DCollection(curves, **style), autolim=False)
    labels = ("X (cm)", "Y (cm)", "Z (cm)")
    if views:
        i, j = views
        ax.set(xlim=(center[i] - radius, center[i] + radius), ylim=(center[j] - radius, center[j] + radius),
               xlabel=labels[i], ylabel=labels[j])
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)
    else:
        ax.set(xlim=(center[0] - radius, center[0] + radius),
               ylim=(center[1] - radius, center[1] + radius),
               zlim=(center[2] - radius, center[2] + radius),
               xlabel=labels[0], ylabel=labels[1], zlabel=labels[2])
        ax.set_box_aspect((1, 1, 1))
        ax.set_proj_type("ortho")
    return colors


def figure(paths, destination, title):
    fig = plt.figure(figsize=(11.5, 9), layout="constrained")
    fig.suptitle(f"{title}   |   {len(paths):,} paths")
    legend = draw(fig.add_subplot(2, 2, 1, projection="3d"), paths)
    for position, view in enumerate(((0, 1), (0, 2), (1, 2)), start=2):
        draw(fig.add_subplot(2, 2, position), paths, view)
    handles = [Line2D([], [], color=color, label=name) for name, color in legend.items()]
    if paths[0][2] is not None:
        handles += [Line2D([], [], color="0.25", label="Measured"),
                    Line2D([], [], color="0.25", linestyle="--", label="Target")]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False, fontsize=8)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=130)
    plt.close(fig)


def grid(groups, destination, title):
    rows = (len(groups) + 3) // 4
    fig = plt.figure(figsize=(15, 4.3 * rows + 0.5))
    fig.subplots_adjust(left=0.035, right=0.98, top=1 - 0.18 / rows,
                        bottom=0.10 / rows, wspace=0.12, hspace=0.26)
    fig.suptitle(f"{title}   |   XYZ (cm)")
    for index, (label, paths) in enumerate(groups.items(), start=1):
        ax = fig.add_subplot(rows, 4, index, projection="3d")
        draw(ax, paths)
        ax.set(title=f"{label} ({len(paths):,})", xlabel="", ylabel="", zlabel="")
        ax.tick_params(axis="both", labelsize=7, pad=0)
    fig.savefig(destination, dpi=130)
    plt.close(fig)


def plot_split(groups, directory, split, planned):
    label = "Planned geometry relative to start" if planned else "Recorded TCP in Panda base coordinates"
    overview = {}
    for kind, paths in groups.items():
        title = kind.replace("zigzag", "polyline").replace("_", " → ")
        location = f"{split} · {title} · {label}"
        overview[title] = paths
        figure(paths, directory / kind / "trajectories.png", location)
        planes = defaultdict(list)
        for path in paths:
            planes[path[0]].append(path)
        grid(planes, directory / kind / "by_plane.png", location)
    grid(overview, directory / "trajectories.png", f"{split.title()}   |   {label}")


def plot_plan(plan, root):
    for split, entries in plan.items():
        groups = defaultdict(list)
        for _, motion in entries:
            groups[motion.kind].append((motion.plane, 100 * motion.path, None))
        plot_split(groups, root / split, split, True)


def plot_recorded(root, manifest):
    splits = defaultdict(lambda: defaultdict(list))
    for shard, episodes in groupby(manifest, key=lambda entry: entry["shard"]):
        paths = []
        with h5py.File(root / shard) as data:
            for entry in episodes:
                slot, count = entry["slot"], entry["motion_samples"]
                target = np.vstack((entry["start_xyz_m"], data["target_xyz_base_m"][:count, slot]))
                measured = data["state/tcp_pose_base"][:count + 1, slot, :3]
                path = (entry["plane"], 100 * target, 100 * measured)
                paths.append(path)
                splits[entry["split"]][entry["kind"]].append(path)
        directory = (root / shard).parent
        figure(paths, directory / "trajectories.png", directory.relative_to(root).as_posix())
    for split, groups in splits.items():
        plot_split(groups, root / split, split, False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.corpus / "manifest.json").read_text(encoding="utf-8"))
    plot_recorded(args.corpus, manifest)
