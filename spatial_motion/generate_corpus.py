"""Generate language-conditioned, GPU-batched Panda trajectories."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import gymnasium as gym
import numpy as np

from .collect import as_numpy, run_episode
from .verify import verify

KINDS = ("line", "diagonal", "circle", "ellipse", "helix", "zigzag",
         "line_circle", "circle_helix")


def segment(kind, samples, size, variant):
    u = np.linspace(0, 1, samples)
    side = (-1) ** variant
    radius = size / 1000
    if kind in ("line", "diagonal"):
        delta = np.array([-size, side * (20 + size // 2),
                          0 if kind == "line" else (-1) ** (variant // 2) * (20 + size // 3)])
        return u[:, None] * delta / 1000, (
            f"Move in a straight line by ({delta[0]}, {delta[1]}, {delta[2]}) mm "
            "along the Panda base X, Y, and Z axes."
        )
    if kind == "zigzag":
        width = size // 2
        vertices = np.array([[0, 0, 0], [-size / 3, side * width, 0],
                             [-2 * size / 3, -side * width, 0], [-size, 0, 0]])
        points = np.column_stack([np.interp(u, np.linspace(0, 1, 4), col)
                                  for col in vertices.T]) / 1000
        return points, (f"Trace a horizontal three-segment zigzag spanning {size} mm along "
                        f"negative X and {2 * width} mm across Y, starting toward "
                        f"{'positive' if side == 1 else 'negative'} Y.")
    direction = "counterclockwise" if side == 1 else "clockwise"
    angle = side * (4 if kind == "helix" else 2) * np.pi * u
    if kind == "helix":
        radial_mm = size // 2
        rise_mm = size + 20
        radial = radial_mm / 1000
        return np.column_stack((radial * (np.cos(angle) - 1),
                                radial * np.sin(angle), rise_mm / 1000 * u)), (
            f"Trace a two-turn {direction} helix viewed from positive Z, with "
            f"radius {radial_mm} mm and total rise {rise_mm} mm."
        )
    minor = size if kind == "circle" else (3 * size) // 5
    points = np.column_stack((radius * (np.cos(angle) - 1),
                              minor / 1000 * np.sin(angle), np.zeros_like(u)))
    shape = (f"circle with radius {size} mm" if kind == "circle"
             else f"ellipse with X and Y semiaxes {size} and {minor} mm")
    return points, (f"Trace a horizontal {shape}, {direction} viewed from positive Z, "
                    "starting at its positive-X edge.")


def trajectory(kind, steps, size, variant):
    if kind in ("line_circle", "circle_helix"):
        first, second = ("line", "circle") if kind == "line_circle" else ("circle", "helix")
        split = steps // 2
        # A shared endpoint preserves continuity and gives each part its own reference.
        a, instruction_a = segment(first, split + 1, size, variant)
        b, instruction_b = segment(second, steps - split, size, variant)
        return np.concatenate((a, a[-1] + b[1:])), (
            f"{instruction_a} Then, from the endpoint, {instruction_b[0].lower()}{instruction_b[1:]}"
        ), [first, second]
    points, instruction = segment(kind, steps, size, variant)
    return points, instruction, [kind]


def examples(split, count, steps, rng):
    entries = []
    for kind in KINDS:
        for variant in range(count):
            size = 24 + 2 * variant if split == "train" else 25 + 8 * variant
            points, language, segments = trajectory(kind, steps, size, variant)
            metadata = {
                "id": f"{split}-{kind}-{variant:03d}", "split": split,
                "kind": kind, "segments": segments, "size_mm": size,
                "variant": variant, "language": language, "frame": "panda_base",
            }
            entries.append((points.astype(np.float32), metadata))
    return [entries[i] for i in rng.permutation(len(entries))]


def write_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def record(args):
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    config = {k: v for k, v in vars(args).items() if k != "output"}
    config_path = output / "config.json"
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise ValueError("Resume requires the same configuration")
    else:
        if any(output.iterdir()):
            raise ValueError("Use an empty output directory for a new run")
        write_json(config_path, config)

    rng = np.random.default_rng(args.seed)
    manifest = []
    env = None
    batch_size = None
    try:
        for split, count in (("train", args.train_per_kind), ("test", args.test_per_kind)):
            entries = examples(split, count, args.steps, rng)
            for index in range(0, len(entries), args.num_envs):
                batch = entries[index:index + args.num_envs]
                label = f"{split}_{index // args.num_envs:04d}"
                destination = output / label
                seed = args.seed + index + (10000 if split == "test" else 0)
                metadata = [item[1] for item in batch]
                if not destination.exists():
                    if len(batch) != batch_size:
                        if env is not None:
                            env.close()
                        env = None
                        batch_size = len(batch)
                        env = gym.make("SpatialMotion-v1", num_envs=batch_size,
                                       sim_backend="gpu", render_backend="gpu", obs_mode="rgb")
                    env.reset(seed=seed)
                    start = as_numpy(env.unwrapped.tcp_in_base.p)
                    targets = start[None] + np.stack([item[0] for item in batch], axis=1)
                    # Publish only complete, verified shards.
                    with TemporaryDirectory(prefix=f".{label}-", dir=output) as temporary:
                        staging = Path(temporary) / label
                        metrics = run_episode(env, targets, staging, "gpu", args.hold,
                                              args.tolerance_mm / 1000, seed, label,
                                              video_envs=args.video_envs, episodes=metadata)
                        if not metrics["passed"]:
                            raise RuntimeError(f"{label} exceeded tracking tolerance")
                        verify(staging)
                        staging.rename(destination)
                metrics = json.loads((destination / "metrics.json").read_text())
                if not metrics["passed"]:
                    raise RuntimeError(f"{label} failed its tracking check")
                for slot, example in enumerate(metadata):
                    manifest.append({**example, "shard": f"{label}/expert.h5", "slot": slot,
                                     "max_error_mm": metrics["max_mm"][slot]})
                write_json(output / "manifest.json", manifest)
                print(f"Verified {label}: {len(batch)} episodes, "
                      f"max {max(metrics['max_mm']):.2f} mm", flush=True)
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-per-kind", type=int, default=32)
    parser.add_argument("--test-per-kind", type=int, default=8)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--video-envs", type=int, default=2)
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--hold", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--tolerance-mm", type=float, default=10)
    args = parser.parse_args()
    if (not 1 <= args.train_per_kind <= 32 or not 0 <= args.test_per_kind <= 8
            or args.num_envs < 1 or not 0 <= args.video_envs <= args.num_envs
            or args.steps < 120 or args.hold < 1 or args.steps + args.hold >= 20000
            or not np.isfinite(args.tolerance_mm) or args.tolerance_mm <= 0):
        parser.error("Require 1–32 train and 0–8 test variants, valid batch/video counts, "
                     "at least 120 reference steps, and positive hold/tolerance")
    record(args)
