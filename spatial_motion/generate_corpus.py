"""Build, pilot, or record the SpatialMotion Cartesian-product corpus."""
import argparse
import json
from pathlib import Path

import numpy as np

from .plan import SPLITS, START_INDICES, STARTS, motions, pilot_plan
from .plot_corpus import plot_plan, plot_recorded

HOLD = 20
START_TOLERANCE_M = 0.002
TRACK_TOLERANCE_M = 0.010


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def requests(plan, pilot):
    geometries = {}
    for split, entries in plan.items():
        for case, motion in entries:
            key = (motion.kind, motion.plane, tuple(motion.parameters.items()))
            group = geometries.setdefault(key, (motion, []))
            ranks = (0,) if pilot and split == "train" else START_INDICES[split]
            for rank in ranks:
                group[1].append((split, f"{case}-{rank + 1}", rank))
    return sorted(geometries.values(), key=lambda group: len(group[0].path))


def targets_for(paths, starts):
    longest = max(map(len, paths))
    return np.stack([np.pad(path, ((0, longest - len(path)), (0, 0)), mode="edge")
                     for path in paths], axis=1) + starts


def check_candidates(env, paths, starts, seed):
    import torch

    from .collect import prepare

    _, tail = prepare(env, seed, starts)
    raw = env.unwrapped
    lengths = torch.as_tensor([len(path) for path in paths], device=raw.device)
    goals = torch.as_tensor(targets_for(paths, starts), device=raw.device)
    origin = torch.as_tensor(starts, device=raw.device)
    passed = torch.linalg.vector_norm(raw.tcp_in_base.p - origin, dim=1) <= START_TOLERANCE_M
    for step in range(len(goals) + HOLD):
        goal = goals[min(step, len(goals) - 1)]
        _, _, terminated, truncated, _ = env.step(torch.cat((goal, tail), dim=1))
        error = torch.linalg.vector_norm(raw.tcp_in_base.p - goal, dim=1)
        check = (step < lengths) | (step == lengths + HOLD - 1)
        passed &= (~check | (error <= TRACK_TOLERANCE_M)) & ~terminated & ~truncated
    return passed.cpu().numpy()


def select_starts(env, group, seed):
    batch = env.unwrapped.num_envs
    paths = [motion.path[1:] for motion, _ in group]
    paths.extend([paths[-1]] * (batch - len(paths)))
    required = np.asarray([1 + max(rank for _, _, rank in slots) for _, slots in group])
    chosen = [[] for _ in group]
    available = np.ones((len(group), len(STARTS)), dtype=bool)
    distances = np.linalg.norm(STARTS[:, None] - STARTS[None], axis=-1)
    center_score = -np.linalg.norm(STARTS - (STARTS.max(axis=0) + STARTS.min(axis=0)) / 2, axis=1)
    candidate = np.zeros(batch, dtype=int)
    active = np.arange(len(group))
    while len(active):
        for slot in active:
            score = distances[chosen[slot]].min(axis=0) if chosen[slot] else center_score
            candidate[slot] = np.argmax(np.where(available[slot], np.round(score, 7), -np.inf))
        passed = check_candidates(env, paths, STARTS[candidate], seed)
        for slot in active:
            selection = candidate[slot]
            available[slot, selection] = False
            if passed[slot]:
                chosen[slot].append(selection)
        active = [slot for slot in active if len(chosen[slot]) < required[slot] and available[slot].any()]
    return chosen


def plan_starts(geometries, batch, args):
    import gymnasium as gym

    selected = {split: [] for split in SPLITS}
    missing = []
    env = gym.make("SpatialMotion-v1", num_envs=batch, sim_backend=args.sim,
                   render_backend="none", obs_mode="state_dict")
    try:
        for offset in range(0, len(geometries), batch):
            group = geometries[offset:offset + batch]
            for (motion, slots), starts in zip(group, select_starts(env, group, args.seed)):
                for split, case, rank in slots:
                    entry = dict(motion.metadata(), id=case, split=split, start_rank=rank + 1)
                    if rank >= len(starts):
                        missing.append(dict(entry, reason="insufficient reachable starts"))
                    else:
                        start = STARTS[starts[rank]]
                        selected[split].append((motion, dict(entry, start_xyz_m=start.tolist())))
            print(f"Checked starts for {min(offset + batch, len(geometries))}/{len(geometries)} "
                  f"geometries, missing episodes: {len(missing)}", flush=True)
    finally:
        env.close()
    return selected, missing


def record(selected, missing, args):
    import gymnasium as gym
    import h5py

    from .collect import run_episode
    from .metrics import measure
    from .verify import verify

    root = args.output
    manifest = []
    env = None
    batch = 0
    try:
        for split, entries in selected.items():
            for offset in range(0, len(entries), args.num_envs):
                group = entries[offset:offset + args.num_envs]
                if len(group) != batch:
                    if env is not None:
                        env.close()
                    batch = len(group)
                    env = gym.make("SpatialMotion-v1", num_envs=batch, sim_backend=args.sim,
                                   render_backend="gpu", obs_mode="rgb")
                directory = root / split / f"batch_{offset // args.num_envs:04d}"
                directory.parent.mkdir(exist_ok=True)
                metadata = [entry for _, entry in group]
                starts = np.asarray([entry["start_xyz_m"] for entry in metadata], dtype=np.float32)
                goals = targets_for([motion.path[1:] for motion, _ in group], starts)
                result = run_episode(env, goals, directory, "gpu", HOLD, TRACK_TOLERANCE_M,
                                     args.seed, directory.name, video_envs=args.video_envs,
                                     episodes=metadata, starts=starts, rgb_compression="gzip")
                verify(directory)
                with h5py.File(directory / "expert.h5") as data:
                    for slot, (motion, entry) in enumerate(group):
                        entry.update(shard=(directory / "expert.h5").relative_to(root).as_posix(),
                                     slot=slot, start_error_mm=result["start_mm"][slot],
                                     max_error_mm=result["max_mm"][slot],
                                     final_error_mm=result["final_mm"][slot])
                        if entry["start_error_mm"] > 2 or max(entry["max_error_mm"],
                                                              entry["final_error_mm"]) > 10:
                            missing.append(dict(entry, reason="failed recorded tracking"))
                            continue
                        if motion.kind in ("line", "circle"):
                            count = len(motion.path)
                            actual = data["state/tcp_pose_base"][:count, slot, :3]
                            entry["geometry_metrics"] = measure(entry, motion.path, actual)
                        manifest.append(entry)
                write_json(root / "manifest.json", manifest)
                write_json(root / "missing.json", missing)
    finally:
        if env is not None:
            env.close()
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan-only", action="store_true")
    mode.add_argument("--pilot", action="store_true")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--sim", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--video-envs", type=int, default=1)
    args = parser.parse_args()
    if args.sim == "cpu":
        args.num_envs = 1
    plan = {split: [(f"{split}-{index:04d}", motion) for index, motion in enumerate(motions(split))]
            for split in SPLITS}
    if args.pilot:
        plan = pilot_plan(plan)
    geometries = requests(plan, args.pilot)
    args.output.mkdir(parents=True)
    counts = {split: sum(item[0] == split for _, group in geometries for item in group)
              for split in SPLITS}
    write_json(args.output / "plan.json", {"episodes": counts, "geometries": [
        dict(motion.metadata(), episodes=[dict(split=split, id=case, start_rank=rank + 1)
                                           for split, case, rank in slots])
        for motion, slots in geometries
    ]})
    print(f"Planned episodes: {counts}", flush=True)
    plot_plan(plan, args.output / "plan")
    if args.plan_only:
        return 0
    selected, missing = plan_starts(geometries, min(args.num_envs, len(geometries)), args)
    manifest = record(selected, missing, args)
    write_json(args.output / "manifest.json", manifest)
    write_json(args.output / "missing.json", missing)
    plot_recorded(args.output, manifest)
    print(f"Accepted {len(manifest)} episodes, missing {len(missing)}", flush=True)
    return int(bool(missing))


if __name__ == "__main__":
    raise SystemExit(main())
