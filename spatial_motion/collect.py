"""Run and record a batched closed-loop Cartesian expert."""
import argparse
import csv
import json
from contextlib import ExitStack, closing
from pathlib import Path

import gymnasium as gym
import h5py
import numpy as np
import torch
from mani_skill.utils.geometry.rotation_conversions import matrix_to_euler_angles, quaternion_to_matrix

from .env import CAMERAS, CONTROL_MODE
from .trajectories import NAMES, load_waypoints, reference

ACTION_COLUMNS = ("target_x_m", "target_y_m", "target_z_m",
                  "target_rx_rad", "target_ry_rad", "target_rz_rad", "gripper")


def as_numpy(tensor):
    return tensor if isinstance(tensor, np.ndarray) else tensor.detach().cpu().numpy()


APPROACH_STEPS = 60
APPROACH_HOLD = 20


def prepare(env, seed, starts=None):
    """Reset, lock the wrist pose, and move to the requested start before capture."""
    obs, _ = env.reset(seed=seed)
    raw = env.unwrapped
    rotation = matrix_to_euler_angles(quaternion_to_matrix(raw.tcp_in_base.q), "XYZ")
    grip = torch.ones((raw.num_envs, 1), device=raw.device)
    tail = torch.cat((rotation, grip), dim=1)
    if starts is not None:
        initial = raw.tcp_in_base.p.clone()
        delta = torch.as_tensor(starts, device=raw.device) - initial
        for step in range(1, APPROACH_STEPS + APPROACH_HOLD + 1):
            t = min(step / APPROACH_STEPS, 1)
            target = initial + t * t * (3 - 2 * t) * delta
            obs, _, _, _, _ = env.step(torch.cat((target, tail), dim=1))
    return obs, tail


def save_state(group, state):
    for key, value in state.items():
        if isinstance(value, dict):
            save_state(group.create_group(key), value)
        else:
            group.create_dataset(key, data=as_numpy(value))


def run_episode(env, targets, directory, render, hold, tolerance, seed, name, renderer=None,
                video_envs=None, episodes=None, starts=None, rgb_compression="lzf"):
    directory.mkdir()
    raw = env.unwrapped
    obs, tail = prepare(env, seed, starts)
    if renderer is not None:
        obs = renderer.capture()
    robot = raw.agent.robot
    batch = raw.num_envs
    fps = raw.control_freq
    video_envs = batch if video_envs is None else min(video_envs, batch)
    targets = np.concatenate((targets, np.repeat(targets[-1:], hold, axis=0)))
    count = len(targets)
    target_tensor = torch.as_tensor(targets, device=raw.device)
    errors = np.empty((count, batch), np.float32)
    if starts is not None:
        start_error = np.linalg.norm(starts - as_numpy(raw.tcp_in_base.p), axis=1)

    with ExitStack() as stack:
        h5 = stack.enter_context(h5py.File(directory / "expert.h5", "w"))
        h5.attrs.update(
            env_id="SpatialMotion-v1", control_mode=CONTROL_MODE, frame="panda_base",
            position_unit="m", rotation_unit="rad", action_layout=",".join(ACTION_COLUMNS),
            control_hz=fps, seed=seed, trajectory=name, hold_steps=hold,
            state_alignment="state[t], action[t], state[t+1]", complete=False,
            sim_backend=raw.backend.sim_backend, render_backend=render,
            render_device=renderer.device if renderer is not None else str(raw.backend.render_backend),
        )
        if render != "none":
            h5.attrs["video_envs"] = video_envs
        if starts is not None:
            h5.attrs["preroll_steps"] = APPROACH_STEPS + APPROACH_HOLD
            save_state(h5.create_group("initial_state"), raw.get_state_dict())
        if episodes is not None:
            h5.create_dataset("episode/metadata_json", data=[json.dumps(item, ensure_ascii=False)
                                                            for item in episodes],
                              dtype=h5py.string_dtype("utf-8"))
        qpos = h5.create_dataset("state/qpos", (count + 1, batch, 9), "f4")
        qvel = h5.create_dataset("state/qvel", qpos.shape, "f4")
        tcp = h5.create_dataset("state/tcp_pose_base", (count + 1, batch, 7), "f4")
        actions = h5.create_dataset("action", (count, batch, len(ACTION_COLUMNS)), "f4")
        h5.create_dataset("target_xyz_base_m", data=targets)
        error_vectors = h5.create_dataset("error_xyz_base_m", targets.shape, "f4")
        norms = h5.create_dataset("error_norm_m", errors.shape, "f4")
        done = h5.create_dataset("terminated", errors.shape, "?")
        truncated = h5.create_dataset("truncated", errors.shape, "?")
        rows = csv.writer(stack.enter_context((directory / "tracking.csv").open("w", newline="")))
        rows.writerow(["step", "env", "time_s", "target_x_m", "target_y_m", "target_z_m",
                       "tcp_x_m", "tcp_y_m", "tcp_z_m", "error_x_m", "error_y_m",
                       "error_z_m", "error_norm_m", *(f"action_{key}" for key in ACTION_COLUMNS)])

        videos = {}
        rgb = {}
        calibration = {}
        if render != "none":
            import imageio.v2 as imageio

            for camera in CAMERAS:
                shape = tuple(obs["sensor_data"][camera]["rgb"].shape)
                rgb[camera] = h5.create_dataset(
                    f"rgb/{camera}", (count + 1, *shape), "u1",
                    chunks=(1, 1, *shape[1:]), compression=rgb_compression,
                    compression_opts=1 if rgb_compression == "gzip" else None,
                )
                calibration[camera] = {
                    key: h5.create_dataset(f"camera_param/{camera}/{key}",
                                           (count + 1, *tensor.shape), "f4")
                    for key, tensor in obs["sensor_param"][camera].items()
                }
                for index in range(video_envs):
                    path = directory / f"{camera}_env{index:03d}.mp4"
                    videos[camera, index] = stack.enter_context(closing(imageio.get_writer(
                        path, fps=fps, codec="libx264",
                        ffmpeg_params=["-threads", "1"],
                    )))

        def record_state(index, observation):
            qpos[index] = as_numpy(robot.get_qpos())
            qvel[index] = as_numpy(robot.get_qvel())
            tcp_pose = raw.tcp_in_base.raw_pose
            tcp[index] = as_numpy(tcp_pose)
            for camera in rgb:
                frames = as_numpy(observation["sensor_data"][camera]["rgb"])
                rgb[camera][index] = frames
                for key, dataset in calibration[camera].items():
                    dataset[index] = as_numpy(observation["sensor_param"][camera][key])
                for slot in range(video_envs):
                    videos[camera, slot].append_data(frames[slot])
            return tcp_pose[:, :3]

        record_state(0, obs)
        for step, target in enumerate(target_tensor):
            action = torch.cat((target, tail), dim=1)
            obs, _, term, trunc, _ = env.step(action)
            if renderer is not None:
                obs = renderer.capture()
            current = record_state(step + 1, obs)
            actual, command = as_numpy(current), as_numpy(action)
            residual = targets[step] - actual
            errors[step] = np.linalg.norm(residual, axis=1)
            actions[step] = command
            error_vectors[step] = residual
            norms[step] = errors[step]
            done[step], truncated[step] = as_numpy(term), as_numpy(trunc)
            if not np.isfinite(actual).all() or term.any() or trunc.any():
                raise RuntimeError(f"Invalid state or episode ended at step {step}")
            for slot in range(batch):
                rows.writerow([step, slot, (step + 1) / fps, *targets[step, slot],
                               *actual[slot], *residual[slot], errors[step, slot], *command[slot]])
        h5.attrs["complete"] = True

    lengths = ([entry["motion_samples"] for entry in episodes]
               if episodes is not None else [count - hold] * batch)
    tracking = [errors[:length, slot] for slot, length in enumerate(lengths)]
    maximum = [float(1000 * path.max()) for path in tracking]
    final = [float(1000 * errors[length + hold - 1, slot]) for slot, length in enumerate(lengths)]
    result = {
        "trajectory": name, "num_envs": batch, "tracking_steps": count - hold,
        "hold_steps": hold, "control_hz": fps, "render": render,
        "video_envs": video_envs if render != "none" else 0,
        "rms_mm": [float(1000 * np.sqrt(np.mean(path ** 2))) for path in tracking],
        "p95_mm": [float(1000 * np.percentile(path, 95)) for path in tracking],
        "max_mm": maximum, "final_mm": final, "tolerance_mm": 1000 * tolerance,
        "passed": max(maximum + final) <= 1000 * tolerance,
    }
    if starts is not None:
        result["start_mm"] = (1000 * start_error).tolist()
        result["passed"] &= bool((start_error <= 0.002).all())
    (directory / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--trajectory", choices=(*NAMES, "all"), default="all")
    inputs.add_argument("--waypoints", type=Path, help="Base-frame xyz in a .npy array")
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--hold", type=int, default=30)
    parser.add_argument("--tolerance-mm", type=float, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--sim", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--render", choices=("none", "cpu", "gpu", "software"), default="none")
    args = parser.parse_args()
    if (args.num_envs < 1 or args.steps < 2 or args.hold < 0
            or not np.isfinite(args.tolerance_mm) or args.tolerance_mm <= 0):
        parser.error("Expected positive environment count/tolerance, steps >= 2, hold >= 0")
    if args.sim == "cpu" and args.num_envs != 1:
        parser.error("ManiSkill PhysX CPU supports one environment")
    if args.render == "software" and args.sim != "cpu":
        parser.error("Software rendering requires single-environment CPU simulation")
    custom = load_waypoints(args.waypoints, args.num_envs) if args.waypoints else None
    if (len(custom) if custom is not None else args.steps) + args.hold >= 20000:
        parser.error("Trajectory and hold must be shorter than the episode limit")

    native_render = "none" if args.render == "software" else args.render
    env = gym.make(
        "SpatialMotion-v1", num_envs=args.num_envs, sim_backend=args.sim,
        render_backend=native_render, obs_mode="state_dict" if native_render == "none" else "rgb",
    )
    renderer = None
    try:
        if args.render == "software":
            from .software_render import SoftwareRenderer

            renderer = SoftwareRenderer(env.unwrapped)
        args.output.mkdir(parents=True)
        names = ("custom",) if custom is not None else (
            NAMES if args.trajectory == "all" else (args.trajectory,)
        )
        results = []
        for name in names:
            env.reset(seed=args.seed)
            start = as_numpy(env.unwrapped.tcp_in_base.p)
            targets = custom if custom is not None else reference(name, start, args.steps)
            results.append(run_episode(env, targets, args.output / name, args.render,
                                       args.hold, args.tolerance_mm / 1000, args.seed, name, renderer))
        (args.output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
        return 0 if all(result["passed"] for result in results) else 1
    finally:
        try:
            if renderer is not None:
                renderer.close()
        finally:
            env.close()


if __name__ == "__main__":
    raise SystemExit(main())
