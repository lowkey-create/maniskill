"""Multi-stroke SVG path drawing in ManiSkill 3.0.1 (DrawSVG-v1).

Run:
    python draw_svg_multistroke.py

The demo uses two disconnected strokes forming an 'A'.  It records an MP4,
ManiSkill HDF5 rollout, and a per-control-step CSV file.
"""

from pathlib import Path
import csv
import numpy as np
import gymnasium as gym
import mani_skill.envs  # registers ManiSkill Gym environments
from svgpathtools import parse_path, Line, QuadraticBezier, CubicBezier
from transforms3d.quaternions import quat2mat
from mani_skill.envs.tasks.drawing.draw_svg import DrawSVGEnv
from mani_skill.utils.wrappers.record import RecordEpisode

# Two disjoint strokes: the sides of A, then its crossbar.
# We start with path-data syntax (the "d" attribute), not a whole .svg document.
SVG_D = "M 20 80 L 50 20 L 80 80 M 32 60 L 68 60"
OUT = Path("results/svg_A")
OUT.mkdir(parents=True, exist_ok=True)
DRAW_Z = 0.024      # canvas top 0.020 m; ink threshold is < 0.028 m
LIFT_Z = 0.090      # clearly off the canvas
SPACING_M = 0.008   # target waypoints at roughly 8 mm intervals
MAX_STEPS = 460     # DrawSVG's internal score buffer is only 500 steps

# Smaller visible ink marks; this must be set before constructing the scene.
DrawSVGEnv.BRUSH_RADIUS = 0.004
# The stock task's 0.1 m threshold is generous; use a stricter diagnostic.
DrawSVGEnv.THRESHOLD = 0.01

path = parse_path(SVG_D)
subpaths = path.continuous_subpaths()
assert len(subpaths) >= 2, "This example needs multiple strokes"

# Reproduce the stock DrawSVG-v1 normalization EXACTLY for L/Q/C segments.
# (It uses 5 parameter samples on each Bezier, then scales by maximum value,
# and centers by the mean of the start of each resulting line segment.)
official_lines = []
for seg in path:
    if isinstance(seg, Line):
        points = [seg.start, seg.end]
    elif isinstance(seg, (QuadraticBezier, CubicBezier)):
        points = [seg.point(float(t)) for t in np.linspace(0, 1, 5)]
    else:
        raise ValueError("The stock DrawSVG goal renderer only handles L/Q/C. Convert arcs/shapes to paths first.")
    for a, b in zip(points[:-1], points[1:]):
        official_lines.append([[a.real, a.imag], [b.real, b.imag]])

raw_lines = np.asarray(official_lines, dtype=np.float64)
scale = 0.25 / float(raw_lines.max())
center_xy = (raw_lines[:, 0, :] * scale).mean(axis=0)

env = gym.make(
    "DrawSVG-v1",
    svg=SVG_D,
    obs_mode="rgb",  # records camera observations for a future VLA dataset
    control_mode="pd_ee_delta_pos",  # panda_stick: THREE actions, x/y/z
    sim_backend="physx_cpu",
    render_mode="rgb_array",
    max_episode_steps=490,
)
# Some 3.0.1 versions don't initialize this for custom continuous SVGs.
env.unwrapped.continuous = len(subpaths) == 1

env = RecordEpisode(
    env,
    output_dir=str(OUT),
    save_video=True,
    save_trajectory=True,
    trajectory_name="expert_demo",
    video_fps=20,
)
obs, info = env.reset(seed=42)
raw = env.unwrapped

# The goal outline pose supplies the environment's RANDOM yaw and translation.
# pose.q uses the (w, x, y, z) convention expected by transforms3d.
goal_p = raw.goal_outline.pose.p[0].detach().cpu().numpy().astype(np.float64)
goal_q = raw.goal_outline.pose.q[0].detach().cpu().numpy().astype(np.float64)
rotation = quat2mat(goal_q)

def to_world(p):
    local = np.array([
        p.real * scale - center_xy[0],
        p.imag * scale - center_xy[1],
        0.01,
    ])
    return rotation @ local + goal_p

# Sanity check: our transform must match the environment's first world goal point.
expected_first = to_world(path[0].start)
actual_first = raw.points[0, 0].detach().cpu().numpy()
error = np.linalg.norm(expected_first - actual_first)
print(f"坐标变换校验误差: {error:.8f} m")
if error > 1e-4:
    raise RuntimeError("SVG normalization/pose differs from this script; inspect the installed DrawSVG source")

# Interpolate each subpath independently, so a pen-up occurs between subpaths.
strokes = []
for subpath in subpaths:
    samples = []
    for segment in subpath:
        n = max(2, int(np.ceil(segment.length() * scale / SPACING_M)))
        for j in range(n + 1):
            if samples and j == 0:
                continue
            samples.append(to_world(segment.point(j / n)))
    strokes.append(np.asarray(samples))

print("笔画数量:", len(strokes))
print("每笔目标点数:", [len(s) for s in strokes])
print("动作维度:", env.action_space.shape)
assert env.action_space.shape[-1] == 3, "panda_stick pd_ee_delta_pos must be 3D"

rows = []
step_id = 0

def tcp():
    return raw.agent.tcp.pose.p[0].detach().cpu().numpy().copy()

def move_to(target, stroke_id, phase, max_iters=12, tol=0.004):
    """Feedback position control, with automatic failure if trajectory budget is used."""
    global step_id
    target = np.asarray(target, dtype=np.float64)
    for _ in range(max_iters):
        current = tcp()
        err = target - current
        if np.linalg.norm(err) < tol:
            break
        if step_id >= MAX_STEPS:
            raise RuntimeError("Reached safe 460-step limit of the stock DrawSVG-v1 task")
        # PandaStick maps normalized [-1, 1] to target EE deltas [-0.1, 0.1] m.
        action = np.clip(err / 0.1, -0.30, 0.30).astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        step_id += 1
        now = tcp()
        rows.append([step_id, stroke_id, phase, *target, *now, *action])
        if bool(truncated.any()):
            raise RuntimeError("Episode truncated before the drawing was complete")
    return float(np.linalg.norm(target - tcp()))

try:
    for sid, stroke in enumerate(strokes):
        print(f"开始第 {sid + 1} 笔，{len(stroke)} 个点")
        first = stroke[0].copy()
        above_first = first.copy()
        above_first[2] = LIFT_Z
        move_to(above_first, sid, "approach", max_iters=55, tol=0.006)
        first[2] = DRAW_Z
        move_to(first, sid, "pen_down", max_iters=30, tol=0.003)

        largest_error = 0.0
        for point in stroke[1:]:
            goal = point.copy()
            goal[2] = DRAW_Z
            e = move_to(goal, sid, "draw", max_iters=5, tol=0.004)
            largest_error = max(largest_error, e)
        print(f"第 {sid + 1} 笔最大末端目标残差: {largest_error*1000:.1f} mm")

        above_last = stroke[-1].copy()
        above_last[2] = LIFT_Z
        move_to(above_last, sid, "pen_up", max_iters=30, tol=0.006)

finally:
    with (OUT / "controller_steps.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "stroke", "phase", "target_x", "target_y", "target_z",
                         "tcp_x", "tcp_y", "tcp_z", "action_x", "action_y", "action_z"])
        writer.writerows(rows)
    env.close()

print(f"完成：共执行 {step_id} 个控制步；输出目录 {OUT}")
print("检查：expert_demo.h5、expert_demo.json、MP4、controller_steps.csv")
