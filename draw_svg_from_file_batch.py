"""Draw an SVG file with ManiSkill 3.0.1 DrawSVG-v1 / PandaStick.

Examples (inside the existing ManiSkill Python environment):
    python draw_svg_from_file.py svgs/test_A.svg
    python draw_svg_from_file.py svgs/flower.svg --preview-only
    python draw_svg_from_file.py svgs/flower.svg --max-steps 8000 --spacing 0.008
    python draw_svg_from_file.py svgs/flower.svg --max-steps 8000 --step-reserve 128

Features:
  - Reads actual SVG XML (paths, lines, polygons, rectangles, circles, ellipses).
  - Handles SVG transform attributes and keeps disconnected strokes separate.
  - Converts curves to line segments compatible with the stock DrawSVG renderer.
  - Runs the existing position-feedback controller; records MP4/HDF5/CSV.
  - Extends the stock 500-step evaluation buffer and ink-dot capacity WITHOUT
    modifying the installed ManiSkill source code.

Note:
  - One SVG is drawn per environment instance (one env.reset() per run).
  - The stock ManiSkill 3.0.1 reset function hardcodes a 500-element buffer.
    This script grows the buffer AFTER its one reset. Do not add another reset
    without applying the same resize before further steps (and patching the
    stock reset implementation, which otherwise expects exactly 500 entries).
  - Larger --max-steps creates MORE ink-dot actors during scene construction;
    using many thousands of steps can be slow and memory intensive.
  - This draws vector OUTLINES, not fills/text/images/clipping/CSS effects.
"""

import argparse
import csv
import json
import math
import shutil
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
from svgpathtools import parse_path
from svgpathtools.document import CONVERSIONS
from svgpathtools.parser import parse_transform
from svgpathtools.path import transform


# ============================================================================
# 1. 统一参数区：改这里的默认值，或运行时使用对应的命令行参数
# ============================================================================
@dataclass(frozen=True)
class DrawingConfig:
    # 步数：可通过 --max-steps / --step-reserve 覆盖
    max_steps: int = 2000                # 脚本最多发出多少次 env.step(action)
    step_reserve: int = 64               # 预留步数，避免内部数组和 Episode 过早触顶

    #图片尺寸
    target_draw_size_m: float = 0.45

    # 轨迹采样：可通过 --spacing 覆盖
    spacing_m: float = 0.008             # 相邻轨迹目标点的近似间距（米）
    seed: int = 42

    # 机器人运动：--action-limit、--draw-iters、--draw-tol 可以覆盖
    action_scale_m: float = 0.10         # pd_ee_delta_pos: action=1 对应 0.10 m 指令位移
    action_limit: float = 0.30           # 每个动作分量限制在 [-0.30, 0.30]
    approach_iters: int = 55
    pen_down_iters: int = 30
    draw_iters: int = 5
    pen_up_iters: int = 30
    approach_tol_m: float = 0.006
    pen_down_tol_m: float = 0.003
    draw_tol_m: float = 0.004
    pen_up_tol_m: float = 0.006

    # 绘制和坐标映射：基于原成功 A 示例，通常不要动
    draw_z_m: float = 0.024
    lift_z_m: float = 0.090
    norm_span: float = 70.0
    norm_center: float = 50.0
    canvas_local_z_m: float = 0.010
    brush_radius_m: float = 0.004
    success_threshold_m: float = 0.010
    video_fps: int = 20

    @property
    def episode_limit(self) -> int:
        """Gym 的 Episode 截断阈值，始终高于脚本 max_steps。"""
        return self.max_steps + self.step_reserve // 2

    @property
    def dot_capacity(self) -> int:
        """墨点对象 + dots_dist 记录数组的共同容量。"""
        return max(1000, self.max_steps + self.step_reserve)


DEFAULTS = DrawingConfig()


def parse_args():
    parser = argparse.ArgumentParser(
        description="SVG file -> ManiSkill DrawSVG expert trajectory (extended steps)"
    )
    parser.add_argument("svg", type=Path, help="输入 SVG，例如 svgs/flower.svg")
    parser.add_argument("--output", type=Path, default=None,
                        help="输出目录；默认 results/<SVG 文件名>")

    group = parser.add_argument_group("轨迹和步数")
    group.add_argument("--spacing", type=float, default=DEFAULTS.spacing_m,
                       help=f"轨迹采样间距，单位 m（默认 {DEFAULTS.spacing_m}）")
    group.add_argument("--max-steps", type=int, default=DEFAULTS.max_steps,
                       help=f"脚本控制步数上限（默认 {DEFAULTS.max_steps}）")
    group.add_argument("--step-reserve", type=int, default=DEFAULTS.step_reserve,
                       help=f"Episode/墨点容量的预留步数（默认 {DEFAULTS.step_reserve}）")
    group.add_argument("--seed", type=int, default=DEFAULTS.seed)

    group = parser.add_argument_group("运动控制（不指定就沿用原参数）")
    group.add_argument("--action-limit", type=float, default=DEFAULTS.action_limit,
                       help="单个动作维度的最大归一化幅度，必须在 (0, 1] 内")
    group.add_argument("--draw-iters", type=int, default=DEFAULTS.draw_iters,
                       help="每个绘画目标点最多尝试多少个控制步")
    group.add_argument("--draw-tol", type=float, default=DEFAULTS.draw_tol_m,
                       help="绘画目标点到达判据，单位 m")

    parser.add_argument("--preview-only", action="store_true",
                        help="只解析 SVG，输出预览与参数，不启动仿真")
    parser.add_argument("--no-video", action="store_true",
                        help="不生成 MP4（HDF5 专家轨迹照常记录）")
    parser.add_argument("--no-csv", action="store_true",
                        help="不额外写逐步 controller_steps.csv（HDF5 动作照常记录）")
    parser.add_argument("--obs-mode", choices=("rgb", "state"), default="rgb",
                        help="rgb 适合后续 VLA；state 更快、更省空间但不包含图像观测")
    args = parser.parse_args()

    cfg = replace(
        DEFAULTS,
        spacing_m=args.spacing,
        max_steps=args.max_steps,
        step_reserve=args.step_reserve,
        seed=args.seed,
        action_limit=args.action_limit,
        draw_iters=args.draw_iters,
        draw_tol_m=args.draw_tol,
    )
    validate_config(cfg, parser)
    return args.svg, args.output, args.preview_only, cfg, args.no_video, args.no_csv, args.obs_mode


def validate_config(cfg, parser):
    if cfg.max_steps < 1:
        parser.error("--max-steps 必须 >= 1")
    if cfg.step_reserve < 2:
        parser.error("--step-reserve 必须 >= 2")
    if not (math.isfinite(cfg.spacing_m) and cfg.spacing_m > 0):
        parser.error("--spacing 必须是正的有限数")
    if not (math.isfinite(cfg.action_limit) and 0 < cfg.action_limit <= 1):
        parser.error("--action-limit 必须在 (0, 1] 内")
    if cfg.draw_iters < 1:
        parser.error("--draw-iters 必须 >= 1")
    if not (math.isfinite(cfg.draw_tol_m) and cfg.draw_tol_m > 0):
        parser.error("--draw-tol 必须是正的有限数")
    if cfg.lift_z_m <= cfg.draw_z_m:
        parser.error("lift_z_m 必须高于 draw_z_m")


# ============================================================================
# 2. SVG 解析、采样和归一化（保留原脚本的处理逻辑）
# ============================================================================
def local_name(xml_tag):
    return xml_tag.rsplit("}", 1)[-1]


def read_svg_strokes(file_path):
    """按 SVG XML 顺序读取轮廓，累积父子级 transform。"""
    root = ET.parse(file_path).getroot()
    result = []
    unsupported = {"text", "tspan", "textPath", "image", "use", "foreignObject"}
    skip = {"defs", "clipPath", "mask", "symbol", "pattern", "marker", "filter"}
    unsupported_seen = set()

    def visit(element, inherited):
        tag = local_name(element.tag)
        if tag in skip:
            return
        style = element.attrib.get("style", "").replace(" ", "").lower()
        if element.attrib.get("display", "") == "none" or "display:none" in style:
            return
        if element.attrib.get("visibility", "") == "hidden" or "visibility:hidden" in style:
            return
        if "transform:" in style:
            raise ValueError("CSS style='transform:...' 不支持；请改用 transform='...'")
        if tag in unsupported:
            unsupported_seen.add(tag)
            return

        matrix = inherited @ parse_transform(element.attrib.get("transform"), strict=True)
        if tag in CONVERSIONS:
            d_string = CONVERSIONS[tag](element)
            if d_string.strip():
                path = transform(parse_path(d_string), matrix)
                for subpath in path.continuous_subpaths():
                    if len(subpath):
                        result.append(subpath)
        for child in element:
            visit(child, matrix)

    visit(root, np.eye(3))
    if unsupported_seen:
        print("注意：跳过了不支持的 SVG 元素:", ", ".join(sorted(unsupported_seen)))
        print("      文字请先转路径；<use> 请展开；位图需先矢量化。")
    if not result:
        raise ValueError("未解析到可绘制矢量路径（需要 path/line/circle/rect 等 SVG 元素）")
    return result


def flatten_and_normalize(paths, cfg):
    """离散采样成折线，并把图案映射到原版 DrawSVG 兼容的坐标范围。"""
    xs_min, xs_max, ys_min, ys_max = [], [], [], []
    for subpath in paths:
        for segment in subpath:
            xmin, xmax, ymin, ymax = segment.bbox()
            xs_min.append(xmin)
            xs_max.append(xmax)
            ys_min.append(ymin)
            ys_max.append(ymax)
    min_x, max_x = min(xs_min), max(xs_max)
    min_y, max_y = min(ys_min), max(ys_max)
    span = max(max_x - min_x, max_y - min_y)
    if not np.isfinite(span) or span < 1e-8:
        raise ValueError("图形边界无效或所有点重合")

    cx = (min_x + max_x) / 2.0
    cy = (min_y + max_y) / 2.0
    # 输出坐标约为 [15, 85]。原版环境用 0.25/max 缩放 SVG。
    approx_m_per_svg_unit = (cfg.norm_span / span) * (0.25 / 85.0)

    def normalize(z):
        return complex(
            cfg.norm_center + (z.real - cx) * cfg.norm_span / span,
            cfg.norm_center + (z.imag - cy) * cfg.norm_span / span,
        )

    polylines = []
    for subpath in paths:
        points = []
        for segment in subpath:
            length = segment.length(error=1e-4)
            n = max(1, int(math.ceil(length * approx_m_per_svg_unit / cfg.spacing_m)))
            if type(segment).__name__ != "Line":
                n = max(4, n)
            for j in range(n + 1):
                if points and j == 0:
                    continue
                points.append(normalize(segment.point(j / n)))
        cleaned = []
        for point in points:
            if not cleaned or abs(point - cleaned[-1]) > 1e-6:
                cleaned.append(point)
        if len(cleaned) >= 2:
            polylines.append(cleaned)

    if not polylines:
        raise ValueError("采样后没有有效线段")
    commands = []
    for points in polylines:
        commands.append(f"M {points[0].real:.7f} {points[0].imag:.7f}")
        commands.extend(f"L {p.real:.7f} {p.imag:.7f}" for p in points[1:])
    svg_d = " ".join(commands)
    return polylines, svg_d


def stock_mapping(svg_d, goal_pose_p, goal_pose_q, cfg):
    """精确复现 ManiSkill 3.0.1 的 SVG 归一化和随机旋转/平移。"""
    from transforms3d.quaternions import quat2mat

    path = parse_path(svg_d)
    lines = [
        [[segment.start.real, segment.start.imag],
         [segment.end.real, segment.end.imag]]
        for segment in path
    ]
    raw_lines = np.asarray(lines, dtype=np.float64)
    scale = cfg.target_draw_size_m / float(raw_lines.max())
    center_xy = (raw_lines[:, 0, :] * scale).mean(axis=0)
    rotation = quat2mat(goal_pose_q)

    def to_world(point):
        local = np.array([
            point.real * scale - center_xy[0],
            point.imag * scale - center_xy[1],
            cfg.canvas_local_z_m,
        ])
        return rotation @ local + goal_pose_p

    return path, to_world


def write_goal_svg(output, svg_d):
    (output / "normalized_goal.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">\n'
        f'  <path d="{svg_d}" stroke="#222" stroke-width="1" fill="none"/>\n'
        '</svg>\n', encoding="utf-8"
    )


def write_metadata(output, data):
    (output / "drawing_metadata.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ============================================================================
# 3. 执行绘画与数据导出
# ============================================================================
def run():
    svg_arg, output_arg, preview_only, cfg, no_video, no_csv, obs_mode = parse_args()
    svg_file = svg_arg.expanduser().resolve()
    if not svg_file.is_file():
        raise FileNotFoundError(f"SVG file not found: {svg_file}")

    output = output_arg or (Path("results") / svg_file.stem)
    output.mkdir(parents=True, exist_ok=True)
    paths = read_svg_strokes(svg_file)
    polylines, svg_d = flatten_and_normalize(paths, cfg)
    point_counts = [len(p) for p in polylines]
    total_waypoints = sum(point_counts)

    print(f"输入 SVG: {svg_file}")
    print(f"笔画数量: {len(polylines)}")
    print(f"各笔画目标点数: {point_counts}")
    print(f"总目标点数: {total_waypoints}; 单步目标间距: {cfg.spacing_m*1000:.1f} mm")
    print(f"步数规划: 脚本 {cfg.max_steps}, Gym {cfg.episode_limit}, 内部容量 {cfg.dot_capacity}")
    if total_waypoints >= cfg.max_steps:
        print("警告：目标点数已经达到或超过步数预算（抬笔/落笔也消耗步数）。")
        print("      可以增大 --max-steps 或 --spacing。")
    if cfg.dot_capacity >= 5000 and not preview_only:
        print(f"注意：环境将预创建 {cfg.dot_capacity} 个墨点对象，构建/渲染可能较慢。")

    write_goal_svg(output, svg_d)
    shutil.copy2(svg_file, output / "source.svg")
    metadata = {
        "source_file": str(svg_file),
        "stroke_count": len(polylines),
        "points_per_stroke": point_counts,
        "total_waypoints": total_waypoints,
        "spacing_m": cfg.spacing_m,
        "max_steps": cfg.max_steps,
        "seed": cfg.seed,
        "draw_z_m": cfg.draw_z_m,
        "lift_z_m": cfg.lift_z_m,
        "limits": {
            "script_max_steps": cfg.max_steps,
            "gym_episode_limit": cfg.episode_limit,
            "ink_and_evaluation_capacity": cfg.dot_capacity,
        },
        "controller_config": asdict(cfg),
        "run_status": "preview_only" if preview_only else "prepared",
        "obs_mode": obs_mode,
        "save_video": not no_video,
        "save_csv": not no_csv,
        "task_success": None,
        "note": "Programmatic controller demonstration, not a VLA policy rollout.",
    }
    write_metadata(output, metadata)
    if preview_only:
        print("预览完成：normalized_goal.svg、source.svg、drawing_metadata.json")
        return

    # 只有开始物理仿真时才导入 ManiSkill / PyTorch。
    import gymnasium as gym
    import torch
    import mani_skill.envs  # noqa: F401  注册 ManiSkill Gym 环境
    from mani_skill.envs.tasks.drawing.draw_svg import DrawSVGEnv
    from mani_skill.utils.wrappers.record import RecordEpisode

    # 重要：这些类属性必须在 gym.make() 之前设置，因为墨点在建场景时创建。
    DrawSVGEnv.BRUSH_RADIUS = cfg.brush_radius_m
    DrawSVGEnv.THRESHOLD = cfg.success_threshold_m
    DrawSVGEnv.MAX_DOTS = cfg.dot_capacity

    env = gym.make(
        "DrawSVG-v1",
        svg=svg_d,
        obs_mode=obs_mode,
        control_mode="pd_ee_delta_pos",
        sim_backend="physx_cpu",
        render_mode="rgb_array",
        max_episode_steps=cfg.episode_limit,
    )
    env.unwrapped.continuous = len(polylines) == 1
    env = RecordEpisode(
        env,
        output_dir=str(output),
        save_video=not no_video,
        save_trajectory=True,
        trajectory_name="expert_demo",
        video_fps=cfg.video_fps,
    )

    rows = []
    step_id = 0
    completed = False
    task_success = False
    try:
        # 原版 _initialize_episode 使用固定 500 个元素初始化 dots_dist。
        # 因此必须先 reset 一次，再将其替换为扩大的记录数组。
        env.reset(seed=cfg.seed)



        # from unittest.mock import patch

        # raw = env.unwrapped

        # FIXED_ANGLE_DEG = 0.0  # 在这里控制角度，单位：度

        # def fixed_goal_rotation(b, **kwargs):
        #     angle = np.deg2rad(FIXED_ANGLE_DEG)

        #     q = torch.zeros(
        #         (b, 4),
        #         dtype=torch.float32,
        #         device=raw.device
        #     )

        #     q[:, 0] = np.cos(angle / 2)
        #     q[:, 3] = np.sin(angle / 2)

        #     return q

        # with patch(
        #     "mani_skill.envs.tasks.drawing.draw_svg.randomization.random_quaternions",
        #     side_effect=fixed_goal_rotation
        # ):
        #     env.reset(seed=cfg.seed)




        raw = env.unwrapped
        if len(raw.dots) < cfg.dot_capacity:
            raise RuntimeError(
                f"实际墨点数量 {len(raw.dots)} 小于所需容量 {cfg.dot_capacity}；"
                "请确认在 gym.make() 前设置了 DrawSVGEnv.MAX_DOTS"
            )
        raw.dots_dist = torch.full(
            (raw.num_envs, cfg.dot_capacity),
            -1.0,
            dtype=raw.dots_dist.dtype,
            device=raw.device,
        )
        print(f"容量检查: len(dots)={len(raw.dots)}, dots_dist.shape={tuple(raw.dots_dist.shape)}")

        goal_p = raw.goal_outline.pose.p[0].detach().cpu().numpy().astype(np.float64)
        goal_q = raw.goal_outline.pose.q[0].detach().cpu().numpy().astype(np.float64)
        flat_path, to_world = stock_mapping(svg_d, goal_p, goal_q, cfg)
        expected_first = to_world(flat_path[0].start)
        actual_first = raw.points[0, 0].detach().cpu().numpy()
        map_error = np.linalg.norm(expected_first - actual_first)
        print(f"坐标变换校验误差: {map_error:.8f} m")
        if map_error > 1e-4:
            raise RuntimeError("SVG 归一化/姿态与当前安装的 ManiSkill 源码不匹配")

        strokes = [np.asarray([to_world(point) for point in points]) for points in polylines]
        assert env.action_space.shape[-1] == 3, "panda_stick 需要 XYZ 三维动作"

        def tcp():
            return raw.agent.tcp.pose.p[0].detach().cpu().numpy().copy()

        def move_to(target, stroke_id, phase, max_iters, tol):
            nonlocal step_id
            target = np.asarray(target, dtype=np.float64)
            for _ in range(max_iters):
                current = tcp()
                error = target - current
                if np.linalg.norm(error) < tol:
                    break
                if step_id >= cfg.max_steps:
                    raise RuntimeError(
                        f"已到达脚本步数上限 --max-steps={cfg.max_steps}，"
                        "可以提高 --max-steps 或增大 --spacing"
                    )
                action = np.clip(
                    error / cfg.action_scale_m,
                    -cfg.action_limit,
                    cfg.action_limit,
                ).astype(np.float32)
                obs, reward, terminated, truncated, info = env.step(action)
                step_id += 1
                if not no_csv:
                    actual = tcp()
                    rows.append([step_id, stroke_id, phase, *target, *actual, *action])
                if bool(truncated.any()) if hasattr(truncated, "any") else bool(truncated):
                    raise RuntimeError("Gym Episode 被截断，但尚未完成绘画")
            return float(np.linalg.norm(target - tcp()))

        for stroke_id, stroke in enumerate(strokes):
            print(f"开始第 {stroke_id+1}/{len(strokes)} 笔：{len(stroke)} 点")
            first = stroke[0].copy()
            above_first = first.copy()
            above_first[2] = cfg.lift_z_m
            move_to(above_first, stroke_id, "approach",
                    cfg.approach_iters, cfg.approach_tol_m)
            first[2] = cfg.draw_z_m
            move_to(first, stroke_id, "pen_down",
                    cfg.pen_down_iters, cfg.pen_down_tol_m)

            max_err = 0.0
            for point in stroke[1:]:
                goal = point.copy()
                goal[2] = cfg.draw_z_m
                remaining = move_to(goal, stroke_id, "draw",
                                    cfg.draw_iters, cfg.draw_tol_m)
                max_err = max(max_err, remaining)
            print(f"第 {stroke_id+1} 笔最大跟踪残差: {max_err * 1000:.1f} mm")

            above_last = stroke[-1].copy()
            above_last[2] = cfg.lift_z_m
            move_to(above_last, stroke_id, "pen_up",
                    cfg.pen_up_iters, cfg.pen_up_tol_m)

        # 流程执行完不代表任务评测成功：另外记录官方 DrawSVG success_check。
        task_success = bool(raw.success_check().all().item())
        completed = True
        print(f"绘画完成：{len(strokes)} 笔，{step_id} 控制步，任务成功={task_success}，输出 {output}")
    finally:
        if not no_csv:
            with (output / "controller_steps.csv").open("w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([
                    "step", "stroke", "phase", "target_x", "target_y", "target_z",
                    "tcp_x", "tcp_y", "tcp_z", "action_x", "action_y", "action_z",
                ])
                writer.writerows(rows)
        metadata["actual_control_steps"] = step_id
        metadata["task_success"] = task_success if completed else None
        metadata["run_status"] = "completed" if completed else "failed_or_interrupted"
        write_metadata(output, metadata)
        env.close()

    print("结果：expert_demo.h5 / .json、drawing_metadata.json" +
          ("、MP4" if not no_video else "") + ("、controller_steps.csv" if not no_csv else ""))


if __name__ == "__main__":
    run()
