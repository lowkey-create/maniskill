"""Draw an SVG file with ManiSkill 3.0.1 DrawSVG-v1 / PandaStick.

Examples (run inside the existing ManiSkill virtual environment):
    python draw_svg_from_file.py svgs/test_A.svg
    python draw_svg_from_file.py svgs/flower.svg --preview-only
    python draw_svg_from_file.py svgs/flower.svg --max-steps 900 --spacing 0.010

Features:
  - Reads actual SVG XML, not just one `d` string.
  - Handles <path>, <line>, <polyline>, <polygon>, <rect>, <circle>,
    <ellipse> and common SVG group/element `transform` attributes.
  - Keeps individual strokes separate, with pen lift between them.
  - Converts arcs/curves into sampled straight segments because the stock
    DrawSVG goal renderer supports only L/Q/C.
  - Normalizes the drawing to a predictable ~20cm envelope.
  - Preserves the original A-demo's verified robot controller and logs.

Limitations: This is a vector *outline* drawing tool, NOT an SVG rasterizer.
Text, images, SVG <use>, clipping, fills, CSS transforms and styles are not
interpreted. Convert text to paths and expand <use> before using it.
Large drawings may exceed robot reach or require many control steps.
"""

import argparse
import csv
import json
import math
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from svgpathtools import parse_path
from svgpathtools.document import CONVERSIONS
from svgpathtools.parser import parse_transform
from svgpathtools.path import transform

# Same calibrated heights/controller as successful original A example.
DRAW_Z = 0.024
LIFT_Z = 0.090
NORM_SPAN = 70.0
NORM_CENTER = 50.0
CANVAS_LOCAL_Z = 0.010


def parse_args():
    parser = argparse.ArgumentParser(description="SVG file -> ManiSkill DrawSVG trajectory")
    parser.add_argument("svg", type=Path, help="Input .svg file, e.g. svgs/test_A.svg")
    parser.add_argument("--output", type=Path, default=None, help="Results directory (default: results/<SVG name>)")
    parser.add_argument("--spacing", type=float, default=0.008, help="Nominal drawing waypoint interval in meters (default 0.008)")
    parser.add_argument("--max-steps", type=int, default=460, help="Maximum robot control steps (default 460)")
    parser.add_argument("--seed", type=int, default=42, help="Environment random seed")
    parser.add_argument("--preview-only", action="store_true", help="Only parse, normalize and export goal SVG; do not start simulation")
    return parser.parse_args()


def local_name(xml_tag):
    return xml_tag.rsplit("}", 1)[-1]


def read_svg_strokes(file_path):
    """Read geometric elements in XML document order, accumulating SVG transforms."""
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
            raise ValueError("CSS style='transform:...' isn't supported. Use SVG transform='...' instead.")
        if tag in unsupported:
            unsupported_seen.add(tag)
            return

        matrix = inherited @ parse_transform(element.attrib.get("transform"), strict=True)
        if tag in CONVERSIONS:
            d_string = CONVERSIONS[tag](element)
            if d_string.strip():
                p = transform(parse_path(d_string), matrix)
                for subpath in p.continuous_subpaths():
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


def flatten_and_normalize(paths, spacing):
    """Sample vector geometry into individual polylines with physical ~spacing."""
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
    # With output coordinates around [15, 85], stock DrawSVG scales by
    # .25/max=~.25/85. A 70-unit line becomes ~20.6cm in world coordinates.
    approx_m_per_svg_unit = (NORM_SPAN / span) * (0.25 / 85.0)

    def normalize(z):
        return complex(
            NORM_CENTER + (z.real - cx) * NORM_SPAN / span,
            NORM_CENTER + (z.imag - cy) * NORM_SPAN / span,
        )

    polylines = []
    for subpath in paths:
        points = []
        for segment in subpath:
            length = segment.length(error=1e-4)
            n = max(1, int(math.ceil(length * approx_m_per_svg_unit / spacing)))
            # Use a minimum of 4 divisions on non-linear segments for curves.
            if type(segment).__name__ != "Line":
                n = max(4, n)
            for j in range(n + 1):
                if points and j == 0:
                    continue
                points.append(normalize(segment.point(j / n)))
        # Do not generate zero-length line segments (stock renderer can fail).
        cleaned = []
        for p in points:
            if not cleaned or abs(p - cleaned[-1]) > 1e-6:
                cleaned.append(p)
        if len(cleaned) >= 2:
            polylines.append(cleaned)

    if not polylines:
        raise ValueError("采样后没有有效线段")
    # Plain M/L-only SVG path data: compatible with stock DrawSVG environment.
    commands = []
    for points in polylines:
        commands.append(f"M {points[0].real:.7f} {points[0].imag:.7f}")
        commands.extend(f"L {p.real:.7f} {p.imag:.7f}" for p in points[1:])
    svg_d = " ".join(commands)
    return polylines, svg_d


def stock_mapping(svg_d, goal_pose_p, goal_pose_q):
    """Recreate ManiSkill 3.0.1 source normalization and the randomized pose."""
    from transforms3d.quaternions import quat2mat
    path = parse_path(svg_d)
    lines = []
    for segment in path:
        lines.append([[segment.start.real, segment.start.imag],
                      [segment.end.real, segment.end.imag]])
    raw_lines = np.asarray(lines, dtype=np.float64)
    scale = 0.25 / float(raw_lines.max())
    center_xy = (raw_lines[:, 0, :] * scale).mean(axis=0)
    rot = quat2mat(goal_pose_q)

    def to_world(p):
        loc = np.array([p.real * scale - center_xy[0],
                        p.imag * scale - center_xy[1], CANVAS_LOCAL_Z])
        return rot @ loc + goal_pose_p

    return path, to_world


def write_goal_svg(output, svg_d):
    (output / "normalized_goal.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">\n'
        f'  <path d="{svg_d}" stroke="#222" stroke-width="1" fill="none"/>\n'
        '</svg>\n', encoding="utf-8"
    )


def run():
    args = parse_args()
    svg_file = args.svg.expanduser().resolve()
    if not svg_file.is_file():
        raise FileNotFoundError(f"SVG file not found: {svg_file}")
    # if not (0.002 <= args.spacing <= 0.05):
    #     raise ValueError("--spacing must be between 0.002 and 0.05 meters")
    # if not (50 <= args.max_steps <= 2000):
    #     raise ValueError("--max-steps must be in [50, 2000]")

    output = args.output or Path("results") / svg_file.stem
    output.mkdir(parents=True, exist_ok=True)

    paths = read_svg_strokes(svg_file)
    polylines, svg_d = flatten_and_normalize(paths, args.spacing)
    point_counts = [len(p) for p in polylines]
    total_waypoints = sum(point_counts)
    print(f"输入 SVG: {svg_file}")
    print(f"笔画数量: {len(polylines)}")
    print(f"各笔画目标点数: {point_counts}")
    print(f"总目标点数: {total_waypoints}; 最大控制步数: {args.max_steps}")
    print(f"单步目标间距: {args.spacing * 1000:.1f} mm")
    if total_waypoints >= args.max_steps:
        print("警告：目标点数已超过控制步数预算，建议增大 --max-steps 或 --spacing。")

    write_goal_svg(output, svg_d)
    shutil.copy2(svg_file, output / "source.svg")
    metadata = {
        "source_file": str(svg_file),
        "stroke_count": len(polylines),
        "points_per_stroke": point_counts,
        "total_waypoints": total_waypoints,
        "spacing_m": args.spacing,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "draw_z_m": DRAW_Z,
        "lift_z_m": LIFT_Z,
        "note": "Programmatic controller demonstration, not a VLA policy rollout.",
    }
    (output / "drawing_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if args.preview_only:
        print("仅解析完成；已生成 normalized_goal.svg 和 drawing_metadata.json")
        return

    # Import GPU simulation only after successful parsing/preflight.
    import gymnasium as gym
    import torch
    import mani_skill.envs
    from mani_skill.envs.tasks.drawing.draw_svg import DrawSVGEnv
    from mani_skill.utils.wrappers.record import RecordEpisode

    # Save visible marks; allow more steps than the stock 500-step buffer.
    DrawSVGEnv.BRUSH_RADIUS = 0.004
    DrawSVGEnv.THRESHOLD = 0.01
    DrawSVGEnv.MAX_DOTS = max(1000, args.max_steps + 32)
    env = gym.make(
        "DrawSVG-v1", svg=svg_d, obs_mode="rgb",
        control_mode="pd_ee_delta_pos", sim_backend="physx_cpu",
        render_mode="rgb_array", max_episode_steps=args.max_steps + 10,
    )
    env.unwrapped.continuous = len(polylines) == 1
    env = RecordEpisode(
        env, output_dir=str(output), save_video=True,
        save_trajectory=True, trajectory_name="expert_demo", video_fps=20,
    )
    rows = []
    step_id = 0
    try:
        env.reset(seed=args.seed)
        raw = env.unwrapped
        # Stock code allocates dots_dist for only 500 control steps.
        # The current episode has just been initialized, so we resize now.
        if args.max_steps > 500:
            raw.dots_dist = torch.full((raw.num_envs, args.max_steps + 32),
                                       -1.0, device=raw.device)

        goal_p = raw.goal_outline.pose.p[0].detach().cpu().numpy().astype(np.float64)
        goal_q = raw.goal_outline.pose.q[0].detach().cpu().numpy().astype(np.float64)
        flat_path, to_world = stock_mapping(svg_d, goal_p, goal_q)

        expected_first = to_world(flat_path[0].start)
        actual_first = raw.points[0, 0].detach().cpu().numpy()
        map_error = np.linalg.norm(expected_first - actual_first)
        print(f"坐标变换校验误差: {map_error:.8f} m")
        if map_error > 1e-4:
            raise RuntimeError("SVG normalization/pose mismatch: inspect installed ManiSkill source")

        strokes = [np.asarray([to_world(p) for p in points]) for points in polylines]
        assert env.action_space.shape[-1] == 3, "panda_stick requires 3D XYZ actions"

        def tcp():
            return raw.agent.tcp.pose.p[0].detach().cpu().numpy().copy()

        def move_to(target, stroke_id, phase, max_iters=12, tol=0.004):
            nonlocal step_id
            target = np.asarray(target, dtype=np.float64)
            for _ in range(max_iters):
                now = tcp()
                error = target - now
                if np.linalg.norm(error) < tol:
                    break
                if step_id >= args.max_steps:
                    raise RuntimeError(
                        f"Exceeded --max-steps={args.max_steps}; "
                        "retry with more steps, or a larger --spacing"
                    )
                action = np.clip(error / 0.1, -0.30, 0.30).astype(np.float32)
                obs, reward, terminated, truncated, info = env.step(action)
                step_id += 1
                reached = tcp()
                rows.append([step_id, stroke_id, phase, *target, *reached, *action])
                if bool(truncated.any()):
                    raise RuntimeError("Episode truncated before finished drawing")
            return float(np.linalg.norm(target - tcp()))

        for sid, stroke in enumerate(strokes):
            print(f"开始第 {sid + 1}/{len(strokes)} 笔：{len(stroke)} 点")
            first = stroke[0].copy()
            above_first = first.copy()
            above_first[2] = LIFT_Z
            move_to(above_first, sid, "approach", max_iters=55, tol=0.006)
            first[2] = DRAW_Z
            move_to(first, sid, "pen_down", max_iters=30, tol=0.003)

            max_err = 0.0
            for point in stroke[1:]:
                goal = point.copy()
                goal[2] = DRAW_Z
                remaining = move_to(goal, sid, "draw", max_iters=5, tol=0.004)
                max_err = max(max_err, remaining)
            print(f"第 {sid + 1} 笔最大跟踪残差: {max_err*1000:.1f} mm")

            above_last = stroke[-1].copy()
            above_last[2] = LIFT_Z
            move_to(above_last, sid, "pen_up", max_iters=30, tol=0.006)

        print(f"绘画完成：{len(strokes)} 笔，{step_id} 控制步，输出 {output}")
    finally:
        with (output / "controller_steps.csv").open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                "step", "stroke", "phase", "target_x", "target_y", "target_z",
                "tcp_x", "tcp_y", "tcp_z", "action_x", "action_y", "action_z",
            ])
            writer.writerows(rows)
        env.close()
    print("结果：MP4、expert_demo.h5 / .json、controller_steps.csv、drawing_metadata.json")


if __name__ == "__main__":
    run()
