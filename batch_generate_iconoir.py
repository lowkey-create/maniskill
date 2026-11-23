#!/usr/bin/env python3
"""Batch-generate ManiSkill DrawSVG expert demonstrations from an SVG directory.

Usage (from maniskill_project/):
  python batch_generate_iconoir.py --limit 10 --jobs 1
  python batch_generate_iconoir.py --jobs 2
  python batch_generate_iconoir.py --jobs 2 --spacing 0.012 --max-spacing 0.016 --target-points 650
  python batch_generate_iconoir.py --dry-run

Requires draw_svg_from_file_batch.py in the same directory. Uses the SAME
SVG parsing/coordinate conversion/controller as the user's original script.
"""

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path

from draw_svg_from_file_batch import DEFAULTS, read_svg_strokes, flatten_and_normalize


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "svg_bench" / "clean" / "iconoir"
DEFAULT_OUTPUT = SCRIPT_DIR / "results" / "iconoir_experts"
RUNNER = SCRIPT_DIR / "draw_svg_from_file_batch.py"
FIELDS = [
    "file", "status", "task_success", "waypoints", "strokes", "spacing_m",
    "max_steps", "actual_steps", "attempts", "seconds", "message", "output",
]


@dataclass
class Item:
    svg: Path
    output: Path
    points: int
    strokes: int
    spacing: float
    max_steps: int


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--jobs", type=int, default=2, help="并行的独立 ManiSkill 进程数（先尝试 1 或 2）")
    p.add_argument("--limit", type=int, default=0, help="只处理排序后的前 N 个文件；0=全部")
    p.add_argument("--dry-run", action="store_true", help="只分析 SVG 和规划参数，不启动仿真")
    p.add_argument("--spacing", type=float, default=0.010, help="初始目标点间距，米；更小更精细")
    p.add_argument("--max-spacing", type=float, default=0.016, help="复杂图标自动放大间距的上限（米）")
    p.add_argument("--target-points", type=int, default=850, help="复杂图标尝试将目标点数降至此值附近")
    p.add_argument("--no-adaptive", action="store_true", help="禁止为复杂 SVG 自动放宽采样间距")
    p.add_argument("--step-factor", type=float, default=2.3, help="每个目标点预估控制步数系数")
    p.add_argument("--stroke-overhead", type=int, default=28, help="每笔额外预留接近/落笔/抬笔动作")
    p.add_argument("--max-steps-cap", type=int, default=12000,
                   help="安全上限；预计超过该值的文件单独列为 over_cap；0=无限制")
    p.add_argument("--retries", type=int, default=1, help="只对步数用尽自动重试次数")
    p.add_argument("--retry-factor", type=float, default=1.65, help="步数用尽时预算扩大倍数")
    p.add_argument("--obs-mode", choices=("rgb", "state"), default="rgb",
                   help="rgb 可用于视觉 VLA 的 SFT；state 无图像但通常更快")
    p.add_argument("--with-video", action="store_true", default=True,help="额外录制每步 MP4（明显增加开销）")
    p.add_argument("--with-csv", action="store_true", help="额外保存逐控制步 CSV")
    p.add_argument("--draw-iters", type=int, default=DEFAULTS.draw_iters)
    p.add_argument("--draw-tol", type=float, default=DEFAULTS.draw_tol_m)
    p.add_argument("--force", action="store_true", help="即使目标目录已经有完整轨迹也重新生成")
    p.add_argument("--timeout", type=int, default=0, help="单次生成最长秒数；0=不限")
    args = p.parse_args()
    if args.jobs < 1 or args.limit < 0 or args.retries < 0 or args.timeout < 0:
        p.error("--jobs>=1, --limit>=0, --retries>=0, --timeout>=0")
    if not 0 < args.spacing <= args.max_spacing:
        p.error("需要 0 < --spacing <= --max-spacing")
    if args.target_points < 2 or args.step_factor <= 0 or args.stroke_overhead < 0:
        p.error("--target-points>=2, --step-factor>0, --stroke-overhead>=0")
    if args.max_steps_cap < 0 or args.retry_factor <= 1:
        p.error("--max-steps-cap>=0, --retry-factor>1")
    if args.draw_iters < 1 or args.draw_tol <= 0:
        p.error("--draw-iters>=1, --draw-tol>0")
    args.input = args.input.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    return args


def plan(svg: Path, out: Path, args):
    paths = read_svg_strokes(svg)
    spacing = args.spacing
    for _ in range(8):
        polylines, _ = flatten_and_normalize(paths, replace(DEFAULTS, spacing_m=spacing))
        count = sum(map(len, polylines))
        if args.no_adaptive or count <= args.target_points or spacing >= args.max_spacing:
            break
        # A curved SVG segment is sampled at least 4 times in the original code,
        # so this is a heuristic and cannot force an arbitrary target point count.
        new_spacing = max(spacing * 1.25, spacing * min(2.0, count / args.target_points))
        spacing = min(args.max_spacing, new_spacing)
    stroke_count = len(polylines)
    # Rounded to 64 to avoid overly precise estimates. MIN 128; actual capacity
    # inside DrawSVGEnv is at least 1000 as set by the drawing script.
    estimate = max(128, math.ceil(args.step_factor * count + args.stroke_overhead * stroke_count + 80))
    steps = int(math.ceil(estimate / 64) * 64)
    return Item(svg=svg, output=out, points=count, strokes=stroke_count,
                spacing=spacing, max_steps=steps)


def metadata_of(out: Path):
    try:
        return json.loads((out / "drawing_metadata.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def valid_h5(out: Path):
    path = out / "expert_demo.h5"
    if not path.is_file() or path.stat().st_size <= 1024 or not (out / "expert_demo.json").is_file():
        return False
    try:
        import h5py
        with h5py.File(path, "r") as f:
            return any(k.startswith("traj_") and "actions" in f[k] and
                       len(f[k]["actions"]) > 0 for k in f.keys())
    except (OSError, ImportError, KeyError, ValueError):
        return False


def row_for(item: Item, status: str, success=None, attempts=0, elapsed=0.,
            actual_steps=None, message=""):
    return dict(file=str(item.svg), status=status, task_success=success,
                waypoints=item.points, strokes=item.strokes,
                spacing_m=f"{item.spacing:.6f}", max_steps=item.max_steps,
                actual_steps=actual_steps, attempts=attempts,
                seconds=round(elapsed, 1), message=message, output=str(item.output))


def generate(item: Item, args):
    item.output.mkdir(parents=True, exist_ok=True)
    logfile = item.output / "generation.log"
    start = time.monotonic()
    tries = 0
    max_steps = item.max_steps
    reason = ""
    status = "failed"
    while tries <= args.retries:
        if args.max_steps_cap and max_steps > args.max_steps_cap:
            status, reason = "over_cap", f"预算 {max_steps} 超出 --max-steps-cap {args.max_steps_cap}"
            break
        tries += 1
        cmd = [sys.executable, "-u", str(RUNNER), str(item.svg),
               "--output", str(item.output), "--spacing", str(item.spacing),
               "--max-steps", str(max_steps), "--obs-mode", args.obs_mode,
               "--draw-iters", str(args.draw_iters), "--draw-tol", str(args.draw_tol)]
        if not args.with_video:
            cmd.append("--no-video")
        if not args.with_csv:
            cmd.append("--no-csv")
        env = os.environ.copy()
        # Avoid several subprocesses multiplying BLAS/OpenMP thread pools.
        env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        env.setdefault("PYTHONUNBUFFERED", "1")
        attempt_start_byte = logfile.stat().st_size if logfile.exists() else 0
        try:
            with logfile.open("a", encoding="utf-8") as f:
                f.write(f"\n===== Attempt {tries} | steps={max_steps} | {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
                f.flush()
                proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                                      env=env, cwd=SCRIPT_DIR,
                                      timeout=(args.timeout or None), check=False)
            ret = proc.returncode
            reason = f"exit={ret}"
        except subprocess.TimeoutExpired:
            ret = 124
            reason = f"timeout>{args.timeout}s"
        except OSError as e:
            ret = 127
            reason = f"process error: {e}"
        md = metadata_of(item.output)
        if ret == 0 and md.get("run_status") == "completed" and valid_h5(item.output):
            status = "completed"
            break
        # Retry only when the actual control step limit was reached.
        try:
            with logfile.open("rb") as log_in:
                log_in.seek(attempt_start_byte)
                tail = log_in.read()[-5000:].decode("utf-8", errors="replace")
        except OSError:
            tail = ""
        budget_hit = ("已到达脚本步数上限" in tail or "Gym Episode 被截断" in tail)
        if budget_hit and tries <= args.retries:
            next_steps = int(math.ceil(max_steps * args.retry_factor / 64) * 64)
            max_steps = max(next_steps, max_steps + 64)
            status = "retrying"
            continue
        status = "failed"
        reason += "; " + ("step budget hit" if budget_hit else "see generation.log")
        break
    md = metadata_of(item.output)
    final = row_for(item, status, md.get("task_success") if status == "completed" else None,
                    tries, time.monotonic() - start, md.get("actual_control_steps"), reason)
    final["max_steps"] = max_steps
    return final


def main():
    args = parse_args()
    if not RUNNER.is_file():
        sys.exit(f"找不到执行器: {RUNNER}")
    if not args.input.is_dir():
        sys.exit(f"找不到 SVG 目录: {args.input}")
    svgs = sorted(p for p in args.input.rglob("*") if p.is_file() and p.suffix.lower() == ".svg")
    if args.limit:
        svgs = svgs[:args.limit]
    if not svgs:
        sys.exit("没有找到 SVG 文件")
    print(f"输入 {args.input}\n输出 {args.output}\nSVG {len(svgs)} 张 | jobs={args.jobs} | obs={args.obs_mode}", flush=True)
    if not args.dry_run:
        args.output.mkdir(parents=True, exist_ok=True)
    results, queued = [], []
    for n, svg in enumerate(svgs, 1):
        out = args.output / svg.relative_to(args.input).with_suffix("")
        try:
            item = plan(svg, out, args)
        except Exception as e:
            r = dict.fromkeys(FIELDS, "")
            r.update(file=str(svg), status="parse_error", message=str(e), output=str(out))
            results.append(r)
            print(f"[解析失败 {n}/{len(svgs)}] {svg.name}: {e}", flush=True)
            continue
        if not args.force and metadata_of(out).get("run_status") == "completed" and valid_h5(out):
            md = metadata_of(out)
            r = row_for(item, "skipped_existing", md.get("task_success"),
                        actual_steps=md.get("actual_control_steps"))
            results.append(r)
            continue
        if args.max_steps_cap and item.max_steps > args.max_steps_cap:
            r = row_for(item, "over_cap", message=f"预计 {item.max_steps} 步超过安全上限 {args.max_steps_cap}")
            results.append(r)
            print(f"[超出容量] {svg.name}: {item.points} 点，预计 {item.max_steps} 步", flush=True)
            continue
        queued.append(item)
    if args.dry_run:
        print(f"规划完成: 可执行 {len(queued)}，已有 {sum(x['status']=='skipped_existing' for x in results)}，无法直接执行 {len(results)-sum(x['status']=='skipped_existing' for x in results)}")
        for i in queued[:20]:
            print(f"  {i.svg.name}: strokes={i.strokes}, points={i.points}, spacing={i.spacing:.4f}m, max_steps={i.max_steps}")
        if len(queued)>20:
            print(f"  ...另外 {len(queued)-20} 张")
        return
    summary_path = args.output / "batch_results.csv"
    with summary_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for r in results:
            writer.writerow(r)
        f.flush()
        print(f"待运行 {len(queued)}，已完成跳过 {sum(x['status']=='skipped_existing' for x in results)}", flush=True)
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            futures = {executor.submit(generate, item, args): item for item in queued}
            for idx, future in enumerate(as_completed(futures), 1):
                item = futures[future]
                try:
                    result = future.result()
                except Exception as e:
                    result = row_for(item, "worker_error", message=repr(e))
                writer.writerow(result)
                f.flush()  # Interrupted batches retain processed item statuses.
                results.append(result)
                print(f"[{idx}/{len(queued)}] {result['status']:<16} "
                      f"success={result['task_success']!s:<5} "
                      f"{item.svg.name}  steps={result['actual_steps']} "
                      f"t={result['seconds']}s", flush=True)
    states = [r["status"] for r in results]
    success = sum(r.get("task_success") is True for r in results)
    false = sum(r.get("task_success") is False for r in results)
    print(f"\n批量完成: 总数={len(results)} | 新生成={states.count('completed')} | 已有={states.count('skipped_existing')} "
          f"| 官方成功={success} | 官方不成功={false} | 其他={len(results)-states.count('completed')-states.count('skipped_existing')}")
    print(f"汇总: {summary_path}\n失败时查看: <SVG对应输出目录>/generation.log")


if __name__ == "__main__":
    main()
