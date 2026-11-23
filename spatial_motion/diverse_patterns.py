"""Deterministic 3-D motion patterns for the diverse-v1 corpus."""
import numpy as np

VERSION = "diverse-v1"
BASE = ("line", "diagonal", "circle", "ellipse", "helix", "zigzag", "arc", "figure_eight")
COMPOUNDS = {
    "line_circle": ("sequential", ("line", "circle")),
    "circle_line": ("sequential", ("circle", "line")),
    "circle_helix": ("sequential", ("circle", "helix")),
    "helix_circle": ("sequential", ("helix", "circle")),
    "zigzag_arc": ("sequential", ("zigzag", "arc")),
    "arc_line_arc": ("sequential", ("arc", "line", "arc")),
    "figure_eight_lift": ("synchronous", ("figure_eight", "lift")),
    "orbit_slide": ("synchronous", ("circle", "line")),
    "ellipse_wave": ("offset", ("ellipse", "wave")),
    "arc_helix_overlap": ("offset", ("arc", "helix")),
}
PLANES = {
    "XY": ((1, 0, 0), (0, 1, 0)),
    "XZ": ((1, 0, 0), (0, 0, 1)),
    "YZ": ((0, 1, 0), (0, 0, 1)),
    "XY tilted +30 deg about X": ((1, 0, 0), (0, np.sqrt(3) / 2, 0.5)),
    "XY tilted -30 deg about X": ((1, 0, 0), (0, np.sqrt(3) / 2, -0.5)),
}


def parameters(kind, plane, radius, direction, rng):
    return {
        "kind": kind, "plane": plane, "radius_or_scale_mm": radius,
        "direction": direction, "ellipse_ratio": int(rng.integers(52, 86)) / 100,
        "arc_degrees": int(rng.choice([90, 120, 180])),
        "turns": int(rng.integers(1, 3)),
        "rise_mm": int(rng.integers(18, 47)),
        "axis_quarter_turns": int(rng.integers(0, 4)),
    }


def component(t, item):
    """Return a displacement from zero in millimetres and its instruction."""
    r = item["radius_or_scale_mm"]
    side = item["direction"]
    kind = item["kind"]
    u, v = np.array(PLANES[item["plane"]], dtype=float)
    turn = item["axis_quarter_turns"]
    axis_u, axis_v = ((u, v), (v, -u), (-u, -v), (-v, u))[turn]
    normal = np.cross(axis_u, axis_v)
    z = np.zeros_like(t)
    winding = "counterclockwise" if side == 1 else "clockwise"
    basis = f"in the base {item['plane']} plane, with local U at {turn * 90} degrees from its first axis"
    if kind in ("line", "diagonal", "lift"):
        if kind == "lift":
            delta = np.array([0., 0., side * r])
        else:
            delta = side * r * (axis_u if kind == "line" else axis_u + 0.55 * axis_v + 0.55 * normal)
        vector = ", ".join(f"{x:g}" for x in np.round(delta, 2))
        return t[:, None] * delta, f"move in a straight line by ({vector}) mm in base XYZ"
    if kind == "zigzag":
        cross = np.interp(t, (0, 1/3, 2/3, 1), (0, 0.6 * side, -0.6 * side, 0))
        x, y, description = r * t, r * cross, f"trace a {r:g} mm three-leg zigzag, first turning toward {'positive' if side == 1 else 'negative'} local V"
    elif kind == "wave":
        x, y, description = 1.4 * r * t, side * 0.35 * r * np.sin(2*np.pi*t), f"trace one sinusoidal wave of length {1.4*r:g} mm and amplitude {0.35*r:g} mm, first swaying toward {'positive' if side == 1 else 'negative'} local V"
    elif kind == "figure_eight":
        x = r * np.sin(2*np.pi*t)
        y = side * r * 0.5 * np.sin(4*np.pi*t)
        description = f"trace a figure eight spanning {2*r:g} mm along local U, first moving toward {'positive' if side == 1 else 'negative'} local V"
    else:
        degrees = item["arc_degrees"] if kind == "arc" else 360 * (item["turns"] if kind == "helix" else 1)
        angle = side * np.deg2rad(degrees) * t
        minor = r * item["ellipse_ratio"] if kind == "ellipse" else r
        x, y = r * (np.cos(angle) - 1), minor * np.sin(angle)
        if kind == "ellipse":
            description = f"trace an ellipse with semiaxes {r:g} and {minor:g} mm"
        elif kind == "helix":
            z = item["rise_mm"] * t
            description = f"trace a {item['turns']}-turn helix of radius {r:g} mm and rise {item['rise_mm']} mm along the local plane normal"
        elif kind == "arc":
            description = f"trace a {degrees}-degree arc of radius {r:g} mm"
        else:
            description = f"trace a full circle of radius {r:g} mm"
        description += f" {winding} viewed from the positive plane normal, starting on the positive local U edge"
    return (x[:, None] * axis_u + y[:, None] * axis_v + z[:, None] * normal,
            f"{description} {basis}")


def phase(t, start, stop, style):
    u = np.clip((t - start) / (stop - start), 0, 1)
    return u if style == "uniform" else u*u*(3 - 2*u)


def describe_motion(t, components, schedule, style):
    offset = np.zeros((len(t), 3))
    descriptions = []
    for item, (start, stop) in zip(components, schedule):
        displacement, label = component(phase(t, start, stop, style), item)
        offset += displacement
        descriptions.append(f"{label} from {start:.0%} to {stop:.0%} of the motion")
    return offset, descriptions


def schedule_for(mode, components, rng):
    if mode == "sequential":
        sample = np.linspace(0, 1, 101)
        # Reserve more time for components with a longer or faster local path.
        local_peak = [
            np.linalg.norm(np.diff(component(sample, item)[0], axis=0), axis=1).max()
            for item in components
        ]
        portions = np.asarray(local_peak) * rng.uniform(0.88, 1.12, size=len(components))
        boundaries = np.round(np.r_[0, np.cumsum(portions) / portions.sum()], 2)
        return list(zip(boundaries[:-1].tolist(), boundaries[1:].tolist()))
    if mode == "offset":
        return [(0, 1), (int(rng.integers(18, 34)) / 100, int(rng.integers(78, 96)) / 100)]
    return [(0, 1)] * len(components)


def examples(split, count, steps, rng):
    entries = []
    scales = np.arange(18 if split == "train" else 19, 43, 2)
    for kind in (*BASE, *COMPOUNDS):
        mode, kinds = COMPOUNDS.get(kind, ("single", (kind,)))
        for variant in range(count):
            size = int(scales[variant % len(scales)])
            primary_plane = tuple(PLANES)[variant % len(PLANES)]
            side = 1 if variant % 2 == 0 else -1
            amplitudes = [1.0] + [int(rng.integers(40, 76))/100 for _ in kinds[1:]]
            components = [
                parameters(part, primary_plane if index == 0 else str(rng.choice(tuple(PLANES))),
                           round(size * scale, 2), side if index == 0 else int(rng.choice([-1, 1])), rng)
                for index, (part, scale) in enumerate(zip(kinds, amplitudes))
            ]
            schedule = schedule_for(mode, components, rng)
            style = "uniform" if variant % 3 == 0 else "ease_in_out"
            length = int(rng.integers(round(steps * 0.64), round(steps * 0.94) + 1))
            while True:
                points, labels = describe_motion(np.linspace(0, 1, length), components, schedule, style)
                peak = float(np.linalg.norm(np.diff(points, axis=0), axis=1).max())
                if peak <= 3.8:
                    break
                if length == steps:
                    raise ValueError(f"{kind} needs more than {steps} steps for the speed limit")
                length = min(steps, max(length + 1, int(np.ceil(length * peak / 3.8))))
            offset = rng.integers([-144, -48, 71], [-69, 49, 135]).tolist()
            verb = "Add these displacements in parallel: " if mode in ("synchronous", "offset") else ""
            joiner = "; then " if mode == "sequential" else "; and "
            language = (f"Starting at the current tool position, {verb}{joiner.join(labels)}. "
                        f"Complete the motion in {(length-1)/20:.2f} seconds with "
                        f"{'uniform phase progression' if style == 'uniform' else 'easing at the start and end of each component'}; then hold the endpoint.")
            metadata = {
                "id": f"{VERSION}-{split}-{kind}-{variant:04d}", "split": split,
                "kind": kind, "segments": list(kinds), "size_mm": size, "variant": variant,
                "language": language, "frame": "panda_base", "pattern_version": VERSION,
                "start_offset_mm": offset, "composition": mode, "phase_profile": style,
                "components": components, "relative_amplitudes": amplitudes,
                "component_windows": schedule, "motion_samples": length,
                "motion_duration_s": (length-1)/20, "peak_target_speed_mm_s": peak * 20,
                "mean_target_speed_mm_s": float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum() * 20/(length-1)),
            }
            points = np.pad(points, ((0, steps-length), (0, 0)), mode="edge")
            entries.append((points.astype(np.float32) / 1000, metadata))
    return [entries[i] for i in rng.permutation(len(entries))]

