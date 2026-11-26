"""Deterministic Cartesian grids from the 9,944-episode SpatialMotion plan."""
from dataclasses import dataclass
from itertools import product

import numpy as np

STEP_M = 0.0035
SPLITS = ("train", "validation", "test")
STARTS = np.asarray(list(product(
    (470, 488, 505, 522, 540), (-45, -22, 0, 22, 45), (240, 260, 280, 300)
)), dtype=np.float32) / 1000
START_INDICES = {"train": (0, 1, 2), "validation": (3,), "test": (0,)}
X, Y, Z = np.eye(3)
PLANES = {"XY": (X, Y), "XZ": (X, Z), "YZ": (Y, Z)}
for axis in ("X", "Y"):
    for angle in (-60, -30, 30, 60):
        c, s = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
        basis = (X, c * Y + s * Z) if axis == "X" else (c * X - s * Z, Y)
        PLANES[f"XY tilted {angle:+}° about base +{axis}"] = basis

HEADINGS = (0, 90, 180, 270)
SIGNS = (-1, 1)
RADIAL = (0, 90)
TURNS = (45, 90, 135)
FACES = ("same", "cross")
PRIMARY = {
    "line": ("length_cm", (10, 15, 20), 15, 12),
    "circle": ("diameter_cm", (10, 15, 20), 15, 12),
    "arc": ("diameter_cm", (16, 20), 20, 18),
    "line_line": ("length_cm", (10, 13), 13, 12),
    "line_circle": ("length_cm", (10, 13), 13, 12),
    "zigzag_circle": ("length_cm", (10, 13), 13, 12),
    "helix": ("diameter_cm", (10, 16), 16, 12),
}
VARIABLES = {
    "line": {"heading_deg": HEADINGS},
    "circle": {"rotation": SIGNS, "radial_deg": RADIAL},
    "arc": {"rotation": SIGNS, "radial_deg": RADIAL, "sweep_deg": (90, 120, 180)},
    "line_line": {"heading_deg": (0, 90), "second_length_cm": (6,),
                  "turn_deg": TURNS, "turn_sign": SIGNS, "face": FACES},
    "line_circle": {"heading_deg": HEADINGS, "face": FACES, "diameter_cm": (4, 6),
                    "rotation": SIGNS, "radial_deg": (0,)},
    "zigzag_circle": {"heading_deg": (0,), "second_length_cm": (3,), "turn_deg": TURNS,
                      "turn_sign": SIGNS, "face": FACES, "diameter_cm": (4,),
                      "rotation": SIGNS, "radial_deg": (0,)},
    "helix": {"rotation": SIGNS, "turns": (1, 2), "rise_cm": (4, 8), "radial_deg": (0,)},
    "circle_line": {"diameter_cm": (10, 14), "rotation": SIGNS, "length_cm": (6, 8),
                    "heading_deg": (0, 180), "face": ("same",), "radial_deg": (90,)},
}


@dataclass
class Motion:
    kind: str
    plane: str
    parameters: dict
    path: np.ndarray
    language: str
    segment_ends: list

    def metadata(self):
        return dict(kind=self.kind, plane=self.plane, parameters=self.parameters,
                    motion_samples=len(self.path) - 1, segment_ends=self.segment_ends,
                    language=self.language, frame="panda_base")


def steps(length):
    return np.minimum(np.arange(1, int(np.ceil(length / STEP_M - 1e-12)) + 1) * STEP_M / length, 1)


def axis_text(vector):
    for index, axis in enumerate((X, Y, Z)):
        for sign, prefix in ((1, "+"), (-1, "−")):
            if np.allclose(vector, sign * axis):
                return f"base {prefix}{'XYZ'[index]}"
    return "(" + ", ".join(f"{x:.6f}" for x in vector) + ") in Panda base XYZ"


def build_motion(kind, plane, p):
    u, v = PLANES[plane]
    normal = np.cross(u, v)
    heading = np.deg2rad(p.get("heading_deg", 0))
    forward = np.cos(heading) * u + np.sin(heading) * v
    tangent = np.cross(normal, forward)
    cross = p.get("face") == "cross"
    plane_text = f"a plane oriented like base {plane}"
    circle_u, circle_v = (tangent, normal) if cross else (u, v)
    circle_plane = (f"a plane with first axis {axis_text(circle_u)} "
                    f"and second axis {axis_text(circle_v)}") if cross else plane_text
    parts, clauses = [np.zeros((1, 3))], []

    def line(length, direction, instruction):
        parts.append(parts[-1][-1] + steps(length / 100)[:, None] * (length / 100) * direction)
        clauses.append(instruction)

    def curve():
        radius = p["diameter_cm"] / 200
        sweep = p.get("sweep_deg", 360 * p.get("turns", 1))
        rise = p.get("rise_cm", 0) / 100
        phase = np.deg2rad(p["radial_deg"])
        extent = np.deg2rad(sweep)
        t = steps(np.hypot(radius * extent, rise))
        angle = phase + p["rotation"] * extent * t
        points = (radius * (np.cos(angle) - np.cos(phase))[:, None] * circle_u
                  + radius * (np.sin(angle) - np.sin(phase))[:, None] * circle_v
                  + rise * t[:, None] * normal)
        parts.append(parts[-1][-1] + points)
        direction = "counterclockwise" if p["rotation"] > 0 else "clockwise"
        diameter = p["diameter_cm"]
        if kind == "arc":
            motion = f"trace a {sweep}° {direction} arc on a circle {diameter} cm in diameter"
        elif kind == "helix":
            turns = "one" if p["turns"] == 1 else "two"
            unit = "turn" if p["turns"] == 1 else "turns"
            motion = f"make {turns} {direction} {unit} {diameter} cm in diameter"
        else:
            motion = f"trace one {direction} circle {diameter} cm in diameter"
        instruction = (f"{motion} in {circle_plane}, starting at radial angle {p['radial_deg']}°")
        if rise:
            instruction += f", while advancing {p['rise_cm']} cm along its positive normal"
        clauses.append(instruction)

    if kind in ("line", "line_line", "line_circle", "zigzag_circle"):
        line(p["length_cm"], forward,
             f"move {p['length_cm']} cm along the {p['heading_deg']}° heading in {plane_text}")
    if kind in ("line_line", "zigzag_circle"):
        toward = p["turn_sign"] * (normal if cross and kind == "line_line" else tangent)
        turn = np.deg2rad(p["turn_deg"])
        line(p["second_length_cm"], np.cos(turn) * forward + np.sin(turn) * toward,
             f"turn {p['turn_deg']}° toward {axis_text(toward)} and move {p['second_length_cm']} cm")
    if kind in ("circle", "arc", "line_circle", "zigzag_circle", "helix", "circle_line"):
        curve()
    if kind == "circle_line":
        line(p["length_cm"], forward, f"move {p['length_cm']} cm along the "
             f"{p['heading_deg']}° heading in {plane_text}")

    language = "Starting at the current TCP position, " + ". Then ".join(clauses) + "."
    return Motion(kind, plane, p, np.concatenate(parts).astype(np.float32),
                  language, np.cumsum([len(part) for part in parts[1:]]).tolist())


def motions(split):
    for kind, parameters in VARIABLES.items():
        if kind == "circle_line":
            if split != "test":
                continue
            grid = parameters
        else:
            key, train, valid, test = PRIMARY[kind]
            grid = {key: {"train": train, "validation": (valid,), "test": (test,)}[split],
                    **parameters}
        for plane, values in product(PLANES, product(*grid.values())):
            yield build_motion(kind, plane, dict(zip(grid, values)))


def pilot_plan(plan):
    """Two cases per action, with both face relations for the three compositions."""
    pilot = {split: [] for split in SPLITS}
    for number, kind in enumerate(VARIABLES):
        splits = ("test", "test") if kind == "circle_line" else (
            "train", "validation" if number % 2 == 0 else "test")
        for index, split in enumerate(splits):
            plane = "XY" if index == 0 else "XY tilted +30° about base +X"
            face = FACES[index] if kind in ("line_line", "line_circle", "zigzag_circle") else "same"
            case, motion = next((case, motion) for case, motion in plan[split]
                                if motion.kind == kind and motion.plane == plane
                                and motion.parameters.get("face", "same") == face
                                and motion.parameters.get("rotation", SIGNS[index]) == SIGNS[index])
            pilot[split].append((case, motion))
    return pilot
