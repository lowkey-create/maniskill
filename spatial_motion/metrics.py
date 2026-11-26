"""Geometric measurements for line and full-circle rollouts in Panda base coordinates."""
import numpy as np

from .plan import PLANES


def angle_degrees(a, b):
    return float(np.rad2deg(np.arccos(np.clip(np.dot(a, b) / np.linalg.norm(a)
                                             / np.linalg.norm(b), -1, 1))))


def measure(trajectory, target, actual):
    params = trajectory["parameters"]
    if trajectory["kind"] == "line":
        desired = target[-1] - target[0]
        measured = actual[-1] - actual[0]
        length_error = float(100 * abs(np.linalg.norm(measured) - np.linalg.norm(desired)))
        heading_error = angle_degrees(desired, measured)
        return dict(length_error_cm=length_error, heading_error_deg=heading_error,
                    passed=length_error <= 1 and heading_error <= 10)

    u, v = PLANES[trajectory["plane"]]
    normal = np.cross(u, v)
    centered = actual - actual.mean(axis=0)
    _, _, basis = np.linalg.svd(centered, full_matrices=False)
    fit_u, fit_n = basis[0], basis[2]
    if np.dot(fit_n, normal) < 0:
        fit_n = -fit_n
    fit_v = np.cross(fit_n, fit_u)
    plane_error = angle_degrees(normal, fit_n)
    x, y = centered @ fit_u, centered @ fit_v
    cx, cy, intercept = np.linalg.lstsq(
        np.column_stack((2 * x, 2 * y, np.ones_like(x))), x * x + y * y, rcond=None
    )[0]
    radius = np.sqrt(intercept + cx**2 + cy**2)
    angles = np.unwrap(np.arctan2(y - cy, x - cx))
    sweep = float(np.rad2deg(angles[-1] - angles[0]))
    sweep_error = abs(abs(sweep) - 360)
    direction = bool(params["rotation"] * sweep > 0)
    closing_error = float(100 * np.linalg.norm(actual[-1] - actual[0]))
    diameter_error = float(abs(200 * radius - params["diameter_cm"]))
    return dict(diameter_error_cm=diameter_error, closure_error_cm=closing_error,
                plane_error_deg=plane_error, sweep_error_deg=sweep_error,
                rotation_correct=direction, passed=direction and diameter_error <= 1
                and closing_error <= 1 and plane_error <= 10 and sweep_error <= 10)
