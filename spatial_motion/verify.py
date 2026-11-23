"""Check recorded state/action alignment and decode camera videos."""
import argparse
import json
import subprocess
from pathlib import Path

import h5py
import numpy as np


def verify(directory):
    metrics = json.loads((directory / "metrics.json").read_text())
    with h5py.File(directory / "expert.h5") as h5:
        n, b, d = h5["action"].shape
        assert bool(h5.attrs["complete"]) and d == 4
        assert h5["state/qpos"].shape == h5["state/qvel"].shape == (n + 1, b, 9)
        assert h5["state/tcp_pose_base"].shape == (n + 1, b, 7)
        assert not h5["terminated"][:].any() and not h5["truncated"][:].any()
        target = h5["target_xyz_base_m"][:]
        tcp = h5["state/tcp_pose_base"][1:, :, :3]
        error = target - tcp
        norm = np.linalg.norm(error, axis=-1)
        np.testing.assert_allclose(h5["error_xyz_base_m"][:], error)
        np.testing.assert_allclose(h5["error_norm_m"][:], norm)
        action = h5["action"][:]
        for key in ("state/qpos", "state/qvel", "state/tcp_pose_base", "action"):
            assert np.isfinite(h5[key][:]).all(), key
        with (directory / "tracking.csv").open() as file:
            rows = np.loadtxt(file, delimiter=",", skiprows=1, ndmin=2)
        assert rows.shape == (n * b, 17)
        np.testing.assert_allclose(rows[:, 0], np.repeat(np.arange(n), b))
        np.testing.assert_allclose(rows[:, 1], np.tile(np.arange(b), n))
        np.testing.assert_allclose(rows[:, 3:], np.concatenate(
            (target, tcp, error, norm[..., None], action), axis=-1).reshape(n * b, 14))
        videos = {}
        if metrics["render"] != "none":
            assert set(h5["rgb"]) == {"third_view", "wrist"}
            for camera, frames in h5["rgb"].items():
                assert frames.shape == (n + 1, b, 192, 256, 3)
                assert frames.dtype == np.uint8
                for slot in range(b):
                    assert np.std(frames[0, slot]) > 0
                    assert np.any(frames[0, slot] != frames[n // 3, slot])
                    if slot >= h5.attrs.get("video_envs", b):
                        continue
                    path = directory / f"{camera}_env{slot:03d}.mp4"
                    probe = json.loads(subprocess.check_output([
                        "ffprobe", "-v", "error", "-select_streams", "v:0",
                        "-count_frames", "-show_streams", "-of", "json", str(path),
                    ]))["streams"][0]
                    assert int(probe["nb_read_frames"]) == n + 1
                    assert (probe["height"], probe["width"]) == (192, 256)
                    videos[path.name] = int(probe["nb_read_frames"])
    return {"trajectory": directory.name, "transitions": n, "envs": b, "videos": videos}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    directory = parser.parse_args().run
    for episode in json.loads((directory / "summary.json").read_text()):
        print(json.dumps(verify(directory / episode["trajectory"])))
