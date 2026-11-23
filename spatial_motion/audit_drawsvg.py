"""Read-only audit of existing DrawSVG recordings."""
import csv
import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np


def audit(path):
    with (path / "controller_steps.csv").open() as file:
        rows = list(csv.DictReader(file))
    action_csv = np.array([[row[k] for k in ("action_x", "action_y", "action_z")]
                           for row in rows], dtype=np.float32)
    tcp_csv = np.array([[row[k] for k in ("tcp_x", "tcp_y", "tcp_z")]
                        for row in rows], dtype=np.float32)
    meta = json.loads((path / "expert_demo.json").read_text())
    with h5py.File(path / "expert_demo.h5") as h5:
        episode = h5["traj_0"]
        actions = episode["actions"][:]
        tcp = episode["obs/extra/tcp_pose"][:]
        n = len(actions)
        assert n == len(rows) == meta["episodes"][0]["elapsed_steps"]
        assert [int(row["step"]) for row in rows] == list(range(1, n + 1))
        np.testing.assert_allclose(action_csv, actions, atol=1e-7)
        np.testing.assert_allclose(tcp_csv, tcp[1:, :3], atol=1e-6)
        assert episode["env_states/articulations/panda_stick"].shape[0] == n + 1
        images = {}
        for camera, entries in episode["obs/sensor_data"].items():
            rgb = entries["rgb"]
            assert rgb.shape[0] == n + 1 and rgb.dtype == np.uint8
            assert np.std(rgb[0]) > 0
            images[camera] = list(rgb.shape)
        assert images
        success = bool(np.asarray(episode["success"][-1]).item())
    info = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
        "-show_entries", "stream=nb_read_frames", "-of", "json", str(path / "0.mp4"),
    ]))
    frames = int(info["streams"][0]["nb_read_frames"])
    assert frames == n + 1
    return {"recording": path.name, "steps": n, "video_frames": frames,
            "rgb": images, "task_success": success}


if __name__ == "__main__":
    for directory in sys.argv[1:]:
        print(json.dumps(audit(Path(directory))), flush=True)
