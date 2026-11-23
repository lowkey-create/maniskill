# SpatialMotion-v1

ManiSkill 3.0.1, Panda, 20 Hz control. Import `spatial_motion` before calling `gym.make("SpatialMotion-v1")`. The default controller is Panda's absolute `pd_ee_pose`. The collector sends each Panda-base XYZ waypoint in metres, holds the reset orientation using XYZ Euler angles in radians, and keeps the gripper open. ManiSkill computes pose error and joint targets. Each waypoint occupies one control step, followed by a configurable endpoint hold.

New recordings store seven action values `[x_m, y_m, z_m, rx_rad, ry_rad, rz_rad, gripper]`. Old four-value `pd_ee_delta_pos` recordings remain readable by the verifier. For callers replaying legacy actions, select `control_mode="pd_ee_delta_pos"` explicitly. Use a fresh corpus directory when switching control modes.

Run from `maniskill_project/`. Use a fresh output directory for each run.

```bash
# CPU physics with state, action and tracking data
OMP_NUM_THREADS=1 .venv/bin/python -m spatial_motion.collect --output spatial_motion/run_state

# CPU physics and dual-view EGL software RGB
.venv/bin/python -m pip install --target spatial_motion/.render_deps moderngl==5.13.0 glcontext==3.1.0
OMP_NUM_THREADS=1 LP_NUM_THREADS=2 LIBGL_ALWAYS_SOFTWARE=1 \
  .venv/bin/python -m spatial_motion.collect --render software --output spatial_motion/run_rgb
.venv/bin/python -m spatial_motion.verify spatial_motion/run_rgb

# GPU PhysX and native SAPIEN RGB (requires a working CUDA and Vulkan device)
.venv/bin/python -m spatial_motion.collect --sim gpu --render gpu --num-envs 4 \
  --output spatial_motion/run_gpu
.venv/bin/python -m spatial_motion.verify spatial_motion/run_gpu

# Absolute Panda-base Cartesian references, one waypoint per control step
.venv/bin/python -m spatial_motion.collect --waypoints spatial_motion/custom_waypoints.npy \
  --output spatial_motion/run_custom
```

Waypoint files are NumPy `.npy` arrays shaped `(T, 3)` or `(T, num_envs, 3)`. A single sequence broadcasts to all environments. References should start near the reset TCP, use continuous waypoints at 20 Hz, and stay in the reachable workspace. The built-ins use 240 reference steps and 30 final-target hold steps. The default pass criterion is at most 10 mm on every reference step and at the end of the hold.

`expert.h5` contains actions, robot joint positions and velocities, TCP poses (xyz + wxyz), targets, signed target-minus-actual errors, and completion flags. State and camera arrays have T+1 entries. Action t controls the transition from state t to t+1, and each error uses the post-action state. The CSV logs one row per control step per environment. Camera-enabled runs save raw RGB, per-frame calibration and one MP4 per camera and environment. The first video frame is the reset observation. `metrics.json` reports per-environment errors and `summary.json` collects all episodes in the run.

`third_view` is fixed in world space, and `wrist` is mounted to `panda_hand`. The optional EGL path renders the real Panda visual meshes using the live ManiSkill link and camera poses. It uses its own lighting and rasterization and labels the backend in HDF5. Native SAPIEN RGB uses the same camera definitions.

Existing DrawSVG outputs can be checked without editing them:

```bash
.venv/bin/python -m spatial_motion.audit_drawsvg results/signature results/test results/flower results/svg_A
```

## Plot recorded TCP paths

Install matplotlib in the project environment if needed. The plot overlays the
target and measured TCP in the Panda base frame and shows per-step error.
The shaded interval marks the endpoint hold.

```bash
.venv/bin/python -m spatial_motion.plot_trajectories spatial_motion/run_rgb --output spatial_motion/run_rgb/trajectories.png
```

## Generate a language-conditioned corpus

On a CUDA and Vulkan host, run from the repository root:

```bash
OMP_NUM_THREADS=1 IMAGEIO_FFMPEG_EXE=/usr/bin/ffmpeg .venv/bin/python -m spatial_motion.generate_corpus --output spatial_motion/run_corpus
```

The defaults generate 256 train and 64 held-out-size test episodes across eight
families, including two sequential compositions. Both splits contain the same
families and use disjoint size grids. This measures parameter generalization.
Each trajectory uses 240 reference steps and 20 endpoint hold steps by default.
Each episode includes an English instruction with geometry and direction, and
keeps both RGB streams and the original observation/action alignment.

`manifest.json` indexes the split, instruction, parameters, HDF5 shard and
environment slot. Each shard also embeds its episode metadata. GPU batches are
verified before publication. Repeat the same command to resume from complete
shards. `--num-envs` sets the batch size and `--video-envs` sets the number of
preview videos per camera and shard. All environments retain raw RGB in HDF5.

Render an equal-scale overview of recorded and target paths by trajectory family:

```bash
.venv/bin/python -m spatial_motion.plot_corpus spatial_motion/run_corpus --output spatial_motion/run_corpus/trajectory_overview.png
```

### Diverse three-dimensional trajectories

Run from the repository root on a CUDA and Vulkan host:

```bash
OMP_NUM_THREADS=1 IMAGEIO_FFMPEG_EXE=/usr/bin/ffmpeg .venv/bin/python -m spatial_motion.generate_corpus \
  --pattern diverse-v1 --output spatial_motion/run_diverse_hz01/corpus \
  --num-envs 128 --train-per-kind 36 --test-per-kind 8
```

This configuration generates 792 episodes in 18 families. It includes eight
single motions and ten compositions using ordered segments, simultaneous vector
addition, or addition with a delayed start and early finish. Reversed orders
are available for line/circle and circle/helix. Component scales, phase windows,
direction, axes and plane are recorded in each HDF5 episode's metadata and the
corpus manifest. The primary scale uses an even-mm training grid and an odd-mm
test grid. The split measures generalization to held-out primary scales within
the same families.

A short approach moves the Panda to its sampled start before the first saved
observation. The recorded horizon is fixed across a batch but the time spent
moving varies. Use `motion_samples` to find the moving reference interval.
References then repeat the endpoint up to the fixed horizon and explicit hold.
The manifest includes the motion duration and commanded mean and peak speed.
