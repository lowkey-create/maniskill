# SpatialMotion-v1

ManiSkill 3.0.1, Panda, 20 Hz control. References and tracking errors are metres in the Panda base frame. Import `spatial_motion` before calling `gym.make("SpatialMotion-v1")`. The action is Panda's normalized `pd_ee_delta_pos` with `root_translation`; the first three components map to +/-0.1 m per axis and the fourth controls the gripper. The collector keeps the gripper open and clips its Cartesian command to +/-0.015 m per axis. It calculates feedback from the measured TCP at every step.

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
