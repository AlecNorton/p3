# P3 workflow

How the P3 pieces fit together, from setup to a flight in the lab. The
assignment page is the specification; this is a map of the code.

```
Vicon pose ──► registration ──► splat renderer ──► your policy ──► Crazyflie
  (metres)     (vicon_transform)   (RGB + depth)    (velocity)      command
```

The drone, its latency and its tracking are real. What the camera sees is
rendered from a Gaussian splat of the room. The simulator (`Code/Wrapper.py`)
and the flight (`splat_hitl.ros_node`) run the same `Runtime`, contract and
scoring, so a policy that flies in one sees identical observations in the other.

## The folders

| Folder | What it is |
|---|---|
| `~/rbe595/splat_hitl` | The library: scene loading, contract, runtime, renderer client, ROS node. Must be on the `dev` branch; `main` cannot load the P3 pack. |
| `~/rbe595/p3` | The starter pack, and our repo. `Code/` is ours to edit; `tools/` is the course's, not submitted. |
| `~/rbe595/p3/scene` | The training course `a3_train`: three windows, stored in Vicon metres. |
| `~/rbe595/cf_vicon_stack` | The lab Docker stack: Vicon bridge, Crazyflie driver, ROS 2. |

The pack's README assumes a separate `p3_starter` folder next to your own. Ours
is one folder, `p3`, so pass `--pack .` to `Wrapper.py` and use paths relative
to `p3` in the `ros_node` command (both are shown below).

### What's in `scene/`

| File | Purpose |
|---|---|
| `a3_train_cleaned_splat.splat` | The Gaussian splat the camera sees (about 950,000 Gaussians, 30 MB). |
| `a3_train_esdf.npy` | Distance field: clearance at any point, used to score collisions. |
| `gates.json` | The windows in order. For scoring only: the policy must not read it. |
| `task.json` | The start pair `train_1`: (0, 0, 0.7) m, facing +x. |
| `policy_contract.json` | The scene's camera; `make_contract.py` builds ours from it. |
| `vicon_transform.json` | Vicon metres → splat units (only a scale here). |

### What's in `Code/`

| File | Purpose |
|---|---|
| `Wrapper.py` | Flies a policy in the simulator and scores it. The file the course runs. |
| `policy_template.py` | `WindowRacer` (our policy, with TODOs) and `HoverHere` (zero velocity, for checking the loop). |
| `fpv.py` | Records what the camera saw, with detections drawn on, into `fpv/<run>/`. |
| `make_coco.py` | Builds a COCO segmentation file from synthetic images and masks. |
| `train_yolo26_seg.py` | Trains a YOLO26 segmentation model from that COCO file. |
| `best.pt` | Our trained detector weights (committed as "Added detection model"). |

### What's in `tools/`

| File | Purpose |
|---|---|
| `make_contract.py` | Writes `p3_contract.json`: our camera resolution and action type. |
| `check_setup.py` | Checks the setup step by step and stops at the first failure. |
| `render_views.py` | Saves camera images at chosen poses, to test the detector. |
| `cpu_renderer.py`, `splat_cpu.py` | A numpy-only splat renderer, used by `--cpu`. |

## 1. Install

```bash
cd ~/rbe595
git clone -b dev https://github.com/pearwpi/splat_hitl.git
pip install -e splat_hitl
pip install matplotlib        # optional: without it, Wrapper.py skips run.png
```

`splat_hitl` needs only numpy, and needs no ROS, GPU or drone. ROS is only used
by `ros_node`, which runs in the lab container. Run the library's tests with
`python3 -m pytest -q` inside `splat_hitl`.

GPU rendering needs a separate Python environment with PyTorch and gsplat
(`splat_hitl/worker/README.md`). `install.md` gives the commands for Turing
(CUDA 12.4). An RTX 50-series laptop GPU likely needs a PyTorch build for CUDA
12.8 or later, and possibly gsplat built from source; this hasn't been tried.

## 2. Set up the pack

From `~/rbe595/p3`:

```bash
python3 tools/make_contract.py            # writes p3_contract.json (320 x 213)
python3 tools/check_setup.py              # stops at the first problem
python3 tools/check_setup.py --render     # also renders the start view on the CPU
```

`make_contract.py --width 480` sets another resolution; the height follows the
3:2 shape. The contract keeps the scene's camera (60.66° horizontal field of
view, RGB plus depth, 15 Hz) and sets the action to a body-frame velocity with
free yaw.

`check_setup.py` checks, in order:

1. numpy imports.
2. `splat_hitl` imports.
3. The scene loads.
4. The scene passes its self-check.
5. `p3_contract.json` matches what `make_contract.py` writes.
6. The start position is inside the mapped volume.
7. `WindowRacer` runs on one fake frame and returns a velocity action.
8. With `--render`, the start view is written to `check_view.png`.

To check only the scene:

```bash
python3 -m splat_hitl.bundle scene         # clearance defaults to 0.10 m
python3 -m splat_hitl.bundle scene 0.05    # the clearance P3 scores with
```

It prints the scene's name, its gates and the collision map's size, and exits
with 0 if the scene passes or 1 if it fails. Lines marked `note` are not errors.

## 3. Train the detector

Train only on synthetic images we make ourselves (Blender or similar). Anything
rendered from the splat (`Wrapper.py`, `render_views.py`, FPV frames) is for
testing only. The window print is `textures/Window.png`, at the size in
`textures/Windows.pptx`.

1. **Render** images and one black-and-white mask per window per frame:
   `images/Image0001.png`, `masks/mask_1_0001.png`, `masks/mask_2_0001.png`, …
   An all-black mask means that window is not in the frame.
2. **Build the COCO file** (needs `opencv-python`):
   ```bash
   python3 Code/make_coco.py --images_dir images --masks_dir masks --output annotations.json
   ```
   All windows are one category.
3. **Train** (needs `ultralytics` and `pyyaml`):
   ```bash
   python3 Code/train_yolo26_seg.py --data-dir <folder> --imgsz 320 --device 0
   ```
   The folder holds the images and one COCO JSON.
4. **Export for flight.** The lab container has numpy but no deep-learning
   library. The pack suggests exporting to ONNX and running it with ONNX
   Runtime. Add it to the container image in `cf_vicon_stack`:
   ```bash
   PIP_EXTRA="onnxruntime" docker compose -f docker/docker-compose.yml build
   ```
   Then check inside the container:
   `python3 -c "import onnxruntime, numpy, rclpy, geometry_msgs.msg; print('ok')"`.
   The pack doesn't include an export command. With ultralytics,
   `yolo export model=Code/best.pt format=onnx` is the usual one.

## 4. Write the policy

Everything goes in `WindowRacer` in `Code/policy_template.py`. `act(obs, state)`
is called 15 times a second.

**Inputs**

| Field | Meaning |
|---|---|
| `obs.rgb` | (H, W, 3) float32 in [0, 1] |
| `obs.depth_m` | (H, W) float32, metres along the camera axis; 4.0 where nothing was hit |
| `state.position_m`, `state.velocity_ms` | Scene metres (our scene is stored in Vicon metres) and m/s |
| `state.yaw_rad` | Heading, counter-clockwise from +x |
| `state.t_s` | Seconds since the run started |

**Output:** `Action("velocity", "body_flu", [vx, vy, vz], yaw_rate_rad_s=w)`, in
m/s in the body frame (x forward, y left, z up). The camera sits at the drone's
centre and looks along body +x.

**The TODOs**

| Method | What to fill in |
|---|---|
| `__init__` | Load the network once, by a path built from `HERE`. Never load it in `act()`. |
| `detect` | Run the network and return `{"mask": ..., "corners": ...}` or `None`. |
| `locate` | Back-project the window's pixels with depth (`pixel_to_camera`, then `camera_to_world`) and find the opening's centre. Fit a plane to get the direction it faces. |
| `velocity_to` | The P2b position controller: world target → world velocity. |
| `act` | The state machine below. |

**State machine:** `search` (turn until a window is in view) → `align` (face it,
move in front of the opening) → `approach` (fly at the opening, keeping it
centred) → `through` (commit and fly straight until past it) → next window, or
`done` (hover) after three. Count windows in `self.passed`.

**Rules**
- Window positions must come from the image. Don't read `gates.json`,
  `state.gate_index`, `state.gate_distance_m` or `state.gate_bearing_rad`.
  `Wrapper.py` blanks the gate fields anyway.
- Rendering, the detector and `act()` share one 66 ms frame on the lab PC's CPU.
  `Wrapper.py` reports how long `act()` took.
- Once the network is fixed at one resolution, set `sensor_fingerprint` to
  `PolicyContract.load("p3_contract.json").observation.sensor.fingerprint()`.
  The runtime then refuses to fly it through any other camera.
- Use relative paths: the course runs `Wrapper.py` on its own machine, against a
  test scene.

To test the detector on splat views:

```bash
python3 tools/render_views.py --cpu                          # six views along the course
python3 tools/render_views.py --cpu --pose 1.0 0.2 0.8 -20   # x y z in Vicon metres, yaw in degrees
```

Each view writes `view_NN.png`, `view_NN_depth.png` and `view_NN_depth.npy` into
`views/`.

## 5. Fly it in the simulator

From `~/rbe595/p3`, pick a renderer:

```bash
python3 Code/Wrapper.py --pack . --worker-port 7790        # lab PC, render server running
python3 Code/Wrapper.py --pack . --worker-python PYTHON    # a GPU here (PYTHON has torch + gsplat)
python3 Code/Wrapper.py --pack . --cpu --max-s 2           # no GPU: a short check only
```

| Option | Default |
|---|---|
| `--policy` | `policy_template:WindowRacer`; use `policy_template:HoverHere` to check the loop |
| `--pair` | The first pair in `task.json` (`train_1`) |
| `--clearance-m` | 0.05 m |
| `--max-s` | 120 simulated seconds |
| `--log` | `run.json` |

It prints the score (windows passed in order, any virtual collision, the time)
and writes `run.json`, `run.png` (if matplotlib is installed) and
`fpv/<run>/frame_*.png`. The simulated drone follows each command exactly, with
no lag, drag or wind.

**`--cpu` is very slow.** Each frame projects, sorts and blends all ~950,000
Gaussians in numpy, which takes a second or more. At 15 frames per simulated
second, a full 120 s run is 1,800 frames, which would take hours. Use it only
to check the code runs, with a small `--max-s`, and use a GPU for whole runs.

To make the FPV video:

```bash
ffmpeg -framerate 15 -i fpv/<run>/frame_%05d.png -pix_fmt yuv420p FPVVideo.mp4
```

## 6. Fly it in the lab

### On the lab PC, outside Docker

Start the render server from `~/rbe595` and leave it running. Ask the TA for the
lab PC's GPU Python.

```bash
python3 -m splat_hitl.worker_server --bundle p3/scene --python <gpu python>
```

It serves on port 7790. Each flight that connects gets its own render worker.

### In the container

These steps follow our lab notes in `~/rbe595/steps.md`, from P2b.

1. Start the container from `cf_vicon_stack`:
   `ROS_DOMAIN_ID=3 docker compose -f docker/docker-compose.yml run --rm cf`
2. In each terminal, set `URI`, `TOPIC` and `VICON_TOPIC` to match `my_drone.yaml`.
3. Start the Vicon bridge: `ros2 launch vicon_receiver client.launch.py`.
4. Start the driver:
   `ros2 launch crazyflie_ros crazyflie.launch.py config:=/course/my_drone.yaml vicon:=false teleop:=false`.
   Add `no_fly:=true` first to check the estimator converges.
5. Take off to 0.7 m:
   `ros2 service call /cf1/takeoff crazyflie_interfaces/srv/Takeoff "{height: 0.7}"`
6. From our folder in `/course`, fly `HoverHere` as a dry run:
   ```bash
   PYTHONPATH=Code:$PYTHONPATH python3 -m splat_hitl.ros_node \
     --topic       $TOPIC \
     --contract    p3_contract.json \
     --transform   scene/vicon_transform.json \
     --gates       scene/gates.json \
     --esdf        scene/a3_train_esdf.npy \
     --clearance-m 0.05 \
     --worker-port 7790 \
     --policy      policy_template:HoverHere \
     --hold-altitude-m 0.7 \
     --log         hover.json \
     --dry-run
   ```
7. Run the same command with `--policy policy_template:WindowRacer --log run.json`,
   still with `--dry-run`.
8. Run it again without `--dry-run` to fly.
9. Land: `ros2 service call /cf1/land crazyflie_interfaces/srv/Land "{}"`.

`--dry-run` does everything except send commands. `--gates` only scores the run.
`ros_node` defaults to a 0.10 m clearance, so pass `--clearance-m 0.05` to match
the simulator. If `ros_node` dies, the driver's 300 ms command timeout stops the
drone. The full procedure is on the course site's Deployment page.

`splat_hitl`'s own notes say only the `--fake-room` mode has flown on hardware;
a rendered flight had not been flown when they were written. Always start with
`--dry-run`.

## Status on this laptop (2026-11-02)

| Item | State |
|---|---|
| `splat_hitl` | On `dev`, installed editable in miniconda's base Python 3.13 |
| `p3_contract.json` | Written, 320 x 213; not committed to git |
| `python3 -m splat_hitl.bundle scene` | Passes (exit code 0) |
| `check_setup.py` | Not run yet |
| `WindowRacer` | All TODOs still empty |
| GPU | RTX 5070 Laptop, 8 GB |
| Not installed | torch, gsplat, matplotlib, opencv-python, ultralytics, onnxruntime |
| `Wrapper.py --cpu` | Started once for 1 simulated second and stopped before it printed anything |

## Problems

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: splat_hitl` | `pip install -e splat_hitl` in the active environment |
| The scene loads but the contract doesn't | `splat_hitl` is on `main`: `git -C splat_hitl fetch origin && git -C splat_hitl checkout dev` |
| `no starter pack at …/p3_starter` | Pass `--pack .` (or set `P3_PACK`) |
| `no p3_contract.json` | `python3 tools/make_contract.py` |
| `no render server on 127.0.0.1:7790` | Start `worker_server`, or use `--worker-python` or `--cpu` |
| `the renderer is drawing another scene` | Restart the render server with `--bundle p3/scene` |
| A run with `--cpu` seems stuck | It isn't: lower `--max-s` or use a GPU |
| `torch.cuda.is_available()` is `False` on Turing | Use a GPU node, after `module load cuda/12.4.0/3mdaov5` |
| `ModuleNotFoundError: onnxruntime` in the container | Rebuild the image with `PIP_EXTRA="onnxruntime"` |
