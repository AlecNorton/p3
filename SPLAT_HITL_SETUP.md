# splat_hitl and lab PC setup: reference for new tests

Everything needed to run a new experiment on the same stack we used for P3:
the `splat_hitl` simulator and flight runtime, the render server, the lab PC
and its Docker container, Turing, and this laptop. It does not cover the P3
window detector itself; P3 appears only as the worked example.

Gathered on 2026-11-16 from `install.md`, `WORKFLOW.md`, `~/rbe595/steps.md`,
`Code/run_sim.sbatch`, the `splat_hitl` (`dev`) and `cf_vicon_stack` sources,
the P3 assignment page, and what we ran on this laptop. Where something is not
known, this file says so.

---

## 1. The pieces

```
 Vicon PC ──LAN──► vicon_receiver ──► /vicon/<obj>/<obj> (PoseStamped)
                                              │
                         ┌────────────────────┘
                         ▼
  ┌─────────────────── lab container (ROS 2 Humble, no GPU) ──────────────────┐
  │ splat_hitl.ros_node  (one timer at the contract rate, 15 Hz)              │
  │   Runtime.step():  latest pose ─► render request ─► policy.act() ─► cmd   │
  │                                     │      ▲                         │    │
  └─────────────────────────────────────┼──────┼─────────────────────────┼────┘
               TCP 127.0.0.1:7790       │      │ depth + RGB             │ cf1/cmd_hover
  ┌──── lab PC, outside Docker ─────────▼──────┴───┐      ┌──────────────▼───────────┐
  │ splat_hitl.worker_server ─► splat_rendering.py │      │ crazyflie_server (driver)│
  │   (GPU Python: torch + gsplat)                 │      │ own 50 Hz send timer     │
  └────────────────────────────────────────────────┘      └──────────────┬───────────┘
                                                                         ▼ Crazyradio
                                                                     Crazyflie
```

The drone, its latency and its tracking are real. The camera image is rendered
from a Gaussian splat of the room at the drone's real pose. The simulator
(`Code/Wrapper.py`) and the flight (`splat_hitl.ros_node`) run the same
`Runtime`, contract, observation builder, action stage and scoring.

### Folders

| Path | What it is |
|---|---|
| `~/rbe595/splat_hitl` | The library (`pearwpi/splat_hitl`, **branch `dev`**; `main` cannot load the P3 pack). Scene loading, contract, runtime, renderer client, render server, ROS node. Needs only numpy. |
| `~/rbe595/splat_hitl/worker/splat_rendering.py` | The GPU render worker (torch + gsplat). Never run by hand. |
| `~/rbe595/cf_vicon_stack` | The lab Docker stack: Vicon bridge, Crazyflie driver, safety core, diagnostics. |
| `~/rbe595/p3` | This repo: `Code/` (ours), `tools/` (course's), `scene/`, `p3_contract.json`. |
| `~/rbe595/steps.md` | Our lab-day checklist from P2b. |

In the container, the parent of `cf_vicon_stack` is mounted at `/course`, so
`splat_hitl`, this repo and `cf_vicon_stack` must be cloned **side by side** in
one folder. On the lab PC that folder is
`~/Documents/Handson_aerial_course_students/group3` (from `steps.md`).

---

## 2. How one tick runs, and how rendering runs alongside the policy

Rendering and the policy run in **separate processes**, often in separate
Python environments, and on the lab PC in separate places: the render worker on
the host GPU, the policy inside the GPU-less container. They do **not** overlap
in time. Each tick is strictly sequential:

```
tick k (every 1/15 s = 66.7 ms, from ros_node's ROS timer)
 ├─ read the latest Vicon pose; drop to HOLDING if it is older than 0.15 s
 ├─ score on the real pose: virtual collision (ESDF) and gate crossings
 ├─ render:  JSON request over the socket ──► worker renders on the GPU ──► depth + RGB back
 │           (blocking; the policy waits for the frame)
 ├─ build the observation (clip, normalise, stack history if a contract says so)
 ├─ policy.act(obs, state)  ◄── timed as policy_s
 ├─ map the Action to a Crazyflie hover command, apply limits
 └─ publish cf1/cmd_hover; count the tick late if render + act took over 0.10 s
```

The driver, `crazyflie_server`, is a separate ROS node with its own 50 Hz send
timer. It keeps resending the last command between our 15 Hz ticks, which is
why the loop does not have to hit 50 Hz.

**Render protocol** (`splat_hitl/renderer.py`, `SplatWorkerClient`): newline-
delimited JSON with base64 float32 payloads.

```
handshake  {"ready": true, "scale_to_metres": ..., "backend": ...}
request    {"position": [normalised], "orientation_rpy": [...], "image_width": W,
            "image_height": H, "fov_x_half_tan": tan(fov_x/2)}
response   {"ok": true, "shape": [H, W], "depth_b64": ..., "rgb_shape": [...], "rgb_b64": ...}
shutdown   {"cmd": "close"}
```

The client converts metres to the splat's normalised units on the way in and
back on the way out, and applies the camera mount. Above that layer everything
is in metres. The worker can be reached two ways:

- `SplatWorkerClient(sensor, worker_command(bundle, python=GPU_PY))` starts the
  worker as a subprocess on this machine. This is `Wrapper.py --worker-python`,
  used on Turing.
- `SplatWorkerClient(sensor, port=7790)` connects to `worker_server`. This is
  `--worker-port 7790` on the lab PC. Each connection gets a fresh worker that
  stops when the flight disconnects. The server listens on 127.0.0.1 only, and
  the container reaches it because it uses host networking.

### The timing budget

| Limit | Value | Where | What happens |
|---|---|---|---|
| Frame period | 66 ms (15 Hz) | assignment page, contract | Rendering, your network and your controller share it. "A loop that cannot keep up is stopped." |
| Tick budget | 100 ms (render + act) | `RuntimeConfig.tick_budget_s` | A slower tick is logged as `tick over budget`. |
| Late ticks | 3 in a row | `late_ticks_to_hold` | The runtime goes to `HOLDING` (hover in place). |
| Holding | over 1.0 s | `hold_before_land_s` | `LANDING` at 0.25 m/s, complete below 0.12 m. |
| Stale pose | over 0.15 s | `pose_stale_s` | `HOLDING`. |
| Mocap orientation lost | over 0.5 s | `ros_node` `QUAT_DEAD_S` | The run is stopped. |
| Step-rate check | measured vs contract | `Runtime` | Too far off the contract's rate ends the run with `rate_mismatch`. |
| Driver command timeout | 0.30 s | `crazyflie_server` | The driver holds the drone where it is. |
| Firmware watchdog | 0.5 s / 2 s | Crazyflie firmware | Levels out, then stops the motors. |

A policy that raises an exception, or returns a non-finite action, ends the run
(`policy_error`). That is deliberate.

**The simulator does not enforce timing.** `Wrapper.py` sets
`tick_budget_s=1e6` and advances a simulated clock by exactly 1/15 s per step,
however long rendering and `act()` take. A policy that is too slow still flies
fine in the simulator. Check the printed `your policy : … ms mean, … ms worst`
yourself.

### Consequences for any new policy

- **Load and warm up in `__init__`, never in `act()`.** Both `ros_node` and
  `Wrapper.py` build the policy before the loop starts, so `__init__` is not
  timed. A TorchScript model is slow on its first two calls while it profiles
  and optimises: we measured 812 ms and 766 ms on the RTX 5070, then about 5 ms.
  `Code/policy_template.py` runs three dummy inferences in `__init__` for this.
- **Measure on the CPU.** The container has no GPU. Code that picks
  `"cuda" if torch.cuda.is_available() else "cpu"` (as `Code/detector.py`
  does) silently runs on the CPU in flight, whatever it did on Turing or this
  laptop.
- **Budget for the worst frame, not the mean.** Three consecutive late ticks
  are enough to hold the drone. Test frames where your expensive code paths
  actually run.
- **Rendering takes part of the 66 ms.** The lab GPU's render time per frame is
  not recorded anywhere we have. Measure it with `Wrapper.py --worker-port
  7790` on the lab PC, which prints `rendering : … ms mean`.

---

## 3. The lab PC

**What we know**

| Item | Value |
|---|---|
| Hostname | `pear-2` (container prompt `student@pear-2:/ws$`; `cf_vicon_stack/fix_symlinks.py`) |
| OS | Linux. The Crazyradio needs a Linux host: Docker Desktop on macOS/Windows cannot pass USB through and fails silently. |
| Course folder | `~/Documents/Handson_aerial_course_students/group3`, mounted as `/course` in the container |
| GPU | Used **only outside Docker**, by `worker_server`. The container has no GPU. |
| GPU Python | A Python with torch + gsplat that the render server is started with. **Ask the TA for its path.** |
| Docker image | `cfvicon:base`, **one image shared by every team**. If your extra package disappears, someone rebuilt it: rebuild with your `PIP_EXTRA`. |
| Network | Container on host networking. Render server on `127.0.0.1:7790`. ROS 2 over UDPv4 to the Vicon PC on the LAN. |
| Loop | 15 Hz. Your network runs **in the container, on the CPU**, inside the 66 ms. |

**What we do not know.** No file we have, and not the assignment page, gives
the lab PC's CPU, GPU model, RAM or driver version. The course wiki
(<https://pear-wiki.wpi.edu/rbe595/>) needs a WPI login and may have them.
Record them on the next lab visit:

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
lscpu | grep -E "Model name|^CPU\(s\)"; free -h; docker images | grep cfvicon
```

---

## 4. Docker (cf_vicon_stack)

### One-time host setup (already done on the lab PC)

```bash
cd cf_vicon_stack && ./docker/host-setup.sh    # as yourself, not sudo; log out and back in afterwards
```

This installs the Crazyradio udev rule, adds you to `plugdev` and `docker`, and
writes `docker/.env` (`USER_UID`, `USER_GID`, `PLUGDEV_GID`) so files the
container writes are yours. `docker/.env` is machine-specific: never copy or
commit it.

### Build, with the packages your policy needs

The base image has numpy and ROS, but **no deep-learning library**. Add one at
build time with `PIP_EXTRA`:

```bash
cd cf_vicon_stack
PIP_EXTRA="onnxruntime" docker compose -f docker/docker-compose.yml build
# or PyTorch, CPU-only:
PIP_EXTRA="torch --extra-index-url https://download.pytorch.org/whl/cpu" \
  docker compose -f docker/docker-compose.yml build
```

Check inside the container afterwards. NumPy must still work with the ROS
message extensions:

```bash
python3 -c "import numpy, rclpy, geometry_msgs.msg, cflib.crtp; print('ok')"
python3 -c "import onnxruntime"        # or: import torch, cv2, ...
```

For more than pip packages, copy `docker/example-student.Dockerfile`
(`FROM cfvicon:base`). Install as root, finish as `student`, and build ROS
packages into an overlay workspace (`OVERLAY_WS=/team_ws`), **never `/ws`**.
The entrypoint sources the overlay automatically. A build that fails
`verify.py` or the four test suites does not produce an image.

### Run

```bash
ROS_DOMAIN_ID=3 docker compose -f docker/docker-compose.yml run --rm cf
# first line printed: cf_vicon_stack | ROS_DISTRO=humble ROS_DOMAIN_ID=3 | transports=UDPv4
```

- `ROS_DOMAIN_ID=3` is **our team's**. Every team on the same domain sees each
  other's setpoints. Valid values are 1–101 and 215–232.
- Each terminal needs its own container, started the same way.

| Compose setting | Effect |
|---|---|
| `network_mode: host` | DDS reaches the Vicon PC; `127.0.0.1:7790` is the host's render server |
| `/dev/bus/usb` bind mount + `device_cgroup_rules: c 189:* rmw` | Crazyradio works across unplug/replug, without `privileged` |
| `../:/ws` | The editable `cf_vicon_stack` copy. The compiled stack lives in `/opt/cfvicon`, so mounts never shadow it. |
| `../..:/course` | The course folder: `splat_hitl`, this repo, your code |
| `PYTHONPATH=/course/splat_hitl` | `splat_hitl` imports without a pip install |
| `FASTDDS_BUILTIN_TRANSPORTS=UDPv4`, `ROS_LOCALHOST_ONLY=0` | Predictable DDS; the container can see the Vicon PC |

---

## 5. Flying in the lab

Our values: `URI=radio://0/80/2M/E7E7E7E708`,
`TOPIC=/vicon/crazyfly_g3/crazyfly_g3`. They must match `/course/my_drone.yaml`
(`cp /ws/src/crazyflie_ros/config/crazyflie.yaml /course/my_drone.yaml` and set
`uri` and `vicon_topic`).

1. **Render server.** On the lab PC, outside Docker, from the course folder.
   Start it on the **same scene** you will fly, and leave it running:
   ```bash
   python3 -m splat_hitl.worker_server --bundle p3/scene --python <gpu python>
   ```
   It starts the worker once and waits for its ready handshake, so a broken GPU
   environment fails here, not mid-flight.
2. **Containers.** Open four terminals, each with `ROS_DOMAIN_ID=3 docker
   compose … run --rm cf`. In each one:
   `export URI=… TOPIC=… VICON_TOPIC=$TOPIC`.
3. **Vicon bridge:** `ros2 launch vicon_receiver client.launch.py`. You should
   see a steady stream of latency lines.
4. **Estimator check:**
   `ros2 launch crazyflie_ros crazyflie.launch.py config:=/course/my_drone.yaml vicon:=false teleop:=false no_fly:=true`.
   An EKF-to-Vicon error in the double digits of cm means the markers or Vicon
   are off.
   - Optional, from `/ws`: `python3 radio_doctor.py --uri $URI` and
     `python3 crazyflie_vicon_teleop.py --uri $URI --topic $TOPIC --no-fly`.
5. **Driver:** the same launch command without `no_fly:=true`.
6. **Take off** to the altitude the runtime will hold:
   `ros2 service call /cf1/takeoff crazyflie_interfaces/srv/Takeoff "{height: 0.7}"`.
7. **Dry run with `HoverHere`**, from `/course/p3`. `--dry-run` does everything
   except publish commands.
   ```bash
   PYTHONPATH=Code:$PYTHONPATH python3 -m splat_hitl.ros_node \
     --topic $TOPIC --contract p3_contract.json \
     --transform scene/vicon_transform.json --gates scene/gates.json \
     --esdf scene/a3_train_esdf.npy --clearance-m 0.05 \
     --worker-port 7790 --policy policy_template:HoverHere \
     --hold-altitude-m 0.7 --log hover.json --dry-run
   ```
8. Run the same command with your policy (`--policy policy_template:WindowRacer
   --log run.json`), still with `--dry-run`. Then run it without `--dry-run` to
   fly.
9. **Land:** `ros2 service call /cf1/land crazyflie_interfaces/srv/Land "{}"`.
   `cf1/emergency` cuts the motors, and the drone falls.

`ros_node` options:

| Option | Default | Notes |
|---|---|---|
| `--topic` | (required) | The Vicon `PoseStamped` topic |
| `--policy` | `hover` | `module:Class`; the module must be on `PYTHONPATH` (hence `PYTHONPATH=Code`) |
| `--contract` | none | Checked against the policy's `sensor_fingerprint` |
| `--transform` | none | Vicon → splat. Omit it only with `--fake-room`. |
| `--gates`, `--esdf` | none | Scoring and collision only. The policy never sees them. |
| `--clearance-m` | **0.10** | The simulator uses 0.05: pass it explicitly |
| `--worker-port` / `--worker` | none | The render server port, or a shell command that starts a worker |
| `--fake-room W D H` | none | Renders a plain box with no GPU. The only mode `splat_hitl`'s notes say has flown on hardware. |
| `--hold-altitude-m` | 0.60 | The start altitude; the policy's vz moves it from there |
| `--rate-hz` | the contract's | Leave it: a mismatch ends the run |
| `--yaw-sign` | 1 | |
| `--prefix` | `/cf1` | |
| `--log`, `--dry-run` | | |

Commands go out on `cf1/cmd_hover` (vx, vy in the body frame, yaw rate,
z distance). FPV frames land in `fpv/<run>/` in the folder you ran from.

---

## 6. Turing (WPI cluster): full GPU simulator runs

Docs: <https://docs.turing.wpi.edu/>. Work on a GPU node, in
`~/scratch/rbe595/p3`.

**Environment.** These are two variants and we have not checked which one
works.
- `install.md`:
  ```bash
  module load cuda/12.4.0/3mdaov5
  pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
  pip install gsplat
  ```
- Our `Code/run_sim.sbatch` instead loads `module load cuda12.6/toolkit/12.6.3`
  and expects torch 2.6.0+cu126, in a conda env at
  `~/.conda/envs/aerial_robotics` (override with `PY=…`).

Also install `splat_hitl` (`pip install -e splat_hitl`, branch `dev`) and
whatever your policy imports (`opencv-python`, …) in that env. gsplat compiles
its CUDA kernels on first use, so the first run is slow.

**Batch job:**

```bash
cd ~/scratch/rbe595/p3 && git pull
sbatch Code/run_sim.sbatch                                  # a3_train, default policy
sbatch Code/run_sim.sbatch --scene /path/to/scene --max-s 90  # extra args go to Wrapper.py
squeue -u $USER
tail -f slurm-p3_sim-<jobid>.out
```

The job requests partition `academic`, account `rbe595`, 1 GPU, 4 CPUs, 16 GB
and 45 minutes. It checks that torch sees a GPU, then runs `Wrapper.py --pack .
--worker-python $PY`. Results land in `runs/<jobid>/`: `run.json`, `run.png`,
`run_log.txt`, `fpv/` and `FPVVideo.mp4` (made with ffmpeg if the module
exists, otherwise OpenCV). The score and timing lines are printed at the end.
`~/rbe595/turing_run_p3` has the same commands. Note that the file header says
to submit `run_sim.sbatch` from the repo root, but the file lives in `Code/`.

**Interactive:** on a GPU node, `python3 Code/Wrapper.py --pack .
--worker-python "$(which python)"`. On a login node `torch.cuda.is_available()`
is `False`: use a GPU node.

---

## 7. This laptop (RTX 5070 Laptop, 8 GB, Blackwell sm_120, 15 GB RAM, no swap)

Conda env `aerial_robotics`, Python 3.10:

| Package | State |
|---|---|
| `splat_hitl` 0.1.0 | Editable from `~/rbe595/splat_hitl` |
| torch 2.11.0+cu128, torchvision 0.26.0+cu128 | CUDA works (`(12, 0)`); the policy's network runs on the GPU |
| numpy 2.2.6, opencv-python | Installed |
| gsplat 1.5.3 (PyPI) | Installed but **uncompiled**: GPU rendering does not work here |
| cuda-nvcc 12.8, gcc/g++ 13.4 (conda-forge) | Installed for the gsplat build. g++ 14 is rejected by torch for CUDA 12.8. |
| `CUDA_HOME`, `TORCH_CUDA_ARCH_LIST=12.0` | Set as env config vars (`conda env config vars unset …` to remove) |
| matplotlib | Not installed, so `Wrapper.py` skips `run.png` |

**Why GPU rendering fails here.** Building gsplat from git compiles 36 of 50
CUDA files, then the kernel's out-of-memory killer stops it on
`RasterizeToPixelsFromWorld3DGSParallelBatchFwd.cu`. `ptxas` alone needs about
9.9 GB for that file, even with `MAX_JOBS=1`. The build also needs torch's
pip CUDA headers on the include path:
`export CPATH=$(ls -d $CONDA_PREFIX/lib/python3.10/site-packages/nvidia/*/include | paste -sd:)`.
Finishing it would need a swap file (sudo) or a machine with more RAM. Use
Turing for GPU runs.

**What works here:**

```bash
conda activate aerial_robotics && cd ~/rbe595/p3
python tools/check_setup.py                              # passes 1-7
python Code/Wrapper.py --pack . --cpu --max-s 3          # numpy renderer
```

Measured: CPU rendering 4–9 s per frame, so 3 simulated seconds took 253 s of
wall clock. Policy `act()` takes about 5 ms per frame on the GPU after warm-up.
`Wrapper.py` writes `run.json` and `fpv/` into the current directory: run it
from a scratch folder (`--log /tmp/x/run.json`) to keep the repo clean.

---

## 8. Running a different test in the same environment

### Policy

```python
import os
from splat_hitl.commands import Action
from splat_hitl.policy import Policy

HERE = os.path.dirname(os.path.abspath(__file__))

class MyTest(Policy):
    name = "my_test"                 # goes into the run log
    sensor_fingerprint = None        # set to the contract's sensor fingerprint once fixed

    def __init__(self):
        # load weights from os.path.join(HERE, ...) and warm up here
        ...

    def reset(self):                 # called before each run
        ...

    def act(self, obs, state) -> Action:
        return Action("velocity", "body_flu", [vx, vy, vz], yaw_rate_rad_s=w)
```

- **Inputs.**
  - `obs.rgb` is (H, W, 3) float32 in [0, 1]. `obs.depth_m` is (H, W) metres
    along the camera axis; empty pixels read the contract's clip depth (4.0 m).
  - `obs.policy_input` is the clipped, normalised, stacked tensor for a
    contracted trained policy.
  - `state` carries `t_s`, `position_m` (scene metres), `velocity_ms`,
    `yaw_rad` (CCW from +x), and gate fields. `Wrapper.py` blanks the gate
    fields, and the course forbids reading them.
- **Outputs.**
  - `Action.kind` is one of `velocity`, `acceleration`, `position`.
  - `frame` is one of `world_enu`, `world_ned`, `body_flu`, `body_frd`.
  - `yaw_rate_rad_s` is CCW-positive about world +z in every frame.
  - Only `velocity`/`body_flu` has been flown in P3. Read
    `splat_hitl/commands.py` (`make_action_stage`) before relying on another
    kind, because `ros_node` always ends up publishing `cmd_hover`.
- **Selecting it.**
  - `Wrapper.py --policy module:Class` imports from `Code/`.
  - `ros_node` needs `PYTHONPATH=Code:$PYTHONPATH`.
  - The course runs `Code/Wrapper.py` on its own machine, so use paths built
    from `HERE`, never absolute ones.
- `Code/policy_template.py:HoverHere` (zero velocity plus FPV recording) is the
  loop check to fly first.

### Camera and action contract

```bash
python3 tools/make_contract.py              # p3_contract.json, 320 x 213
python3 tools/make_contract.py --width 480  # the height follows the 3:2 shape
```

P3's contract keeps the scene camera: 60.66° horizontal FOV, RGB + depth,
depth clipped at 4.0 m, 15 Hz, body-frame velocity with free yaw. `Wrapper.py`
prints it as
`obs 4x213x320 clip_unit clip 4.0, act velocity body_flu x1.50 yaw=free, 15 Hz`.
`check_setup.py` step 5 fails if the file differs from what `make_contract.py`
writes. Once `sensor_fingerprint` is set, the runtime refuses any other camera.

### Scene

A scene is a folder with `manifest.json` that lists its files.
- **Required:** `splat`, `esdf`, `contract`.
- **Optional:** `gates` (a race), `task` (start pairs for `--pair`), and
  `vicon_transform` (**needed to fly**).
- **Provenance files:** `anchors`, `pointcloud`, `splat_frame`,
  `scene_calibration`, `dataparser_transforms`, `constraints`.

The manifest also carries `_render_worker_args` (backend, scale to metres,
empty depth) that `worker_command` uses. Check a scene before using it:

```bash
python3 -m splat_hitl.bundle path/to/scene 0.05   # exit 0 = passes; lines marked "note" are not errors
```

Point everything at the same scene. That means `Wrapper.py --scene …` (or
`--pack`), `tools/render_views.py --scene …`, and the render server's
`--bundle …`. A server on another scene stops the run with "the renderer is
drawing another scene".

### Order of work

1. `python tools/check_setup.py`. Its step 7 hard-codes
   `policy_template.WindowRacer`, so for another class run it once by hand in
   Python.
2. `python Code/Wrapper.py --pack . --cpu --max-s 2 --policy mod:Class`, a
   quick check on any machine.
3. Views for testing perception, at chosen poses:
   `python tools/render_views.py --cpu --pose x y z yaw_deg` (or
   `--worker-port` / `--worker-python`). They are written to `views/`.
4. A full run on Turing: `sbatch Code/run_sim.sbatch --policy mod:Class …`.
5. Time `act()` on the CPU, including worst-case frames, against the 66 ms
   budget less the render time.
6. The lab: rebuild the image if you need a new pip package, start the render
   server on your scene, do the `HoverHere` dry run, then your policy dry run,
   then fly.

Useful `Wrapper.py` flags: `--pack`, `--scene`, `--contract`, `--policy`,
`--pair` (from `task.json`; default the first), `--clearance-m` (0.05),
`--max-s` (120), `--log` (`run.json`). The simulated drone follows each command
exactly, with no lag, drag or wind. FPV video:
`ffmpeg -framerate 15 -i fpv/<run>/frame_%05d.png -pix_fmt yuv420p FPVVideo.mp4`.

### Course rules that carry over

- Train only on synthetic images we generate (Blender or similar). Splat
  renders are for testing only.
- The policy must not read `gates.json` or the gate fields of the state.
- Name the class `ros_node` flies in `README.md`.

---

## 9. Problems we have hit

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: splat_hitl` | `pip install -e splat_hitl` in the active env (in the container it comes from `PYTHONPATH=/course/splat_hitl`) |
| Scene loads but the contract does not | `splat_hitl` is on `main`: `git -C splat_hitl checkout dev` |
| `no starter pack at …/p3_starter` | Pass `--pack .` or set `P3_PACK` |
| `no p3_contract.json` | `python3 tools/make_contract.py` |
| `no render server on 127.0.0.1:7790` | Start `worker_server`, or use `--worker-python` / `--cpu` |
| `the renderer is drawing another scene` | Restart the render server with `--bundle <your scene>` |
| `The provided filename … does not exist` | Weights loaded by a bare relative name. Build the path from `HERE`. |
| First one or two ticks take ~800 ms | TorchScript warm-up. Run dummy inferences in `__init__`. |
| `--cpu` run looks stuck | It isn't: several seconds per frame. Lower `--max-s`. |
| `torch.cuda.is_available()` is `False` on Turing | Use a GPU node, after `module load` |
| `ModuleNotFoundError: onnxruntime` / `torch` in the container | The shared image was rebuilt: rebuild with `PIP_EXTRA` |
| `import geometry_msgs.msg` breaks after a pip install | A NumPy major-version change. Rebuild, pinning `numpy<2` in `PIP_EXTRA`. |
| `ros2 topic list` empty | `ROS_DOMAIN_ID` mismatch, or `ROS_LOCALHOST_ONLY` set |
| gsplat build on the laptop: g++ 14 rejected | Install `gcc_linux-64=13 gxx_linux-64=13` (conda-forge) |
| gsplat build on the laptop: `cusparse.h` missing | Put torch's `site-packages/nvidia/*/include` on `CPATH` |
| gsplat build on the laptop: `Killed`, code 137 | Out of RAM (`ptxas` about 9.9 GB). Needs swap or a bigger machine. |
