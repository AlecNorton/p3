# Install

## The simulator

```bash
conda create -n aerial_robotics python=3.10
conda activate aerial_robotics
mkdir -p ~/rbe595 && cd ~/rbe595
git clone -b dev https://github.com/pearwpi/splat_hitl.git
pip install -e splat_hitl
pip install matplotlib            # optional: the run.png plot
```

Use `-b dev`: `main` cannot load this pack. Put this starter pack in `~/rbe595`
too, and set up from it:

```bash
python3 tools/make_contract.py
python3 tools/check_setup.py
```

`check_setup.py` stops at the first problem. `Wrapper.py --cpu` renders with
numpy, at seconds per frame: enough to check your code runs. For whole runs,
render on a GPU.

## A GPU of your own: Turing

On [Turing](https://docs.turing.wpi.edu/), in the same environment, on a GPU
node:

```bash
module load cuda/12.4.0/3mdaov5      # after every login
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install gsplat
```

Then `python3 Code/Wrapper.py --worker-python $(which python)`.

## The lab PC

Its render server draws the scene on the lab PC's GPU, for the simulator and
for the drone. Start it outside Docker, from `~/rbe595`, and leave it running:

```bash
python3 -m splat_hitl.worker_server --bundle p3_starter/scene --python <gpu python>
```

`<gpu python>` is the lab PC's Python with PyTorch and gsplat: ask the TA.
`Wrapper.py --worker-port 7790` and `ros_node --worker-port 7790` both use it.

## Your network in the lab container

The flight runs in the course container, which has numpy but no deep learning
library. Add one when you build the image, in `cf_vicon_stack`:

```bash
cd ~/rbe595/cf_vicon_stack
PIP_EXTRA="onnxruntime" docker compose -f docker/docker-compose.yml build
```

ONNX Runtime is small and quick on a CPU: export your network to ONNX and run
it with that. For PyTorch instead, use
`PIP_EXTRA="torch --extra-index-url https://download.pytorch.org/whl/cpu"`.
Then check, inside the container:

```bash
python3 -c "import onnxruntime, numpy, rclpy, geometry_msgs.msg; print('ok')"
```

The lab PC has one image for every team: if `onnxruntime` goes missing, rebuild it
the same way.

## Problems

| symptom | fix |
|---|---|
| `ModuleNotFoundError: splat_hitl` | `pip install -e splat_hitl`, in the active environment |
| the scene loads but the contract does not | you are on `main`: `git -C splat_hitl fetch origin && git -C splat_hitl checkout dev` |
| `no render server on 127.0.0.1:7790` | start the render server (above), or use `--worker-python` or `--cpu` |
| `torch.cuda.is_available()` is `False` on Turing | run on a GPU node, after `module load cuda/12.4.0/3mdaov5` |
| `ModuleNotFoundError: onnxruntime` in the container | rebuild the image with `PIP_EXTRA` (above) |
