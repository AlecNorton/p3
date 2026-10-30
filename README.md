# P3 starter pack

The assignment page is the specification.

```
p3_starter/
├── install.md              set up first
├── scene/                  the training course, a3_train
├── textures/               the window print: Window.png and Windows.pptx
├── Code/                   yours: copy into your own folder and edit
│   ├── Wrapper.py          flies your policy in the simulator (the file we run)
│   ├── policy_template.py  detector, state machine and controller: fill in the TODOs
│   └── fpv.py              records what the camera saw, for FPVVideo.mp4
└── tools/                  ours: use them, do not submit them
    ├── check_setup.py      checks your setup
    ├── make_contract.py    writes p3_contract.json: your camera and action
    ├── render_views.py     renders the camera's view at poses you pick
    ├── cpu_renderer.py     renders the splat without a GPU, slowly
    └── splat_cpu.py        the maths behind it
```

## Steps

Keep this folder next to your own (`~/rbe595/p3_starter`, `~/rbe595/p3_Group1`).
In this folder:

```bash
python3 tools/make_contract.py    # 1. write p3_contract.json; --width sets the resolution
python3 tools/check_setup.py      # 2. check the setup
```

Copy `Code/` into your folder, fill in the TODOs, and in your folder fly it in
the simulator with one of:

```bash
python3 Code/Wrapper.py --worker-port 7790        # 3. on the lab PC, with the render server running
python3 Code/Wrapper.py --worker-python PYTHON    #    on your own GPU (Turing: install.md)
python3 Code/Wrapper.py --cpu                     #    anywhere, at seconds per frame
```

It prints the score and writes `run.json`, `run.png` and `fpv/<run>/`.

To fly, in the lab container, with the render server running: first the hover
dry run `check_setup.py` prints, then this from your folder, then again without
`--dry-run` (full procedure: the course site's Deployment page):

```bash
PYTHONPATH=Code:$PYTHONPATH python3 -m splat_hitl.ros_node \
  --topic       /vicon/<object>/<object> \
  --contract    ../p3_starter/p3_contract.json \
  --transform   ../p3_starter/scene/vicon_transform.json \
  --gates       ../p3_starter/scene/gates.json \
  --esdf        ../p3_starter/scene/a3_train_esdf.npy \
  --clearance-m 0.05 \
  --worker-port 7790 \
  --policy      policy_template:WindowRacer \
  --hold-altitude-m 0.7 \
  --log         run.json \
  --dry-run
```

## Notes

- **Training data.** Only synthetic images you make (Blender or similar).
  Anything rendered from a splat — `Wrapper.py`, `render_views.py`, the FPV
  frames — is for testing.
- **Where the windows are** comes from your detector. Do not read
  `gates.json` or the gate fields of `state`: `Wrapper.py` hands your policy a
  state without them, and `--gates` is only there to score the run.
- **Frames.** `state.position_m` is in scene metres. Our scenes are stored in
  the Vicon frame: origin at the L tape, x down the course, z up from the
  floor. Your velocity goes out in the body frame.
- **Timing.** The loop runs at 15 Hz. On the lab PC, rendering and `act()`
  share each 66 ms frame; `Wrapper.py` prints how long `act()` took.
- **FPV video.** Each run leaves `fpv/<run>/frame_*.png`, also after
  `ros_node`. Make the video with
  `ffmpeg -framerate 15 -i fpv/<run>/frame_%05d.png -pix_fmt yuv420p FPVVideo.mp4`.
- **The window.** `textures/Window.png` is the print at full resolution and
  `textures/Windows.pptx` the file it was made from, at its printed size.
