#!/usr/bin/env python3
"""Checks the P3 setup layer by layer and stops at the first failure.

    python3 tools/check_setup.py              # a few seconds, no GPU
    python3 tools/check_setup.py --render     # also renders one frame on the CPU

It ends with the command for a hover dry run in the lab.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.dirname(HERE)


def step(n, what):
    sys.stdout.write("%d. %-46s" % (n, what))
    sys.stdout.flush()


def ok(msg=""):
    print("ok   %s" % msg)


def die(msg, hint):
    print("FAIL")
    sys.exit("\n   %s\n\n   %s\n" % (msg, hint))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", default=PACK)
    ap.add_argument("--code", default=None,
                    help="folder holding your policy_template.py; default <pack>/Code")
    ap.add_argument("--pair", default="train_1")
    ap.add_argument("--render", action="store_true",
                    help="render the start view on the CPU into check_view.png")
    a = ap.parse_args(argv)
    scene_dir = os.path.join(a.pack, "scene")
    code_dir = os.path.abspath(a.code or os.path.join(a.pack, "Code"))

    step(1, "import numpy")
    try:
        import numpy as np
    except ImportError:
        die("numpy is not installed.", "pip install numpy  (inside your conda env)")
    ok(np.__version__)

    step(2, "import splat_hitl")
    try:
        from splat_hitl.bundle import SceneBundle
        from splat_hitl.policy import PolicyState
    except ImportError as e:
        die("splat_hitl is not importable: %s" % e,
            "See install.md. The branch is `dev` -- `main` cannot load this pack.")
    ok()

    step(3, "load the scene")
    try:
        b = SceneBundle.load(scene_dir)
    except Exception as e:
        die("SceneBundle.load(%s) raised %s" % (scene_dir, e),
            "Point --pack at the starter pack folder (the one holding scene/).")
    ok("%s, %d windows" % (b.name, len(b.gates().gates)))

    step(4, "scene self-check")
    rep = b.check(clearance_m=0.05)
    if not rep.ok:
        die("the scene has errors:\n%s" % rep, "Download the starter pack again.")
    ok()

    step(5, "the P3 contract")
    sys.path.insert(0, HERE)
    from make_contract import build                           # noqa: E402
    from splat_hitl.contract import PolicyContract            # noqa: E402
    path = os.path.join(a.pack, "p3_contract.json")
    if not os.path.exists(path):
        die("no p3_contract.json yet.", "python3 tools/make_contract.py")
    c = PolicyContract.load(path)
    want = build(os.path.join(scene_dir, "policy_contract.json"), c.observation.sensor.width)
    if c.fingerprint() != want.fingerprint():
        die("p3_contract.json is not what make_contract.py writes.",
            "Write it again: python3 tools/make_contract.py --width %d"
            % c.observation.sensor.width)
    s = c.observation.sensor
    ok("%d x %d, %s, yaw %s" % (s.width, s.height, c.action.kind, c.action.yaw_mode))

    step(6, "clearance at the %s start" % a.pair)
    pair = next((p for p in b.task()["pairs"] if p["name"] == a.pair), None)
    if pair is None:
        die("no pair %r in scene/task.json." % a.pair, "Pass --pair with a name from task.json.")
    tf = b.transform()
    p_s = tf.point_to_splat(np.array(pair["start"], float)).reshape(3) * tf.metres_per_unit
    r = b.esdf().at(p_s)
    if getattr(r, "outside", False):
        die("the start is outside the mapped volume.",
            "Check that scene/vicon_transform.json is the one this pack shipped.")
    ok("%.0f mm" % (r.metres * 1000))

    step(7, "your policy runs one frame")
    sys.path.insert(0, code_dir)
    try:
        from policy_template import WindowRacer                # noqa: E402
        from splat_hitl.renderer import Observation            # noqa: E402
        pol = WindowRacer(record_fpv=False)
        pol.reset()
        obs = Observation(np.full((s.height, s.width), 4.0, np.float32),
                          np.zeros((s.height, s.width, 3), np.float32), 0.0, s.fingerprint())
        act = pol.act(obs, PolicyState(0.0, p_s, np.zeros(3), 0.0))
    except Exception as e:
        die("policy_template.WindowRacer failed: %r" % (e,),
            "Run it in Python to see the traceback; --code points at its folder.")
    if act is None or act.kind != "velocity":
        die("act() returned %r." % (act,), "Return Action(\"velocity\", \"body_flu\", ...).")
    ok(code_dir if a.code else "")

    if a.render:
        step(8, "render the start view on the CPU")
        from cpu_renderer import CpuSplatRenderer               # noqa: E402
        sys.path.insert(0, os.path.join(a.pack, "Code"))
        from fpv import depth_colour, write_png                 # noqa: E402
        o = CpuSplatRenderer(s, b).render(p_s, (0.0, 0.0, 0.0))
        img = np.concatenate([(np.clip(o.rgb, 0, 1) * 255 + 0.5).astype(np.uint8),
                              depth_colour(o.depth_m)], axis=1)
        write_png("check_view.png", img)
        ok("check_view.png, %.1f s" % o.render_s)

    print("""
Hover dry run: in the lab container, from your own folder next to this pack,
with the render server running (course site, Deployment page).

    PYTHONPATH=Code:$PYTHONPATH python3 -m splat_hitl.ros_node \\
      --topic       /vicon/<object>/<object> \\
      --contract    ../p3_starter/p3_contract.json \\
      --transform   ../p3_starter/scene/vicon_transform.json \\
      --gates       ../p3_starter/scene/gates.json \\
      --esdf        ../p3_starter/scene/a3_train_esdf.npy \\
      --clearance-m 0.05 \\
      --worker-port 7790 \\
      --policy      policy_template:HoverHere \\
      --hold-altitude-m 0.7 \\
      --log         hover.json \\
      --dry-run

--dry-run sends nothing. To fly, drop it, with the driver running and the drone
hovering at 0.7 m. --gates scores the run; your policy may not use it.
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
