#!/usr/bin/env python3
"""Writes p3_contract.json, the contract you fly with.

    python3 tools/make_contract.py                # 320 x 213
    python3 tools/make_contract.py --width 480    # 480 x 320

It keeps the scene's camera -- 60.66 deg horizontal field of view, 3:2, looking
straight ahead, RGB and metric depth at 15 Hz -- at the width you choose, and
sets the action to a velocity in the body frame with a free yaw: what your
controller outputs. Fly and simulate with the same file.
"""
import argparse
import os
import sys
from dataclasses import replace

from splat_hitl.contract import ActionSpec, ControlSpec, ObservationSpec, PolicyContract

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.dirname(HERE)


def build(scene_contract_path, width=320):
    scene = PolicyContract.load(scene_contract_path)
    s = scene.observation.sensor
    width = int(width)
    if not 64 <= width <= 1920:
        raise ValueError("width %d: pick between 64 and 1920 pixels" % width)
    height = int(round(width * s.height / float(s.width)))     # keep the 3:2 shape
    o = scene.observation
    return PolicyContract(
        name="p3",
        observation=ObservationSpec(
            sensor=replace(s, width=width, height=height),      # same field of view
            clip_far_m=o.clip_far_m, normalize=o.normalize, history=1,
            nan_fill_m=o.nan_fill_m, neginf_fill_m=o.neginf_fill_m,
            channels="rgb_depth", rgb_range_max=o.rgb_range_max),
        action=ActionSpec(kind="velocity", frame="body_flu", scale=1.5,
                          yaw_mode="free"),
        control=ControlSpec(rate_hz=scene.control.rate_hz),
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default=os.path.join(PACK, "scene"))
    ap.add_argument("--width", type=int, default=320,
                    help="image width in pixels; the height follows (3:2)")
    ap.add_argument("--out", default=os.path.join(PACK, "p3_contract.json"))
    a = ap.parse_args(argv)

    src = os.path.join(a.scene, "policy_contract.json")
    if not os.path.exists(src):
        sys.exit("no %s -- is --scene pointing at the scene folder?" % src)
    try:
        c = build(src, a.width)
    except ValueError as e:
        sys.exit(str(e))
    c.save(a.out)
    out = os.path.relpath(a.out)
    if out.startswith(".."):          # outside the cwd: print it in full
        out = a.out
    s = c.observation.sensor
    print("wrote %s" % out)
    print("  camera : %d x %d, %.2f deg horizontal, rgb + depth, %.0f Hz"
          % (s.width, s.height, s.fov_x_deg, c.control.rate_hz))
    print("  action : %s in %s, yaw %s" % (c.action.kind, c.action.frame,
                                           c.action.yaw_mode))
    print("\nsimulate and fly with:  --contract %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
