#!/usr/bin/env python3
"""Your P3 policy: find a window, line up with it, fly through it, repeat.
Fill in the TODOs.

    python3 Code/Wrapper.py --policy policy:WindowRacer        # in the simulator
    ros_node ... --policy policy:WindowRacer                   # on the drone

act(obs, state) is called 15 times a second with

    obs.rgb          (H, W, 3) float32 in [0, 1]
    obs.depth_m      (H, W) float32, metres along the camera axis; 4.0 where
                     nothing was hit
    state.t_s        seconds since the run started
    state.position_m, state.velocity_ms
                     scene metres and m/s (our scenes are stored in Vicon metres)
    state.yaw_rad    heading, counter-clockwise from +x

and returns Action("velocity", "body_flu", [vx, vy, vz], yaw_rate_rad_s=w):
m/s in the body frame (x forward, y left, z up) and a counter-clockwise yaw
rate in rad/s. The camera sits at the drone's centre, looks along body +x and
turns with it.

Rendering, your detector and act() share one 66 ms frame on the lab PC's CPU.
Where the windows are must come from obs alone: do not read gates.json, or
state.gate_index, state.gate_distance_m or state.gate_bearing_rad.

The modes (each one a class with start() and update()):

    survey   hold position, sweep yaw +-60 deg at 30 deg/s around the current
             heading, then face the closest tracked gate ahead          -> view
    view     quintic to a point d in front of the gate along its normal
             (d = 1.2, 1.0, 0.8 m: the first inside the flight box), holding yaw;
             on arrival turn head-on (along the normal)                 -> measure
    measure  hold; skip 5 ticks, then 15 ticks. Needs a closed hole in >= 10 of
             them and a pass line within 15 deg of the normal          -> pass
             otherwise view again (3 views, then one 0.2 m back-off), then false
    pass     quintic from here through the centre to 0.6 m past it, yaw held;
             passed when the plane is crossed            -> view / survey / done
    done     hover

The gate list is updated only in survey and measure.
"""
import math
import os

import numpy as np

from splat_hitl.commands import Action
from splat_hitl.policy import Policy

from fpv import FpvRecorder
import detector
from control import PositionController, Quintic, YawController, heading_of

HERE = os.path.dirname(os.path.abspath(__file__))
N_WINDOWS = 3                          # the course has three windows
FOV_X_DEG = 60.656255852364644         # fixed: the drone camera's field of view

# ----------------------------------------------------------------- decisions
SWEEP_RAD = math.radians(60.0)         # survey: +-60 deg around the current heading
SWEEP_RATE = math.radians(30.0)        # survey: 30 deg/s
VIEW_DISTANCES = (1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6)   # view: first inside the box with a clear line
BOX_MARGIN_M = 0.3
BACKOFF_M = 0.2                        # after 3 views: one more, 0.2 m further back
MAX_VIEWS = 3
MEASURE_SKIP = 5                       # ticks to let the drone settle
MEASURE_TICKS = 15
MEASURE_VALID_MIN = 10                 # closed-hole frames needed out of MEASURE_TICKS
MATCH_DIST_M = 0.75 * 0.91             # false-gate check: centroid within 3/4 window width
MATCH_ANGLE_DEG = 20.0
MATCH_DEPTH_TOL_M = 0.3                # ... and about d_view away
PASS_MAX_ANGLE_DEG = 15.0              # pass line vs normal
PASS_EXIT_M = 0.6                      # stop 0.6 m past the centre
# The flight box (Vicon metres = scene metres in a3_train and a3_test) and height band.
BOX_MIN = np.array([-0.50, -1.20, 0.10])
BOX_MAX = np.array([5.25, 1.20, 1.80])


def camera_matrix(width, height, fov_x_deg=FOV_X_DEG):
    """Pinhole intrinsics of the drone camera: square pixels, centre at the middle."""
    f = 0.5 * width / math.tan(math.radians(fov_x_deg) / 2.0)
    return np.array([[f, 0.0, width / 2.0],
                     [0.0, f, height / 2.0],
                     [0.0, 0.0, 1.0]])


def pixel_to_camera(u, v, depth_m, K):
    """Pixel (u, v) at depth z -> camera frame (x right, y down, z forward).

    Pixel (i, j) covers [i, i + 1) x [j, j + 1): its centre is (i + 0.5, j + 0.5).
    """
    u, v, z = (np.asarray(a, float) for a in (u, v, depth_m))
    return np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], axis=-1)


def camera_to_world(p_cam, state):
    """Camera frame -> world. Uses the yaw only; roll and pitch are a few degrees."""
    p = np.asarray(p_cam, float)
    body = np.stack([p[..., 2], -p[..., 0], -p[..., 1]], axis=-1)        # FLU
    c, s = math.cos(state.yaw_rad), math.sin(state.yaw_rad)
    world = np.stack([c * body[..., 0] - s * body[..., 1],
                      s * body[..., 0] + c * body[..., 1], body[..., 2]], axis=-1)
    return np.asarray(state.position_m, float) + world


def world_to_body(v_world, yaw_rad):
    """World velocity -> body (x forward, y left, z up)."""
    v = np.asarray(v_world, float)
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    return np.array([c * v[0] + s * v[1], -s * v[0] + c * v[1], v[2]])


def wrap_pi(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def in_box(p, margin=BOX_MARGIN_M):
    p = np.asarray(p, float)
    return bool(np.all(p >= BOX_MIN + margin) and np.all(p <= BOX_MAX - margin))


def clamp_box(p, margin=BOX_MARGIN_M):
    return np.clip(np.asarray(p, float), BOX_MIN + margin, BOX_MAX - margin)


# --------------------------------------------------------------------------- #
# Passed windows as obstacles: the board (known size, centred on the measured
# centre, in the measured plane) and its two poles, all grown by OBSTACLE_INFLATE_M.
# --------------------------------------------------------------------------- #
BOARD_M = 0.91                          # printed board, square, centred on the opening
POLE_H_M = 1.78                         # wooden poles, floor to top
POLE_OFFSET_M = 0.06                    # pole centre beyond the board's side edge
POLE_R_M = 0.03                         # pole radius, with the rails
OBSTACLE_INFLATE_M = 0.075 + 0.10       # drone radius + margin, as in P2b
LINE_STEP_M = 0.02                      # line check sampling


def gate_axes(g):
    """Unit axes of a gate: n (through the opening), u (up, in the board plane), r (across)."""
    n = np.asarray(g.n, float) / np.linalg.norm(g.n)
    u = np.array([0.0, 0.0, 1.0]) - n[2] * n
    u /= np.linalg.norm(u)
    return n, u, np.cross(u, n)


def obstacle_hits(points, g, inflate=OBSTACLE_INFLATE_M):
    """How many of the (N, 3) points are inside gate g's inflated board or poles."""
    P = np.atleast_2d(np.asarray(points, float))
    n, u, r = gate_axes(g)
    d = P - np.asarray(g.c, float)
    s, a, b = d @ n, d @ r, d @ u               # through, across, up (from the centre)
    half = 0.5 * BOARD_M
    board = (np.abs(s) < inflate) & (np.abs(a) < half + inflate) & (np.abs(b) < half + inflate)
    pole_a = half + POLE_OFFSET_M
    poles = ((np.abs(s) < POLE_R_M + inflate) & (np.abs(np.abs(a) - pole_a) < POLE_R_M + inflate)
             & (P[:, 2] < POLE_H_M + inflate))
    return int(np.count_nonzero(board | poles))


def line_hits(p0, p1, obstacles, step=LINE_STEP_M):
    """Samples of the straight line p0 -> p1 that fall inside any obstacle (0 = clear)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    k = max(2, int(math.ceil(np.linalg.norm(p1 - p0) / step)) + 1)
    P = p0 + np.linspace(0.0, 1.0, k)[:, None] * (p1 - p0)
    return sum(obstacle_hits(P, g) for g in obstacles)


# --------------------------------------------------------------------------- #
# The modes. update() returns (world velocity, yaw rate, next mode name or None).
# --------------------------------------------------------------------------- #
class Survey:
    name = "survey"

    def start(self, r):
        s = r.state
        self.hold = np.array(s.position_m, float)
        self.targets = [wrap_pi(s.yaw_rad + SWEEP_RAD), wrap_pi(s.yaw_rad - SWEEP_RAD)]
        self.k = 0
        self.face = None
        r.ctrl.reset()

    def update(self, r):
        s = r.state
        r.gates.update(r.seen)
        v = r.velocity_to(self.hold, s)
        if self.face is None and self.k < len(self.targets):          # sweeping
            tgt = self.targets[self.k]
            if r.yaw_ctrl.arrived(s.yaw_rad, tgt):
                self.k += 1
            return v, r.yaw_ctrl.step(s.yaw_rad, tgt, max_rate=SWEEP_RATE), None
        if self.face is None:                                          # sweep done
            g = r.gates.select(s.position_m)
            if g is None:
                self.k = 0                         # nothing tracked yet: sweep again
                return v, 0.0, None
            r.set_target(g)
            self.face = heading_of(g.c - self.hold)
        if r.yaw_ctrl.arrived(s.yaw_rad, self.face, tol=math.radians(3.0)):
            return v, 0.0, "view"
        return v, r.yaw_ctrl.step(s.yaw_rad, self.face, max_rate=SWEEP_RATE), None


class View:
    name = "view"

    def start(self, r):
        s, g = r.state, r.target
        r.view_tries += 1
        dists = (r.d_view + BACKOFF_M,) if r.backoff_active else VIEW_DISTANCES
        p = np.asarray(s.position_m, float)
        obstacles = [x for x in r.gates.gates if x.status == "passed"]
        self.vp, fallback = None, []
        for d in dists:
            cand = g.c - d * g.n
            cand[2] = g.c[2]                       # at the gate centre's height
            hits = line_hits(p, cand, obstacles)
            if in_box(cand) and hits == 0:         # inside the box, line clear of passed windows
                self.vp, r.d_view = cand, d
                break
            fallback.append((hits, d, clamp_box(cand)))
        if self.vp is None:                        # nothing clear: the least-blocked one, clamped
            fallback = [(line_hits(p, c, obstacles), d, c) for _, d, c in fallback]
            hits, r.d_view, self.vp = min(fallback, key=lambda f: (f[0], -f[1]))
            r.log("no clear viewpoint for gate %d: using d %.1f m, %d line samples blocked"
                  % (g.id, r.d_view, hits))
        elif r.d_view != dists[0]:
            r.log("viewpoint for gate %d at d %.1f m (longer ones blocked or outside the box)"
                  % (g.id, r.d_view))
        self.traj = Quintic(s.position_m, self.vp)
        self.t0, self.yaw_hold = s.t_s, s.yaw_rad
        self.heading = heading_of(g.n)
        self.turning = False
        r.ctrl.reset()

    def update(self, r):
        s = r.state
        t = s.t_s - self.t0
        if not self.turning:                       # fly there, yaw held
            p_d, v_d = self.traj.sample(t)
            v = r.velocity_to(p_d, s, v_d)
            if self.traj.done(t, s.position_m):
                self.turning = True
                r.ctrl.reset()
            return v, r.yaw_ctrl.step(s.yaw_rad, self.yaw_hold), None
        v = r.velocity_to(self.vp, s)              # on arrival: turn head-on
        if r.yaw_ctrl.arrived(s.yaw_rad, self.heading):
            return v, 0.0, "measure"
        return v, r.yaw_ctrl.step(s.yaw_rad, self.heading), None


class Measure:
    name = "measure"

    def start(self, r):
        self.hold = np.array(r.state.position_m, float)
        self.heading = heading_of(r.target.n)
        self.k = 0
        self.matched, self.valid = [], []
        r.ctrl.reset()

    def match(self, r, g):
        """The detection that is the target: the false-gate check (1, 2, 3)."""
        best, best_d = None, MATCH_DIST_M
        for d in r.seen:
            dist = float(np.linalg.norm(d["c"] - g.c))
            if (dist < best_d and detector.angle_deg(d["n"], g.n) < MATCH_ANGLE_DEG
                    and abs(d["dist"] - r.d_view) < MATCH_DEPTH_TOL_M):
                best, best_d = d, dist
        return best

    def update(self, r):
        s, g = r.state, r.target
        v = r.velocity_to(self.hold, s)
        w = r.yaw_ctrl.step(s.yaw_rad, self.heading)
        self.k += 1
        tgt = self.match(r, g)
        r.gates.update([d for d in r.seen if d is not tgt])   # the target starts fresh
        if self.k <= MEASURE_SKIP:
            return v, w, None
        if tgt is not None:
            self.matched.append(tgt)
            if tgt["hole_closed"]:
                self.valid.append(tgt)
        if self.k < MEASURE_SKIP + MEASURE_TICKS:
            return v, w, None
        return v, w, self.decide(r, g)

    @staticmethod
    def fuse(dets, use_centre):
        C = np.array([d["centre"] if use_centre and d["centre"] is not None else d["c"]
                      for d in dets])
        N = np.array([d["n"] for d in dets])
        N[N[:, 0] < 0] *= -1.0
        n = N.mean(axis=0)                 ###might have to change (mean of normals)
        return np.median(C, axis=0), n / np.linalg.norm(n)

    def decide(self, r, g):
        p = np.asarray(r.state.position_m, float)
        if len(self.valid) >= MEASURE_VALID_MIN:
            g.c, g.n = self.fuse(self.valid, use_centre=True)
            line = g.c - p
            ang = math.degrees(math.acos(float(np.clip(line @ g.n / np.linalg.norm(line), -1, 1))))
            if ang <= PASS_MAX_ANGLE_DEG:
                g.status = "measured"
                r.log("measured gate %d at %s, n %s, pass line %.0f deg"
                      % (g.id, np.round(g.c, 3), np.round(g.n, 3), ang))
                return "pass"
            r.log("pass line %.0f deg off the normal: view again" % ang)
            return r.retry(g)
        if not self.matched:                       # nothing there: a false gate
            g.status = "false"
            r.log("gate %d not found in measure: false" % g.id)
            return r.next_target()
        g.c, g.n = self.fuse(self.matched, use_centre=True)   # re-aim at the fresh estimate
        r.log("closed hole in %d/%d ticks: view again" % (len(self.valid), MEASURE_TICKS))
        return r.retry(g)


class Pass:
    name = "pass"

    def start(self, r):
        s, g = r.state, r.target
        p0 = np.array(s.position_m, float)
        u = g.c - p0
        u /= np.linalg.norm(u)
        self.end = g.c + PASS_EXIT_M * u
        self.traj = Quintic(p0, self.end)
        self.t0, self.yaw_hold = s.t_s, s.yaw_rad
        self.crossed = False
        r.ctrl.reset()

    def update(self, r):
        s, g = r.state, r.target
        t = s.t_s - self.t0
        p_d, v_d = self.traj.sample(t)
        v = r.velocity_to(p_d, s, v_d)
        w = r.yaw_ctrl.step(s.yaw_rad, self.yaw_hold)
        if not self.crossed and float(g.n @ (np.asarray(s.position_m, float) - g.c)) > 0:
            self.crossed = True
            g.status = "passed"
            r.passed += 1
            r.log("crossed gate %d (%d/%d)" % (g.id, r.passed, N_WINDOWS))
        if not self.traj.done(t, s.position_m):
            return v, w, None
        if not self.crossed:                       # arrived without crossing: look again
            g.status = "tracked"
            return v, w, r.retry(g)
        if r.passed >= N_WINDOWS:
            return v, w, "done"
        return v, w, r.next_target()


class Done:
    name = "done"

    def start(self, r):
        self.hold = np.array(r.state.position_m, float)
        r.ctrl.reset()

    def update(self, r):
        return r.velocity_to(self.hold, r.state), 0.0, None


class WindowRacer(Policy):
    """Detector + state machine + your P2b controller. Velocity out."""

    name = "p3_window_racer"
    # Once your network is fixed at one resolution, set this to your contract's
    # PolicyContract.load("p3_contract.json").observation.sensor.fingerprint()
    # and the runtime will refuse to fly it through any other camera.
    sensor_fingerprint = None

    def __init__(self, record_fpv=True):
        ######## TODO - LOAD YOUR NETWORK ########
        self.net, self.device = detector.load_model(
            os.path.join(HERE, "weights", "best.torchscript"))
        # TorchScript optimises on its first calls: pay for that here, not in flight.
        blank = detector.rgb01_to_bgr8(np.zeros((213, 320, 3), np.float32))
        for _ in range(3):
            detector.predict(self.net, self.device, blank, imgsz=320)
        self.rng = np.random.default_rng(0)
        ######## END TODO ########
        self.fpv = FpvRecorder() if record_fpv else None
        self.K = None
        self.reset()

    def reset(self):
        """Called once before each run. Reset the state machine here."""
        self.mode = "survey"
        self.passed = 0
        self.window = None          # your last estimate of the window you are flying to
        self.gates = detector.GateList()
        # max_speed above the 0.8 m/s quintic cap leaves room for the P correction
        self.ctrl = PositionController(max_speed=1.0, max_climb=0.6)
        self.yaw_ctrl = YawController(kp=2.0, max_rate=math.radians(45.0))
        self.modes = {m.name: m for m in (Survey(), View(), Measure(), Pass(), Done())}
        self.cur = None
        self.state, self.seen = None, []
        self.target, self.view_tries, self.backoff_active = None, 0, False
        self.d_view = VIEW_DISTANCES[0]

    # -- perception ----------------------------------------------------------
    def detect(self, rgb, depth_m):
        """The windows in the image, or None.

        Returns {"mask": union of the masks kept, "corners": hole centroids
        (u, v) of the closed ones, "dets": per-mask image detections}; the FPV
        recorder draws "mask" and "corners".
        """
        ######## TODO - YOUR NETWORK ########
        res = detector.predict(self.net, self.device, detector.rgb01_to_bgr8(rgb), imgsz=320)
        dets = detector.analyse_masks(res)
        if not dets:
            return None
        corners = [d["hole_uv"] for d in dets if d["hole_closed"]]
        return {"mask": np.any([d["mask"] for d in dets], axis=0),
                "corners": corners or None, "dets": dets}
        ######## END TODO ########

    def locate(self, det, depth_m, state):
        """Detection + depth -> where the windows are in the world: a list of
        {c, n, n_inl, hole_closed, centre, dist}, pieces of one window joined."""
        ######## TODO - WHERE IS THE WINDOW ########
        return detector.locate_all(det["dets"], depth_m, self.K,
                                   lambda p: camera_to_world(p, state),
                                   state.position_m, self.rng)
        ######## END TODO ########

    # -- control -------------------------------------------------------------
    def velocity_to(self, target_world, state, velocity_ff=(0.0, 0.0, 0.0)):
        """Your P2b position loop: a world target -> a world velocity in m/s."""
        ######## TODO - YOUR P2B CONTROLLER ########
        return self.ctrl.step(state.position_m, target_world, velocity_ff)
        ######## END TODO ########

    # -- state machine helpers -------------------------------------------------
    def log(self, msg):
        print("%6.2f s  [%s] %s" % (self.state.t_s, self.mode, msg))

    def switch(self, name):
        if self.cur is not None:
            self.log("-> %s" % name)
        self.mode = name
        self.cur = self.modes[name]
        self.cur.start(self)

    def set_target(self, g):
        self.target, self.view_tries, self.backoff_active = g, 0, False
        self.d_view = VIEW_DISTANCES[0]
        self.log("target gate %d at %s" % (g.id, np.round(g.c, 3)))

    def next_target(self):
        g = self.gates.select(self.state.position_m)
        if g is None:
            self.target = None
            return "survey"
        self.set_target(g)
        return "view"

    def retry(self, g):
        if self.view_tries < MAX_VIEWS:
            return "view"
        if not self.backoff_active:
            self.backoff_active = True
            return "view"
        g.status = "false"
        self.log("gate %d: no clean measurement after backing off: false" % g.id)
        return self.next_target()

    # -- the state machine -----------------------------------------------------
    def act(self, obs, state) -> Action:
        if self.K is None:
            h, w = obs.depth_m.shape
            self.K = camera_matrix(w, h)

        det = self.detect(obs.rgb, obs.depth_m)
        seen = None if det is None else self.locate(det, obs.depth_m, state)

        v_world, yaw_rate = np.zeros(3), 0.0
        ######## TODO - THE STATE MACHINE ########
        self.state, self.seen = state, (seen or [])
        if self.cur is None:
            self.switch("survey")
        v_world, yaw_rate, nxt = self.cur.update(self)
        if nxt is not None and nxt != self.mode:
            self.switch(nxt)
        self.window = self.target
        ######## END TODO ########

        if self.fpv is not None:
            self.fpv.add(obs.rgb, obs.depth_m,
                         mask=None if det is None else det.get("mask"),
                         points=None if det is None else det.get("corners"),
                         mode=self.mode)
        return Action("velocity", "body_flu", world_to_body(v_world, state.yaw_rad),
                      yaw_rate_rad_s=float(yaw_rate))


class HoverHere(Policy):
    """Zero velocity, camera recorded. Fly it first each session to check the loop."""

    name = "p3_hover"
    sensor_fingerprint = None

    def __init__(self):
        self.fpv = FpvRecorder()

    def act(self, obs, state) -> Action:
        self.fpv.add(obs.rgb, obs.depth_m)
        return Action("velocity", "body_flu", np.zeros(3))
