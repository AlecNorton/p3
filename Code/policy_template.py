#!/usr/bin/env python3
"""Your P3 policy: find a window, line up with it, fly through it, repeat.
Fill in the TODOs.

    python3 Code/Wrapper.py                                   # in the simulator
    ros_node ... --policy policy_template:WindowRacer          # on the drone

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
"""
import math
import os

import numpy as np

from splat_hitl.commands import Action
from splat_hitl.policy import Policy
import torch_model
import cv2

from fpv import FpvRecorder

HERE = os.path.dirname(os.path.abspath(__file__))
N_WINDOWS = 3                          # the course has three windows
FOV_X_DEG = 60.656255852364644         # fixed: the drone camera's field of view


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

def calc_normal_vector(p1, p2, p3):
    """Calculate normal vector of triangle

    Parameters
    ----------
    p1 : np.array
        3D point
    p2 : np.array
        3D point
    p3 : np.array
        3D point

    Returns
    -------
    normal_vector : np.array
        normal vector (3,)

    """
    # calculate two vectors of triangle
    v1 = p2 - p1
    v2 = p3 - p1

    # calculate normal vector
    normal_vector = np.cross(v1, v2)

    # normalize vector
    normal_vector = normal_vector / np.linalg.norm(normal_vector)

    return normal_vector

def distance_to_plane(point, normal, origin):
    dot_product = np.dot(normal, point) - np.dot(normal, origin)
    if np.isclose(dot_product, 0):
        return 0.0
    else:
        distance = abs(dot_product) / np.linalg.norm(normal)
        return distance
    
def ransac_normal_vector_estimation(points_3d, inlier_radio_th = 0.7, inlier_dist = 0.1, p = 0.99):
    center = np.mean(points_3d, axis=0)

    max_iter = int(np.floor(np.log(1.0-p)/np.log(1.0-(1.0-inlier_radio_th)**3)))

    for ite in range(max_iter):
        # Random sampling
        sampled_ids = np.random.choice(points_3d.shape[0], size=3,
                                       replace=False)
        sampled_points = points_3d[sampled_ids, :]
        p1 = sampled_points[0, :]
        p2 = sampled_points[1, :]
        p3 = sampled_points[2, :]
        normal_vector = calc_normal_vector(p1, p2, p3)

        # calc inlier ratio
        n_inliner = 0
        for i in range(points_3d.shape[0]):
            p = points_3d[i, :]
            if distance_to_plane(p, normal_vector, center) <= inlier_dist:
                n_inliner += 1
        inlier_ratio = n_inliner / points_3d.shape[0]
        print(f"Iter:{ite}, {inlier_ratio=}")
        if inlier_ratio > inlier_radio_th:
            return center, normal_vector

    return center, None

def wrap_pi(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class WindowRacer(Policy):
    """Detector + state machine + your P2b controller. Velocity out."""

    name = "p3_window_racer"
    # Once your network is fixed at one resolution, set this to your contract's
    # PolicyContract.load("p3_contract.json").observation.sensor.fingerprint()
    # and the runtime will refuse to fly it through any other camera.
    sensor_fingerprint = None

    def __init__(self, record_fpv=True):
        ######## TODO - LOAD YOUR NETWORK ########
        # Keep the weights next to this file and load them by a path built from
        # HERE, e.g. onnxruntime.InferenceSession(os.path.join(HERE, "weights",
        # "window.onnx")). Load once, here: never inside act().
        self.net, self.device = torch_model.load_model('best.torchscript', 'cpu')
        ######## END TODO ########
        self.fpv = FpvRecorder() if record_fpv else None
        self.K = None
        self.reset()

    def reset(self):
        """Called once before each run. Reset the state machine here."""
        self.mode = "search"
        self.passed = 0
        self.window = None          # your last estimate of the window you are flying to

    # -- perception ----------------------------------------------------------
    def detect(self, rgb, depth_m):
        """The closest window in the image, or None.

        Return a dict with whatever your network gives, e.g.
        {"mask": (H, W) bool, "corners": (4, 2) pixel array}; the FPV recorder
        draws "mask" and "corners" if present.
        """
        det = torch_model.predict(self.net, self.device, rgb, 320)
        return torch_model.closest(rgb, depth_m, det)

    def locate(self, det, depth_m, state):
        """Detection + depth -> where the window is in the world, or None."""

        #Back projecting. 
        mask = det['mask'].astype('uint8')*255
        pixels = np.nonzero(mask)
        x, y  = pixels[1], pixels[0]
        world_points = []
        for i in range(0, len(x)-1):
            u = x[i]
            v = y[i]
            z = depth_m[v, u]
            world_points.append(camera_to_world(pixel_to_camera(u, v, z, self.K), state))

        world_points = np.asarray(world_points)
        center, normal_vec = ransac_normal_vector_estimation(world_points)
        # Back-project the window's pixels with their depths (pixel_to_camera),
        # move them to the world (camera_to_world) and find the opening's centre.
        # A plane through the board's points gives the direction it faces.
        return center, normal_vec
        ######## END TODO ########

    # -- control -------------------------------------------------------------
    def velocity_to(self, target_world, state):
        """Your P2b position loop: a world target -> a world velocity in m/s."""
        ######## TODO - YOUR P2B CONTROLLER ########
        return np.zeros(3)
        ######## END TODO ########

    # -- the state machine -----------------------------------------------------
    def act(self, obs, state) -> Action:
        if self.K is None:
            h, w = obs.depth_m.shape
            self.K = camera_matrix(w, h)
        
        print
        det = self.detect(obs.rgb, obs.depth_m)
        seen = None if det is None else self.locate(det, obs.depth_m, state)

        v_world, yaw_rate = np.zeros(3), 0.0
        ######## TODO - THE STATE MACHINE ########
        # Keep self.window on the window you are flying to: update it from
        # `seen` only when `seen` is that window, and not at all once committed.
        # search    no window in view: turn on the spot until one is
        # align     face it, and move to a point in front of its opening
        # approach  fly at the opening, keeping it centred
        # through   commit: carry on straight until you are past it -- it fills
        #           and then leaves the image before you get there
        # done      after N_WINDOWS: hover
        # Change self.mode when a step is done; count windows in self.passed.
        if self.mode == "search":
            yaw_rate = 0.3
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
