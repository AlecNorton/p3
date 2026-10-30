"""The splat rendered on the CPU, behind splat_hitl's renderer interface.

    from cpu_renderer import CpuSplatRenderer
    renderer = CpuSplatRenderer(contract.observation.sensor, SceneBundle.load("scene"))
    obs = renderer.render(position_m, body_rpy)      # scene metres, radians

Same camera and depth rules as the GPU worker: depth in metres along the camera
axis, empty pixels at the sensor's empty depth (4.0 m), black background. Needs
only numpy, and takes a second or more per frame. Use the GPU worker when you
can (install.md).
"""
import math
import os
import sys
import time

import numpy as np

from splat_hitl.frames import rpy_to_matrix
from splat_hitl.renderer import Observation, RendererClient

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from splat_cpu import Camera, load_splat, quat_to_R, render  # noqa: E402

# Camera axes (x right, y down, z forward) as columns, in the body frame (x forward, y left, z up).
CAM_TO_BODY = np.array([[0.0, 0.0, 1.0],
                        [-1.0, 0.0, 0.0],
                        [0.0, -1.0, 0.0]])


class _SceneMetres:
    """The splat's Gaussians in scene metres: splat file units times metres_per_unit."""

    def __init__(self, splat_path, metres_per_unit):
        pos, scl, rgba, q = load_splat(splat_path)
        M = quat_to_R(q) * scl[:, None, :]
        s = float(metres_per_unit)
        self.mean = pos * s
        self.cov = (M @ np.transpose(M, (0, 2, 1))) * s * s
        self.rgb = rgba[:, :3]
        self.opacity = rgba[:, 3]


class CpuSplatRenderer(RendererClient):
    """RendererClient for a scene bundle, rendered with numpy."""

    def __init__(self, sensor, bundle):
        super().__init__(sensor)
        tf = bundle.transform()
        mpu = tf.metres_per_unit if tf is not None else bundle.esdf().metres_per_unit
        self.scene = _SceneMetres(bundle.path("splat"), mpu)
        self.focal_px = 0.5 * sensor.width / math.tan(math.radians(sensor.fov_x_deg) / 2.0)
        self.empty_depth_m = float(sensor.depth.empty_depth_m)

    def render(self, position_m, body_rpy):
        t0 = time.perf_counter()
        R_world_cam = rpy_to_matrix(*self.camera_rpy(body_rpy)) @ CAM_TO_BODY
        cam = Camera(self.sensor.width, self.sensor.height,
                     np.asarray(position_m, float).reshape(3), R_world_cam.T,
                     focal_px=self.focal_px)
        rgb, acc, depth = render(self.scene, cam)
        depth = np.where(acc > 1e-4, depth, self.empty_depth_m)
        return Observation(depth.astype(np.float32), rgb.astype(np.float32),
                           time.perf_counter() - t0, self.sensor.fingerprint())
