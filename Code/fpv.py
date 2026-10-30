"""Records what your policy saw, with your detections drawn on, for FPVVideo.mp4.

    self.fpv = FpvRecorder()                                   # in __init__
    self.fpv.add(obs.rgb, obs.depth_m, mask=mask, points=pts, mode=self.mode)

Frames are kept in memory and written when the run ends, so recording costs
almost nothing in the loop. Each frame is one PNG: RGB | depth | detections.
Make the video with ffmpeg (on any machine):

    ffmpeg -framerate 15 -i fpv/<run>/frame_%05d.png -vf scale=iw*2:ih*2:flags=neighbor \
        -pix_fmt yuv420p FPVVideo.mp4

Needs only numpy and the standard library, so it also runs in the lab container.
"""
import atexit
import os
import struct
import time
import zlib

import numpy as np

#: colour of the strip along the top of the detection panel, per state
MODE_COLOURS = {"search": (128, 128, 128), "align": (230, 200, 40),
                "approach": (40, 120, 230), "through": (40, 190, 80),
                "done": (255, 255, 255)}


def write_png(path, img):
    """(H, W, 3) uint8 -> PNG, with zlib alone."""
    img = np.ascontiguousarray(img, dtype=np.uint8)
    h, w = img.shape[:2]
    raw = b"".join(b"\x00" + img[y].tobytes() for y in range(h))

    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    with open(path, "wb") as fh:
        fh.write(b"\x89PNG\r\n\x1a\n")
        fh.write(chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)))
        fh.write(chunk(b"IDAT", zlib.compress(raw, 1)))
        fh.write(chunk(b"IEND", b""))


def depth_colour(depth_m, far_m=4.0):
    """Depth -> red (near) ... blue (far), uint8 (H, W, 3)."""
    t = np.clip(np.nan_to_num(np.asarray(depth_m, float), nan=far_m) / far_m, 0.0, 1.0)
    stops = np.array([[255, 0, 0], [255, 255, 0], [0, 255, 255], [0, 0, 255]], float)
    x = t * (len(stops) - 1)
    i = np.minimum(x.astype(int), len(stops) - 2)
    f = (x - i)[..., None]
    return ((1 - f) * stops[i] + f * stops[i + 1]).astype(np.uint8)


class FpvRecorder:
    """Keeps up to `max_frames` frames (2 min at 15 Hz) and writes them once."""

    def __init__(self, out_dir="fpv", max_frames=1800, far_m=4.0):
        self.out_dir = out_dir
        self.max_frames = int(max_frames)
        self.far_m = float(far_m)
        self.frames = []
        self.saved = None
        atexit.register(self.save)          # ros_node has no end-of-run call

    def add(self, rgb, depth_m, mask=None, points=None, mode=None):
        """rgb (H, W, 3) in [0, 1]; mask (H, W) bool; points [(u, v), ...] in pixels."""
        if len(self.frames) >= self.max_frames:
            return
        rgb8 = (np.clip(np.asarray(rgb, float), 0.0, 1.0) * 255 + 0.5).astype(np.uint8)
        d8 = (np.clip(np.nan_to_num(np.asarray(depth_m, float), nan=self.far_m),
                      0.0, self.far_m) / self.far_m * 255 + 0.5).astype(np.uint8)
        m = None if mask is None else np.packbits(np.asarray(mask, bool))
        pts = None if points is None else np.asarray(points, float).reshape(-1, 2)
        self.frames.append((rgb8, d8, m, pts, mode))

    def _panel(self, rgb8, d8, m, pts, mode):
        h, w = rgb8.shape[:2]
        depth = depth_colour(d8.astype(float) / 255.0 * self.far_m, self.far_m)
        det = rgb8.copy()
        if m is not None:
            mask = np.unpackbits(m)[:h * w].reshape(h, w).astype(bool)
            det[mask] = (0.5 * det[mask] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
        if pts is not None:
            for u, v in pts:
                if np.isfinite(u) and np.isfinite(v) and 0 <= u < w and 0 <= v < h:
                    u, v = int(u), int(v)
                    det[max(v - 2, 0):v + 3, max(u - 2, 0):u + 3] = (255, 0, 255)
        if mode in MODE_COLOURS:
            det[:4] = MODE_COLOURS[mode]
        return np.concatenate([rgb8, depth, det], axis=1)

    def save(self):
        """Writes the frames once; later calls return the same folder."""
        if self.saved is not None or not self.frames:
            return self.saved
        d = os.path.join(self.out_dir, time.strftime("%Y%m%d-%H%M%S"))
        os.makedirs(d, exist_ok=True)
        for i, f in enumerate(self.frames):
            write_png(os.path.join(d, "frame_%05d.png" % i), self._panel(*f))
        self.saved = d
        print("fpv: %d frames in %s" % (len(self.frames), d))
        return d
