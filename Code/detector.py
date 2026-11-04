"""Window detector: YOLO26-seg masks -> planes and centres -> a world-frame gate list.

Same structure as torch_model.py (pre-processing, model loading, inference),
with the decisions made for P3 after it:

    image  (per mask, no pose needed)
        confidence >= 0.75, erode 2 px, hole test on the mask's contours
    world  (needs depth and the drone's pose)
        back-project the eroded mask, RANSAC plane (2 cm, 20 tries), centroid and
        normal from the inliers; hole centroid -> ray -> plane = the window centre
        join pieces of one window: centroids < 0.68 m, normals < 15 deg, coplanar
        < 0.1 m; refit the plane on the joined points
    gate list  (across frames)
        match within 0.5 m and 20 deg, weighted centroid and slerp, weight capped at
        25 000 inliers; untracked -> tracked after more than 10 hits

predict() and letterbox() are taken from torch_model.py unchanged except for the
default confidence.
"""
import math
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision

HERE = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------------------------- #
# Parameters (the decisions)
# --------------------------------------------------------------------------- #
CONF_MIN = 0.75              # detection confidence
ERODE_PX = 2                 # erosion before back-projection: drops blended edge depths
DEPTH_MIN_M = 0.05           # renderer near plane
DEPTH_MAX_M = 3.99           # 4.0 means nothing was hit
RANSAC_DIST_M = 0.02
RANSAC_ITERS = 20
RANSAC_SCORE_PTS = 4000      # hypotheses are scored on at most this many points (speed)
HOLE_MIN_FRAC = 0.15         # hole area > 15 % of the outer outline's area
BOARD_W_M = 0.91
MERGE_DIST_M = 0.75 * BOARD_W_M
MERGE_ANGLE_DEG = 15.0
MERGE_PLANE_M = 0.10
TRACK_DIST_M = 0.5
TRACK_ANGLE_DEG = 20.0
TRACK_HITS = 10              # tracked once hits > TRACK_HITS
WEIGHT_CAP = 25000


# --------------------------------------------------------------------------- #
# Pre-processing (from torch_model.py)
# --------------------------------------------------------------------------- #
def letterbox(img, size, color=114):
    """Resize with aspect ratio preserved and pad to (size, size).

    Returns the padded image, scale r, and (left, top) padding.
    """
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = round(h * r), round(w * r)
    if (nh, nw) != (h, w):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas = np.full((size, size, 3), color, dtype=np.uint8)
    canvas[top:top + nh, left:left + nw] = img
    return canvas, r, left, top


def rgb01_to_bgr8(rgb):
    """obs.rgb (H, W, 3) float in [0, 1], RGB -> uint8 BGR, what predict() expects."""
    rgb8 = (np.clip(np.asarray(rgb, np.float32), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    return cv2.cvtColor(rgb8, cv2.COLOR_RGB2BGR)


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def load_model(path, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(path, map_location=device).eval()
    return model, device


# --------------------------------------------------------------------------- #
# Inference (from torch_model.py)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict(model, device, image_bgr, imgsz=320, names=None, nc=1, nm=32,
            conf_thres=CONF_MIN, iou_thres=0.45, mask_thres=0.5, half=False):
    """Run segmentation inference on a BGR uint8 image.

    Returns a dict:
        "masks":  (M, H, W) bool ndarray, one mask per detection, original image size
        "labels": list[str] of length M
        "scores": (M,) float32 ndarray of confidences
        "boxes":  (M, 4) float32 ndarray, xyxy in original image pixels
    """
    names = names or [str(i) for i in range(nc)]
    H, W = image_bgr.shape[:2]

    img, r, dx, dy = letterbox(image_bgr, imgsz)
    x = torch.from_numpy(img[:, :, ::-1].copy()).permute(2, 0, 1)   # BGR->RGB, CHW
    x = x.unsqueeze(0).float().div(255).to(device)
    if half:
        x = x.half()

    out = model(x)
    if not isinstance(out, (list, tuple)):
        raise RuntimeError("Model returned a single tensor; a segmentation export "
                           "should also return mask prototypes.")
    pred, protos = out[0], out[1]
    pred, protos = pred.float()[0], protos.float()[0]

    if pred.shape[0] == 4 + nc + nm:
        pred = pred.transpose(0, 1)
        cxcywh, cls_scores, coeffs = pred[:, :4], pred[:, 4:4 + nc], pred[:, 4 + nc:]
        scores, cls = cls_scores.max(1)
        keep = scores > conf_thres
        cxcywh, scores, cls, coeffs = cxcywh[keep], scores[keep], cls[keep], coeffs[keep]
        xyxy = torch.empty_like(cxcywh)
        xyxy[:, 0] = cxcywh[:, 0] - cxcywh[:, 2] / 2
        xyxy[:, 1] = cxcywh[:, 1] - cxcywh[:, 3] / 2
        xyxy[:, 2] = cxcywh[:, 0] + cxcywh[:, 2] / 2
        xyxy[:, 3] = cxcywh[:, 1] + cxcywh[:, 3] / 2
        idx = torchvision.ops.nms(xyxy, scores, iou_thres)
        xyxy, scores, cls, coeffs = xyxy[idx], scores[idx], cls[idx], coeffs[idx]
    elif pred.shape[-1] == 6 + nm:
        keep = pred[:, 4] > conf_thres
        pred = pred[keep]
        xyxy, scores, cls, coeffs = pred[:, :4], pred[:, 4], pred[:, 5].long(), pred[:, 6:]
    else:
        raise RuntimeError(f"Unexpected prediction shape {tuple(pred.shape)} "
                           f"for nc={nc}, nm={nm}")

    if len(scores) == 0:
        return {"masks": np.zeros((0, H, W), bool), "labels": [],
                "scores": np.zeros(0, np.float32), "boxes": np.zeros((0, 4), np.float32)}

    c, mh, mw = protos.shape
    m = torch.sigmoid(coeffs @ protos.reshape(c, -1)).reshape(-1, mh, mw)
    m = F.interpolate(m[None], size=(imgsz, imgsz), mode="bilinear", align_corners=False)[0]
    ys = torch.arange(imgsz, device=m.device)[None, :, None]
    xs = torch.arange(imgsz, device=m.device)[None, None, :]
    b = xyxy[:, :, None, None]
    inside = (xs >= b[:, 0]) & (xs <= b[:, 2]) & (ys >= b[:, 1]) & (ys <= b[:, 3])
    m = ((m * inside) > mask_thres).cpu().numpy().astype(np.uint8)

    nh, nw = round(H * r), round(W * r)
    masks = np.zeros((len(m), H, W), dtype=bool)
    for i, mk in enumerate(m):
        mk = mk[dy:dy + nh, dx:dx + nw]
        masks[i] = cv2.resize(mk, (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)

    boxes = xyxy.clone().cpu().numpy().astype(np.float32)
    boxes[:, [0, 2]] = np.clip((boxes[:, [0, 2]] - dx) / r, 0, W)
    boxes[:, [1, 3]] = np.clip((boxes[:, [1, 3]] - dy) / r, 0, H)
    return {"masks": masks, "labels": [names[int(k)] for k in cls.cpu().tolist()],
            "scores": scores.cpu().numpy().astype(np.float32), "boxes": boxes}


# --------------------------------------------------------------------------- #
# Image-space analysis: erosion and the hole test
# --------------------------------------------------------------------------- #
def hole_info(mask):
    """Is the window's opening enclosed by this mask?

    The largest top-level outline is the window; its largest child outline is
    the hole. Closed if that hole's area > HOLE_MIN_FRAC of the outer outline's.
    An L, C or U shape has no child outline. Returns (closed, (u, v) centroid of
    the hole in pixel coordinates (pixel centres at +0.5), hole contour or None).
    """
    m = np.ascontiguousarray(mask, dtype=np.uint8)
    cnts, hier = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if not cnts or hier is None:
        return False, None, None
    hier = hier[0]
    tops = [i for i in range(len(cnts)) if hier[i][3] < 0]
    if not tops:
        return False, None, None
    outer = max(tops, key=lambda i: cv2.contourArea(cnts[i]))
    kids = [i for i in range(len(cnts)) if hier[i][3] == outer]
    if not kids:
        return False, None, None
    hole = max(kids, key=lambda i: cv2.contourArea(cnts[i]))
    a_out, a_hole = cv2.contourArea(cnts[outer]), cv2.contourArea(cnts[hole])
    if a_out <= 0 or a_hole <= HOLE_MIN_FRAC * a_out:
        return False, None, None
    M = cv2.moments(cnts[hole])
    if M["m00"] <= 0:
        return False, None, None
    uv = (M["m10"] / M["m00"] + 0.5, M["m01"] / M["m00"] + 0.5)
    return True, uv, cnts[hole]


def analyse_masks(result, conf_min=CONF_MIN, erode_px=ERODE_PX):
    """predict() output -> one image-space detection per mask kept."""
    k = np.ones((2 * erode_px + 1, 2 * erode_px + 1), np.uint8)
    dets = []
    for mask, score in zip(result["masks"], result["scores"]):
        if score < conf_min or not mask.any():
            continue
        closed, uv, _ = hole_info(mask)
        dets.append({"mask": mask.astype(bool), "score": float(score),
                     "eroded": cv2.erode(mask.astype(np.uint8), k).astype(bool),
                     "hole_closed": closed, "hole_uv": uv})
    return dets


# --------------------------------------------------------------------------- #
# World-space geometry
# --------------------------------------------------------------------------- #
def pixel_to_camera(u, v, depth_m, K):
    """Same as policy_template.pixel_to_camera: camera frame x right, y down, z forward."""
    u, v, z = (np.asarray(a, float) for a in (u, v, depth_m))
    return np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], axis=-1)


def backproject(eroded, depth_m, K, to_world):
    """Eroded mask pixels with a real depth -> (N, 3) world points."""
    valid = eroded & (depth_m > DEPTH_MIN_M) & (depth_m < DEPTH_MAX_M) & np.isfinite(depth_m)
    vs, us = np.nonzero(valid)
    if len(us) == 0:
        return np.zeros((0, 3))
    return to_world(pixel_to_camera(us + 0.5, vs + 0.5, depth_m[vs, us], K))


def fit_plane_ransac(P, rng, dist=RANSAC_DIST_M, iters=RANSAC_ITERS):
    """RANSAC plane, then centroid and normal refit on the inliers (SVD).

    Returns (centroid, unit normal with n_x >= 0, inlier boolean mask) or None.
    """
    N = len(P)
    if N < 3:
        return None
    S = P if N <= RANSAC_SCORE_PTS else P[rng.choice(N, RANSAC_SCORE_PTS, replace=False)]
    best, best_cnt = None, -1
    for _ in range(iters):
        a, b, c = S[rng.choice(len(S), 3, replace=False)]
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n = n / nn
        cnt = int(np.count_nonzero(np.abs((S - a) @ n) < dist))
        if cnt > best_cnt:
            best, best_cnt = (a, n), cnt
    if best is None:
        return None
    a, n = best
    inl = np.abs((P - a) @ n) < dist
    if np.count_nonzero(inl) < 3:
        return None
    Q = P[inl]
    c = Q.mean(axis=0)
    n = np.linalg.svd(Q - c, full_matrices=False)[2][2]
    inl = np.abs((P - c) @ n) < dist            # inliers of the refit plane
    if np.count_nonzero(inl) < 3:
        return None
    c = P[inl].mean(axis=0)
    if n[0] < 0:                                 # flying direction: windows face back along -x
        n = -n
    return c, n, inl


def ray_plane(uv, K, to_world, cam_pos, c, n):
    """Pixel (u, v) -> ray from the camera -> where it meets the plane (c, n)."""
    d = to_world(pixel_to_camera(uv[0], uv[1], 1.0, K)) - cam_pos
    denom = float(n @ d)
    if abs(denom) < 1e-6:
        return None
    t = float(n @ (c - cam_pos)) / denom
    return None if t <= 0 else cam_pos + t * d


def angle_deg(a, b):
    """Angle between two directions, ignoring sign."""
    return math.degrees(math.acos(min(1.0, abs(float(np.dot(a, b))))))


def _world_det(mask, eroded, score, P, K, to_world, cam_pos, rng):
    fit = fit_plane_ransac(P, rng)
    if fit is None:
        return None
    c, n, inl = fit
    closed, uv, _ = hole_info(mask)
    centre = ray_plane(uv, K, to_world, cam_pos, c, n) if closed else None
    return {"mask": mask, "eroded": eroded, "score": score, "P": P,
            "c": c, "n": n, "n_inl": int(np.count_nonzero(inl)),
            "hole_closed": closed and centre is not None, "hole_uv": uv, "centre": centre,
            "dist": float(np.linalg.norm(c - cam_pos))}


def locate_all(img_dets, depth_m, K, to_world, cam_pos, rng=None):
    """Image detections -> world detections, pieces of one window joined.

    Each world detection: c (inlier centroid), n (normal, n_x >= 0), n_inl,
    hole_closed, centre (hole centroid on the plane, or None), dist (camera to c).
    """
    rng = rng if rng is not None else np.random.default_rng()
    cam_pos = np.asarray(cam_pos, float)
    out = []
    for d in img_dets:
        P = backproject(d["eroded"], depth_m, K, to_world)
        w = _world_det(d["mask"], d["eroded"], d["score"], P, K, to_world, cam_pos, rng)
        if w is not None:
            out.append(w)
    return merge_pieces(out, K, to_world, cam_pos, rng)


def merge_pieces(dets, K, to_world, cam_pos, rng):
    """Join detections that are pieces of the same window; refit after every join."""
    dets = list(dets)
    changed = True
    while changed:
        changed = False
        for i in range(len(dets)):
            for j in range(i + 1, len(dets)):
                a, b = dets[i], dets[j]
                if (np.linalg.norm(a["c"] - b["c"]) < MERGE_DIST_M
                        and angle_deg(a["n"], b["n"]) < MERGE_ANGLE_DEG
                        and abs(float(a["n"] @ (b["c"] - a["c"]))) < MERGE_PLANE_M):
                    joined = _world_det(a["mask"] | b["mask"], a["eroded"] | b["eroded"],
                                        max(a["score"], b["score"]),
                                        np.vstack([a["P"], b["P"]]), K, to_world, cam_pos, rng)
                    if joined is None:
                        continue
                    dets[i] = joined
                    del dets[j]
                    changed = True
                    break
            if changed:
                break
    return dets


# --------------------------------------------------------------------------- #
# Gate list across frames
# --------------------------------------------------------------------------- #
def slerp(a, b, t):
    """Unit vectors a -> b, fraction t. b's sign is flipped to a's side first."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a @ b < 0:
        b = -b
    om = math.acos(min(1.0, float(a @ b)))
    if om < 1e-6:
        v = (1 - t) * a + t * b
    else:
        v = (math.sin((1 - t) * om) * a + math.sin(t * om) * b) / math.sin(om)
    return v / np.linalg.norm(v)


class Gate:
    """One window in the world. status: untracked -> tracked -> measured -> passed, or false."""
    _next_id = 0

    def __init__(self, c, n, w):
        self.id = Gate._next_id
        Gate._next_id += 1
        self.c = np.asarray(c, float)
        self.n = np.asarray(n, float)
        self.w = float(min(w, WEIGHT_CAP))
        self.hits = 1
        self.status = "untracked"

    def __repr__(self):
        return "Gate(%d, %s, c=%s, hits=%d)" % (self.id, self.status, np.round(self.c, 3), self.hits)


class GateList:
    """Running list of windows. update() only from the modes that look (Survey, Measure)."""

    def __init__(self):
        self.gates = []

    def update(self, dets):
        """Match each detection to one gate (0.5 m, 20 deg), at most one per gate per frame.

        Untracked and tracked gates are updated: centroid weighted by inlier counts,
        normal by weighted slerp, weight capped. Measured, passed and false gates
        are frozen: a detection that matches one is absorbed, so it cannot start a
        duplicate. A detection that matches nothing starts an untracked gate.
        """
        used = set()
        for d in sorted(dets, key=lambda d: -d["n_inl"]):
            best, best_dist = None, TRACK_DIST_M
            for g in self.gates:
                if g.id in used:
                    continue
                dist = float(np.linalg.norm(d["c"] - g.c))
                if dist < best_dist and angle_deg(d["n"], g.n) < TRACK_ANGLE_DEG:
                    best, best_dist = g, dist
            if best is None:
                self.gates.append(Gate(d["c"], d["n"], d["n_inl"]))
                continue
            used.add(best.id)
            if best.status not in ("untracked", "tracked"):
                continue                                   # frozen: absorb
            w = float(d["n_inl"])
            best.c = (best.w * best.c + w * d["c"]) / (best.w + w)
            best.n = slerp(best.n, d["n"], w / (best.w + w))
            if best.n[0] < 0:
                best.n = -best.n
            best.w = min(best.w + w, WEIGHT_CAP)
            best.hits += 1
            if best.status == "untracked" and best.hits > TRACK_HITS:
                best.status = "tracked"

    def select(self, position):
        """Closest tracked gate ahead along +x, or None."""
        x = float(position[0])
        ahead = [g for g in self.gates if g.status == "tracked" and g.c[0] > x]
        return min(ahead, key=lambda g: g.c[0] - x, default=None)
