"""Gaussian-splat renderer on the CPU, numpy only: for pictures, not speed.

The same model as gsplat's classic mode (EWA projection, 0.3 px^2 dilation,
front-to-back alpha compositing); centre depths agree with gsplat to 2 mm.
Works in Vicon metres. A 1920 x 1080 frame takes about 15 s.

    import numpy as np
    from splat_hitl.bundle import SceneBundle
    from splat_cpu import Scene, Camera, render

    b = SceneBundle.load("scene")
    scene = Scene(b.path("splat"), b.transform())
    eye = np.array([-0.1, 0.25, 1.0])
    cam = Camera(640, 360, eye, Camera.look_at(eye, eye + [1, 0, 0]), focal_px=554)
    rgb, alpha, depth = render(scene, cam)
"""
import numpy as np

TILE = 16


def load_splat(path):
    rec = np.fromfile(path, dtype=np.uint8).reshape(-1, 32)
    f = rec[:, :24].copy().view("<f4").reshape(-1, 6)
    pos = f[:, :3].astype(np.float64)
    scl = np.maximum(f[:, 3:6].astype(np.float64), 1e-7)
    rgba = rec[:, 24:28].astype(np.float64) / 255.0
    q = (rec[:, 28:32].astype(np.float64) - 128.0) / 128.0
    q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    return pos, scl, rgba, q


def quat_to_R(q):
    w, x, y, z = q.T
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
                     np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
                     np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], 1)


class Scene:
    """Gaussians in the Vicon frame: means (N,3), covariances (N,3,3), rgb (N,3), opacity (N,)."""

    def __init__(self, splat_path, tf):
        pos, scl, rgba, q = load_splat(splat_path)
        R = quat_to_R(q)
        M = R * scl[:, None, :]
        cov = M @ np.transpose(M, (0, 2, 1))                    # splat units^2
        Rt = np.asarray(tf.R, float)
        self.mean = tf.point_to_vicon(pos)
        self.cov = np.einsum("ji,njk,kl->nil", Rt, cov, Rt) / tf.scale ** 2   # R^T C R / s^2
        self.rgb = rgba[:, :3]
        self.opacity = rgba[:, 3]

    def select(self, box):
        (x0, x1), (y0, y1), (z0, z1) = box
        m = self.mean
        return (m[:, 0] >= x0) & (m[:, 0] <= x1) & (m[:, 1] >= y0) & (m[:, 1] <= y1) & (m[:, 2] >= z0) & (m[:, 2] <= z1)


class Camera:
    """World (Vicon) -> image. Rows of R are the camera's right, down and forward axes."""

    def __init__(self, width, height, position, R, ortho_px_per_m=None, focal_px=None):
        self.W, self.H = int(width), int(height)
        self.C = np.asarray(position, float)
        self.R = np.asarray(R, float)
        self.ortho = ortho_px_per_m
        self.f = focal_px
        self.cx, self.cy = self.W / 2.0, self.H / 2.0

    @staticmethod
    def look_at(eye, target, up=(0, 0, 1)):
        eye, target, up = (np.asarray(v, float) for v in (eye, target, up))
        fwd = target - eye; fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, up); right /= np.linalg.norm(right)
        down = np.cross(fwd, right)
        return np.stack([right, down, fwd])

    def to_cam(self, P):
        return (np.asarray(P, float).reshape(-1, 3) - self.C) @ self.R.T

    def project(self, P):
        """(u, v, depth) for world points. u, v in pixels, pixel i spans [i, i+1)."""
        X = self.to_cam(P)
        if self.ortho:
            return self.cx + self.ortho * X[:, 0], self.cy + self.ortho * X[:, 1], X[:, 2]
        z = X[:, 2]
        zs = np.where(np.abs(z) < 1e-9, 1e-9, z)
        return self.cx + self.f * X[:, 0] / zs, self.cy + self.f * X[:, 1] / zs, z


def render(scene, cam, keep=None, background=(0.0, 0.0, 0.0), near=0.05, chunk=3072, verbose=False):
    idx = np.flatnonzero(keep) if keep is not None else np.arange(len(scene.opacity))
    mean, cov, rgb, op = scene.mean[idx], scene.cov[idx], scene.rgb[idx], scene.opacity[idx]
    Xc = cam.to_cam(mean)
    z = Xc[:, 2]
    ok = (z > near) & (op >= 1.0 / 255.0)
    Xc, cov, rgb, op, z = Xc[ok], cov[ok], rgb[ok], op[ok], z[ok]
    Cc = np.einsum("ij,njk,lk->nil", cam.R, cov, cam.R)               # camera-frame covariance
    if cam.ortho:
        s = cam.ortho
        u, v = cam.cx + s * Xc[:, 0], cam.cy + s * Xc[:, 1]
        S2 = Cc[:, :2, :2] * s * s
    else:
        f = cam.f
        # gsplat clamps the Jacobian's x/z, y/z to 1.3x the field of view
        lx, ly = 1.3 * (cam.W / 2) / f, 1.3 * (cam.H / 2) / f
        tx = np.clip(Xc[:, 0] / z, -lx, lx) * z
        ty = np.clip(Xc[:, 1] / z, -ly, ly) * z
        J = np.zeros((len(z), 2, 3))
        J[:, 0, 0] = f / z; J[:, 0, 2] = -f * tx / z ** 2
        J[:, 1, 1] = f / z; J[:, 1, 2] = -f * ty / z ** 2
        S2 = J @ Cc @ np.transpose(J, (0, 2, 1))
        u, v = cam.cx + f * Xc[:, 0] / z, cam.cy + f * Xc[:, 1] / z
    a = S2[:, 0, 0] + 0.3; b = S2[:, 0, 1]; c = S2[:, 1, 1] + 0.3
    det = a * c - b * b
    good = det > 1e-12
    lam = 0.5 * (a + c) + np.sqrt(np.maximum(0.25 * (a - c) ** 2 + b * b, 0.0))
    # radius where alpha falls to 1/255, never past 3 sigma
    kr = np.sqrt(np.maximum(2.0 * np.log(np.maximum(255.0 * op, 1.0)), 0.0))
    r = np.ceil(np.minimum(3.0, kr) * np.sqrt(lam))
    ntx, nty = -(-cam.W // TILE), -(-cam.H // TILE)
    tx0 = np.floor((u - r) / TILE).astype(np.int64); tx1 = np.floor((u + r) / TILE).astype(np.int64)
    ty0 = np.floor((v - r) / TILE).astype(np.int64); ty1 = np.floor((v + r) / TILE).astype(np.int64)
    good &= (tx1 >= 0) & (tx0 < ntx) & (ty1 >= 0) & (ty0 < nty) & (r > 0)
    tx0, tx1 = np.clip(tx0, 0, ntx - 1), np.clip(tx1, 0, ntx - 1)
    ty0, ty1 = np.clip(ty0, 0, nty - 1), np.clip(ty1, 0, nty - 1)
    g = np.flatnonzero(good)
    nx, ny = tx1[g] - tx0[g] + 1, ty1[g] - ty0[g] + 1
    cnt = nx * ny
    gid = np.repeat(g, cnt)
    off = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    rnx = np.repeat(nx, cnt)
    tile = (np.repeat(ty0[g], cnt) + off // rnx) * ntx + np.repeat(tx0[g], cnt) + off % rnx
    order = np.lexsort((z[gid], tile))
    tile, gid = tile[order], gid[order]
    bounds = np.searchsorted(tile, np.arange(ntx * nty + 1))
    if verbose:
        print("  %d gaussians, %d tile pairs over %dx%d tiles" % (len(g), len(gid), ntx, nty))
    conA, conB, conC = c / det, -b / det, a / det
    out_rgb = np.zeros((nty * TILE, ntx * TILE, 3))
    out_acc = np.zeros((nty * TILE, ntx * TILE))
    out_dep = np.zeros((nty * TILE, ntx * TILE))
    lx = np.arange(TILE) + 0.5
    PX, PY = np.meshgrid(lx, lx)
    PX, PY = PX.ravel(), PY.ravel()
    for t in range(ntx * nty):
        lo, hi = bounds[t], bounds[t + 1]
        if lo == hi:
            continue
        ty_, tx_ = divmod(t, ntx)
        px, py = PX + tx_ * TILE, PY + ty_ * TILE
        T = np.ones(TILE * TILE); C = np.zeros((TILE * TILE, 3)); D = np.zeros(TILE * TILE)
        for k in range(lo, hi, chunk):
            ids = gid[k:min(hi, k + chunk)]
            dx = px[:, None] - u[ids][None, :]; dy = py[:, None] - v[ids][None, :]
            q = conA[ids] * dx * dx + 2.0 * conB[ids] * dx * dy + conC[ids] * dy * dy
            al = np.minimum(0.99, op[ids] * np.exp(-0.5 * q))
            al[al < 1.0 / 255.0] = 0.0
            Tc = np.cumprod(1.0 - al, axis=1)
            w = al * np.concatenate([np.ones((len(px), 1)), Tc[:, :-1]], axis=1) * T[:, None]
            C += w @ rgb[ids]; D += w @ z[ids]
            T = T * Tc[:, -1]
            if T.max() < 1e-4:
                break
        sl = (slice(ty_ * TILE, ty_ * TILE + TILE), slice(tx_ * TILE, tx_ * TILE + TILE))
        acc = 1.0 - T
        out_rgb[sl] = (C + T[:, None] * np.asarray(background)).reshape(TILE, TILE, 3)
        out_acc[sl] = acc.reshape(TILE, TILE)
        out_dep[sl] = np.where(acc > 1e-6, D / np.maximum(acc, 1e-12), np.inf).reshape(TILE, TILE)
    return (np.clip(out_rgb[:cam.H, :cam.W], 0, 1), out_acc[:cam.H, :cam.W], out_dep[:cam.H, :cam.W])
