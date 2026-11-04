"""YOLO26 (nano) segmentation inference with plain PyTorch + OpenCV.
 
No `ultralytics` import needed at inference time. Export once (on any machine
that has ultralytics) with the same imgsz the model was trained/used with:
 
    yolo export model=weights.pt format=torchscript imgsz=320
 
Then:
 
    python yolo26_seg_infer.py weights.torchscript image.jpg --imgsz 320 --names yourclass
"""
import argparse
 
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
 
 
# --------------------------------------------------------------------------- #
# Pre-processing
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
 
 
# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def load_model(path, device=None):
    print("Path: " + path)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(path, map_location=device).eval()
    return model, device
 
 
# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict(model, device, image_bgr, imgsz=320, names=None, nc=1, nm=32,
            conf_thres=0.25, iou_thres=0.45, mask_thres=0.5, half=False):
    """Run segmentation inference on a cv2.imread() image.
 
    Returns a dict:
        "masks":  (M, H, W) bool ndarray, one mask per detection, original image size
        "labels": list[str] of length M
        "scores": (M,) float32 ndarray of confidences
        "boxes":  (M, 4) float32 ndarray, xyxy in original image pixels
    """
    names = names or [str(i) for i in range(nc)]
    H, W = image_bgr.shape[:2]
 
    # ---- preprocess ----
    img, r, dx, dy = letterbox(image_bgr, imgsz)
    x = torch.from_numpy(img[:, :, ::-1].copy()).permute(2, 0, 1)   # BGR->RGB, CHW
    x = x.unsqueeze(0).float().div(255).to(device)
    if half:
        x = x.half()
 
    # ---- forward ----
    out = model(x)
    if not isinstance(out, (list, tuple)):
        raise RuntimeError("Model returned a single tensor; a segmentation export "
                           "should also return mask prototypes.")
    pred, protos = out[0], out[1]
    pred, protos = pred.float()[0], protos.float()[0]               # protos: (nm, mh, mw)
 
    # ---- decode boxes / scores / mask coefficients ----
    if pred.shape[0] == 4 + nc + nm:
        # raw head output: (4+nc+nm, N) -> needs confidence filter + NMS
        pred = pred.transpose(0, 1)                                  # (N, 4+nc+nm)
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
        # end-to-end (NMS-free) output: (N, 6+nm) = [x1,y1,x2,y2,conf,cls,coeffs...]
        keep = pred[:, 4] > conf_thres
        pred = pred[keep]
        xyxy, scores, cls, coeffs = pred[:, :4], pred[:, 4], pred[:, 5].long(), pred[:, 6:]
    else:
        raise RuntimeError(f"Unexpected prediction shape {tuple(pred.shape)} "
                           f"for nc={nc}, nm={nm}")
 
    if len(scores) == 0:
        return {"masks": np.zeros((0, H, W), int), "labels": [],
                "scores": np.zeros(0, np.float32), "boxes": np.zeros((0, 4), np.float32)}
 
    # ---- build masks (in letterbox space) ----
    c, mh, mw = protos.shape
    m = torch.sigmoid(coeffs @ protos.reshape(c, -1)).reshape(-1, mh, mw)
    m = F.interpolate(m[None], size=(imgsz, imgsz), mode="bilinear", align_corners=False)[0]
 
    # zero everything outside each box
    ys = torch.arange(imgsz, device=m.device)[None, :, None]
    xs = torch.arange(imgsz, device=m.device)[None, None, :]
    b = xyxy[:, :, None, None]
    inside = (xs >= b[:, 0]) & (xs <= b[:, 2]) & (ys >= b[:, 1]) & (ys <= b[:, 3])
    m = ((m * inside) > mask_thres).cpu().numpy().astype(np.uint8)
 
    # ---- map back to original image ----
    nh, nw = round(H * r), round(W * r)
    masks = np.zeros((len(m), H, W), dtype=int)
    for i, mk in enumerate(m):
        mk = mk[dy:dy + nh, dx:dx + nw]                              # strip padding
        masks[i] = cv2.resize(mk, (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
 
    boxes = xyxy.clone().cpu().numpy().astype(np.float32)
    boxes[:, [0, 2]] = np.clip((boxes[:, [0, 2]] - dx) / r, 0, W)
    boxes[:, [1, 3]] = np.clip((boxes[:, [1, 3]] - dy) / r, 0, H)
 
    return {
        "masks": masks,
        "labels": [names[int(k)] for k in cls.cpu().tolist()],
        "scores": scores.cpu().numpy().astype(np.float32),
        "boxes": boxes,
    }
 
 
# --------------------------------------------------------------------------- #
# Visualisation
# --------------------------------------------------------------------------- #
def visualize(image_bgr, result, alpha=0.45):
    """Overlay masks, boxes and 'label score' text on a copy of the image."""
    rng = np.random.default_rng(0)
    vis = image_bgr.copy()
    overlay = image_bgr.copy()
    colors = [tuple(int(v) for v in rng.integers(50, 255, 3)) for _ in result["labels"]]
 
    for mask, color in zip(result["masks"], colors):
        overlay[mask] = color
    vis = cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0)
 
    for box, label, score, color in zip(result["boxes"], result["labels"],
                                        result["scores"], colors):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {score:.2f}"
        (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = max(y1, th + bl)
        cv2.rectangle(vis, (x1, ty - th - bl), (x1 + tw, ty), color, -1)
        cv2.putText(vis, text, (x1, ty - bl), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 1, cv2.LINE_AA)
    return vis

def find_center(image, mask):
    vis = image.copy()
    mask = mask.astype('int')
    nonzeros = np.nonzero(mask)
    x = nonzeros[0]
    y = nonzeros[1]
    avg_x = int(np.average(x))
    avg_y = int(np.average(y))
    cv2.circle(vis, (avg_y, avg_x), 10,  (0, 0, 255), -1)
    cv2.imshow('win', vis)
    cv2.waitKey()
    return vis, (avg_x, avg_y)

def closest(image_bgr, depth_m,  result):
    vis = image_bgr.copy()
    print(vis.shape)
    min_depth = float('inf')
    closest_mask = None
    count = 0
    for mask in result["masks"]:
        mask = mask.astype('uint8')
        #Erode the mask.
        radius = 6
        thickness = 1
        new_mask = erode(mask.copy(), radius, thickness)
        new_mask = new_mask.astype('uint8')

        masked_depth = cv2.bitwise_and(depth_m, depth_m, mask=new_mask)

        if(not np.count_nonzero(masked_depth) > 0):
            print("Eroded mask not found. Continue.")
            continue
        depth_avg = np.sum(masked_depth) / np.count_nonzero(masked_depth)
        print("Depth_avg: " + str(depth_avg))
        count += 1
        if(depth_avg < min_depth):
            print("Found new closest mask.")
            print(count)
            closest_mask = new_mask
            min_depth = depth_avg
    if(closest_mask is not None):
        return {"mask": closest_mask.astype('bool')}
    else:
        return None
 
def erode(mask, radius, thickness):
    out_radius = radius
    in_radius = radius + thickness
    out_k = 2*out_radius + 1
    out_kernel = np.ones((out_k, out_k), np.float32)
    out_kernel /= out_kernel.sum()
    out = cv2.filter2D(mask.astype(np.float32), -1, out_kernel, borderType = cv2.BORDER_CONSTANT)
    out = out > .999
    in_k = 2*in_radius+1
    in_kernel = np.ones((in_k, in_k), np.float32)
    in_kernel /= in_kernel.sum()
    in_ = cv2.filter2D(mask.astype(np.float32), -1, in_kernel, borderType= cv2.BORDER_CONSTANT)
    in_ = in_ > .999
    return out & ~in_



# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="path to weights.torchscript")
    ap.add_argument("image", help="path to input image")
    ap.add_argument("--imgsz", type=int, default=320,
                    help="size the model was exported with (multiple of 32)")
    ap.add_argument("--names", nargs="+", default=["object"], help="class names")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out", default="result.jpg")
    args = ap.parse_args()
 
    model, device = load_model(args.model)
    image_path = args.image + '.png'
    depth_path = args.image + '_depth.npy'
    depth = np.load(depth_path)
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(args.image)
 
    result = predict(model, device, image, imgsz=args.imgsz, names=args.names,
                     nc=len(args.names), conf_thres=args.conf)
    print({k: (v.shape if hasattr(v, "shape") else v) for k, v in result.items()})
    
    cv2.imwrite(args.out, visualize(image, result))

    mask = closest(image, depth, result)['mask'].astype('uint8')*255
    center_img, center = find_center(image, mask)
    print("Center: " + str(center))
    cv2.imwrite('closest.jpg', mask)
    cv2.imwrite('center.jpg', center_img)

    
    print("saved", args.out)
