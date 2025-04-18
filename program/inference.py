#!/usr/bin/env python
"""inference.py — single‑image object detection for the MoCo➜MLP➜YOLO ensemble.

Run:
    python inference.py --image /path/to/img.jpg \
                       --weights weights/final.pt \
                       --moco /ceph/project/P4-concept-drift/YOLOv8-Anton/data/moco_epoch_100.pt
Outputs fused detections in pixel coordinates of the original image.
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path
from typing import List, Tuple, Dict

import cv2
import numpy as np
import torch
import yaml

# -----------------------------------------------------------------------------
# local project imports --------------------------------------------------------
# -----------------------------------------------------------------------------
import models.MoCo_inference as fv
from nets.mlp_net import MLP
from nets import nn
from utils import util
from utils.WBF import weighted_boxes_fusion
from utils.dataset import resize_static

# -----------------------------------------------------------------------------
# helpers ---------------------------------------------------------------------
# -----------------------------------------------------------------------------

def load_hyp(path: str | os.PathLike) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def strip_module(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remove a leading 'module.' (DDP) prefix from keys (if present)."""
    return {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in state_dict.items()}


def load_checkpoint(ckpt_path: str | os.PathLike,
                    num_classes: int,
                    device: torch.device) -> tuple[MLP, List[torch.nn.Module]]:
    """Instantiate MLP + three YOLOs and load *final.pt* weights robustly."""
    ckpt = torch.load(ckpt_path, map_location=device)

    # models ---------------------------------------------------------------
    mlp = MLP().to(device)
    yolos = [nn.yolo_v8_n(num_classes).to(device) for _ in range(3)]

    # load MLP weights -----------------------------------------------------
    mlp_sd = ckpt.get("mlp")
    assert mlp_sd, "mlp weights not found in checkpoint"
    mlp.load_state_dict(strip_module(mlp_sd), strict=True)

    # load YOLO weights ----------------------------------------------------
    for idx, tag in enumerate(["yolo1", "yolo2", "yolo3"]):
        y_sd = ckpt.get(tag)
        assert y_sd, f"{tag} weights missing in checkpoint"
        yolos[idx].load_state_dict(strip_module(y_sd), strict=True)

    mlp.eval(); [y.eval() for y in yolos]
    return mlp, yolos


def preprocess_image(path: str | os.PathLike, input_sz: int) -> tuple[torch.Tensor, Tuple[int,int], Tuple[float,float], Tuple[float,float]]:
    img0 = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img0 is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    img, ratio, pad = resize_static(img0, input_sz)
    tensor = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)  # 1×1×H×W
    return tensor, img0.shape[:2], ratio, pad


def get_soft_weights(moco: torch.nn.Module, mlp: MLP, img: torch.Tensor, device, temp: float = 0.1) -> torch.Tensor:
    with torch.inference_mode():
        x3 = img.repeat(1, 3, 1, 1) if img.shape[1] == 1 else img
        _, feats = moco(x3.to(device))
        w = torch.softmax(mlp(feats) / temp, dim=1).squeeze(0)  # (3,)
    return w


def run_single_image(img_path: str | os.PathLike,
                     moco: torch.nn.Module,
                     mlp: MLP,
                     yolos: List[torch.nn.Module],
                     names: Dict,
                     device,
                     input_sz: int = 384,
                     conf_thres: float = 0.25,
                     iou_thres: float = 0.65) -> List[Tuple[int,float,float,float,float,float]]:
    img_t, (h0, w0), ratio, pad = preprocess_image(img_path, input_sz)
    img = img_t.to(device).float()

    # MoCo ▸ MLP ----------------------------------------------------------
    weights = get_soft_weights(moco, mlp, img, device)

    # YOLO predictions ----------------------------------------------------
    # YOLO predictions under no‑grad to avoid autograd bookkeeping
    with torch.inference_mode():
        preds = [y(img).float() for y in yolos]
    preds_nms = [util.non_max_suppression(p, conf_thres, iou_thres)[0] for p in preds]

    # to WBF format -------------------------------------------------------
    boxes_l, scores_l, labels_l = [], [], []
    for model_idx, det in enumerate(preds_nms):
        if det is None or not det.numel():
            boxes_l.append(np.zeros((0, 4), dtype=np.float32))
            scores_l.append(np.zeros((0,), dtype=np.float32))
            labels_l.append(np.zeros((0,), dtype=np.float32))
            continue

        # clamp to image bounds before normalisation
        det = det.clone()
        det[:, [0, 2]] = det[:, [0, 2]].clamp(0, input_sz)
        det[:, [1, 3]] = det[:, [1, 3]].clamp(0, input_sz)

        # convert to [0,1] range for WBF
        b = det[:, :4] / input_sz  # (n,4) float32
        # clip again to be sure
        b = b.clamp(0.0, 1.0)

        boxes_l.append(b.detach().cpu().numpy())
        scores_l.append(det[:, 4].detach().cpu().numpy())
        labels_l.append(det[:, 5].detach().cpu().numpy())

    if sum(len(b) for b in boxes_l) == 0:
        return []

    boxes_f, scores_f, labels_f = weighted_boxes_fusion(
        boxes_l,
        scores_l,
        labels_l,
        weights=weights.cpu().tolist(),
        iou_thr=0.55,
        skip_box_thr=conf_thres,
        conf_type='max'  # use max confidence to avoid tiny averaged scores
    )

    # de‑norm & un‑letterbox --------------------------------------------
    boxes_f = boxes_f * input_sz  # broadcast
    boxes_f[:, 0] -= pad[0]; boxes_f[:, 2] -= pad[0]
    boxes_f[:, 1] -= pad[1]; boxes_f[:, 3] -= pad[1]
    boxes_f /= ratio[0]

    out = []
    for (x1, y1, x2, y2), sc, lb in zip(boxes_f, scores_f, labels_f):
        out.append((int(lb), float(sc), float(x1), float(y1), float(x2), float(y2)))
    return out

# -----------------------------------------------------------------------------
# CLI -------------------------------------------------------------------------
# -----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Run inference on one image with MoCo➜MLP➜YOLO ensemble")
    p.add_argument("--image", default="/ceph/project/P4-concept-drift/final_yolo_data_format/YOLOv8-pt/Dataset/images/test/20210405330234748.jpg",
                   help="path to the input image")
    p.add_argument("--weights", default="weights/checkpoints/epoch_2.pt", help="ensemble checkpoint")
    p.add_argument("--moco", default="/ceph/project/P4-concept-drift/YOLOv8-Anton/data/moco_epoch_100.pt", help="MoCo checkpoint")
    p.add_argument("--hyp", default="utils/args.yaml", help="yaml with class names")
    p.add_argument("--input-size", type=int, default=384)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.65)
    args = p.parse_args()


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hyp = load_hyp(args.hyp)

    # models -------------------------------------------------------------
    moco = fv.load_moco_model(args.moco, device)
    moco.eval()

    mlp, yolos = load_checkpoint(args.weights, num_classes=len(hyp["names"]), device=device)

    # inference ----------------------------------------------------------
    dets = run_single_image(
        args.image, moco, mlp, yolos, hyp["names"], device,
        input_sz=args.input_size, conf_thres=args.conf, iou_thres=args.iou)

    if not dets:
        print("No objects detected.")
        return

    for cls, conf, x1, y1, x2, y2 in dets:
        name = hyp["names"].get(cls, str(cls))
        print(f"{name:<12} {conf:.3f}  ({x1:.1f}, {y1:.1f}) – ({x2:.1f}, {y2:.1f})")

if __name__ == "__main__":
    main()
