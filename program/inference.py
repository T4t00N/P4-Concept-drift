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


def load_checkpoint(
    ckpt_path: str | os.PathLike, num_classes: int, device: torch.device
) -> tuple[MLP, List[torch.nn.Module]]:
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

    mlp.eval()
    [y.eval() for y in yolos]
    return mlp, yolos


# ---------------- preprocessing ----------------------------------------------

def preprocess_image(
    path: str | os.PathLike, input_sz: int
) -> tuple[torch.Tensor, Tuple[int, int], Tuple[float, float], Tuple[float, float]]:
    img0 = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img0 is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    img, ratio, pad = resize_static(img0, input_sz)
    tensor = (
        torch.from_numpy(img.astype(np.float32) / 255.0)
        .unsqueeze(0)
        .unsqueeze(0)
    )  # 1×1×H×W
    return tensor, img0.shape[:2], ratio, pad


# ---------------- weight helper ---------------------------------------------

def get_soft_weights(
    moco: torch.nn.Module, mlp: MLP, img: torch.Tensor, device, temp: float = 0.1
) -> torch.Tensor:
    with torch.inference_mode():
        x3 = img.repeat(1, 3, 1, 1) if img.shape[1] == 1 else img
        _, feats = moco(x3.to(device))
        w = torch.softmax(mlp(feats) / temp, dim=1).squeeze(0)  # (3,)
    return w


# ---------------- ensemble forward ------------------------------------------

def run_single_image(
    img_path: str | os.PathLike,
    moco: torch.nn.Module,
    mlp: MLP,
    yolos: List[torch.nn.Module],
    names: Dict,
    device,
    input_sz: int = 384,
    conf_thres: float = 0.25,
    iou_thres: float = 0.65,
) -> List[Tuple[int, float, float, float, float, float]]:
    img_t, (h0, w0), ratio, pad = preprocess_image(img_path, input_sz)
    img = img_t.to(device).float()

    # MoCo ▸ MLP ----------------------------------------------------------
    weights = get_soft_weights(moco, mlp, img, device)

    # YOLO predictions ----------------------------------------------------
    with torch.inference_mode():
        preds = [y(img).float() for y in yolos]
    preds_nms = [util.non_max_suppression(p, conf_thres, iou_thres)[0] for p in preds]

    # to WBF format -------------------------------------------------------
    boxes_l, scores_l, labels_l = [], [], []
    for det in preds_nms:
        if det is None or not det.numel():
            boxes_l.append(np.zeros((0, 4), dtype=np.float32))
            scores_l.append(np.zeros((0,), dtype=np.float32))
            labels_l.append(np.zeros((0,), dtype=np.float32))
            continue

        det = det.clone()
        det[:, [0, 2]] = det[:, [0, 2]].clamp(0, input_sz)
        det[:, [1, 3]] = det[:, [1, 3]].clamp(0, input_sz)

        b = (det[:, :4] / input_sz).clamp(0.0, 1.0)

        boxes_l.append(b.cpu().numpy())
        scores_l.append(det[:, 4].cpu().numpy())
        labels_l.append(det[:, 5].cpu().numpy())

    if sum(len(b) for b in boxes_l) == 0:
        return []

    boxes_f, scores_f, labels_f = weighted_boxes_fusion(
        boxes_l,
        scores_l,
        labels_l,
        weights=weights.cpu().tolist(),
        iou_thr=0.55,
        skip_box_thr=conf_thres,
        conf_type="max",
    )

    boxes_f = boxes_f * input_sz  # back to pixels in letterboxed scale
    boxes_f[:, 0] -= pad[0]
    boxes_f[:, 2] -= pad[0]
    boxes_f[:, 1] -= pad[1]
    boxes_f[:, 3] -= pad[1]
    boxes_f /= ratio[0]  # undo resize

    out = []
    for (x1, y1, x2, y2), sc, lb in zip(boxes_f, scores_f, labels_f):
        if sc < conf_thres:
            continue
        if not np.isfinite([x1, y1, x2, y2]).all():
            continue
        out.append((int(lb), float(sc), float(x1), float(y1), float(x2), float(y2)))
    return out


# ---------------- drawing ----------------------------------------------------

def draw_boxes_and_save(
    img_path: str | os.PathLike,
    detections: List[Tuple[int, float, float, float, float, float]],
    names: Dict[int, str],
    save_path: str | os.PathLike | None,
):
    if save_path is None:
        return

    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")

    for cls, conf, x1, y1, x2, y2 in detections:
        if not np.isfinite([x1, y1, x2, y2]).all():
            continue  # safety
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        color = (0, 255, 0)
        cv2.rectangle(img, p1, p2, color, 2)
        label = f"{names.get(cls, cls)} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (p1[0], p1[1] - th - 4), (p1[0] + tw, p1[1]), color, -1)
        cv2.putText(
            img,
            label,
            (p1[0], p1[1] - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    if save_path == "" or save_path is None:
        p = Path(img_path)
        save_path = p.with_name(f"{p.stem}_pred{p.suffix}")
    else:
        save_path = Path(save_path)

    cv2.imwrite(str(save_path), img)
    print(f"Annotated image saved ➜ {save_path}")


# -----------------------------------------------------------------------------
# CLI -------------------------------------------------------------------------
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Run inference on one image with MoCo➜MLP➜YOLO ensemble"
    )
    p.add_argument("--image", default=r"C:\Users\anto3\Downloads\cluster_39_20210302930130818.jpg",
                   help="path to the input image")
    p.add_argument("--weights", default=r"C:\Users\anto3\Documents\GitHub\P4-Concept-drift\program\weights\checkpoints\epoch_8.pt", help="ensemble checkpoint")
    p.add_argument("--moco", default=r"C:\Users\anto3\Documents\GitHub\P4-Concept-drift\program\weights\checkpoints\moco_epoch_100.pt", help="MoCo checkpoint")
    p.add_argument("--hyp", default="utils/args.yaml", help="yaml with class names")
    p.add_argument("--input-size", type=int, default=384)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.65)
    p.add_argument(
        "--save",
        nargs="?",
        const="",
        default="output.jpg",
        help="save annotated image; pass flag alone to auto‑name",
    )
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hyp = load_hyp(args.hyp)

    # models -------------------------------------------------------------
    moco = fv.load_moco_model(args.moco, device)
    moco.eval()

    mlp, yolos = load_checkpoint(
        args.weights, num_classes=len(hyp["names"]), device=device
    )

    # inference ----------------------------------------------------------
    dets = run_single_image(
        args.image,
        moco,
        mlp,
        yolos,
        hyp["names"],
        device,
        input_sz=args.input_size,
        conf_thres=args.conf,
        iou_thres=args.iou,
    )

    if not dets:
        print("No objects detected.")
    else:
        for cls, conf, x1, y1, x2, y2 in dets:
            name = hyp["names"].get(cls, str(cls))
            print(
                f"{name:<12} {conf:.3f}  ({x1:.1f}, {y1:.1f}) – ({x2:.1f}, {y2:.1f})"
            )

    # save visualisation --------------------------------------------------
    draw_boxes_and_save(args.image, dets, hyp["names"], args.save)


if __name__ == "__main__":
    main()
