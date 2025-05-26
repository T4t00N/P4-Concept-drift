"""
Baseline (no-MoCo, uniform-weight) inference & evaluation
--------------------------------------------------------

• Single image:
    python utils/baseline_inference.py \
           --image  demo.jpg \
           --weights ckpt0.pt ckpt1.pt ckpt2.pt

• Validation set (list or folder):
    python utils/baseline_inference.py \
           --val    val_images.txt          # or  --val path/to/images/
           --weights ckpt0.pt ckpt1.pt ckpt2.pt

The three YOLO checkpoints may be anywhere on disk; pass them in the
order you prefer.  All three receive a fixed voting weight = 1/3.
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import List, Tuple, Dict

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm

# --------------------------------------------------------------------------
# project helpers ----------------------------------------------------------
# --------------------------------------------------------------------------
from utils import util                                  # NMS, IoU, AP, …
from utils.dataset import resize_static                 # static-letterbox
try:                                                    # Weighted Boxes Fusion
    from utils.WBF import weighted_boxes_fusion
except ModuleNotFoundError:
    from ensemble_boxes import weighted_boxes_fusion    # pip install ensemble-boxes

from nets import nn                                     # YOLO-v8-nano factory
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# CLI ----------------------------------------------------------------------
# --------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Fixed-weight YOLO ensemble: inference and evaluation."
    )

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="single image for inference")
    src.add_argument(
        "--val",
        help="validation source: • txt file with one image per line "
             "or • directory containing images (*.jpg *.png …)",
    )

    p.add_argument(
        "--weights",
        nargs=3,
        metavar=("CKPT0", "CKPT1", "CKPT2"),
        required=True,
        help="three individual YOLO checkpoints",
    )
    p.add_argument("--hyp", default="utils/args.yaml", help="yaml with class names")
    p.add_argument("--input-size", type=int, default=384)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.65)
    p.add_argument("--batch-size", type=int, default=8, help="for --val mode")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--save", action="store_true", help="save annotated images")
    p.add_argument("--show", action="store_true", help="display annotated images (single image mode only)") # Added --show argument
    return p.parse_args()


# --------------------------------------------------------------------------
# Utilities ----------------------------------------------------------------
# --------------------------------------------------------------------------
@torch.inference_mode()
def load_three_yolos(ckpt_paths: list[str], num_classes: int, device) \
        -> list[torch.nn.Module]:
    yolos = [nn.yolo_v8_n(num_classes).to(device) for _ in range(3)]
    for net, ck in zip(yolos, ckpt_paths):
        sd = torch.load(ck, map_location=device)
        if isinstance(sd, dict) and "model" in sd:
            sd = sd["model"]
        if isinstance(sd, torch.nn.Module):
            sd = sd.state_dict()
        net.load_state_dict(sd, strict=False)
        net.eval()
    return yolos


def get_uniform_weights(n: int = 3, device=None) -> torch.Tensor:
    w = torch.full((n,), 1.0 / n)
    return w.to(device) if device else w


def load_hyp(path: str | os.PathLike) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ---------- preprocessing -------------------------------------------------
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


# ---------- single-image inference ----------------------------------------
@torch.inference_mode()
def run_single_image(
    img_path: str | os.PathLike,
    yolos: List[torch.nn.Module],
    weights: torch.Tensor,
    device,
    names: Dict,
    input_sz: int = 384,
    conf_thres: float = 0.25,
    iou_thres: float = 0.65,
) -> List[Tuple[int, float, float, float, float, float]]:
    tensor, (h0, w0), ratio, pad = preprocess_image(img_path, input_sz)
    img = tensor.to(device).float()

    # YOLO raw predictions --------------------------------------------------
    with torch.inference_mode():
        preds = [y(img).float() for y in yolos]
    preds_nms = [util.non_max_suppression(p, conf_thres, iou_thres)[0] for p in preds]

    # convert each det tensor to WBF format --------------------------------
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

        boxes_l.append((det[:, :4] / input_sz).cpu().numpy())
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

    boxes_f = boxes_f * input_sz               # back to pixels (letterboxed)
    boxes_f[:, 0] -= pad[0];  boxes_f[:, 2] -= pad[0]
    boxes_f[:, 1] -= pad[1];  boxes_f[:, 3] -= pad[1]
    boxes_f /= ratio[0]                          # undo resize

    out = []
    for (x1, y1, x2, y2), sc, lb in zip(boxes_f, scores_f, labels_f):
        if sc < conf_thres:
            continue
        if not np.isfinite([x1, y1, x2, y2]).all():
            continue
        out.append((int(lb), float(sc), float(x1), float(y1), float(x2), float(y2)))
    return out


# ---------- evaluation loop -----------------------------------------------
@torch.inference_mode()
def evaluate_dataset(
    file_list: List[str],
    yolos: List[torch.nn.Module],
    weights: torch.Tensor,
    device,
    input_sz: int = 384,
    conf_thres: float = 0.25,
    iou_thres: float = 0.65,
):
    tps, confs, pred_cls, target_cls = [], [], [], []
    num_images_with_dets = 0
    num_images_with_gts = 0

    print(f"\nStarting evaluation on {len(file_list)} images...")
    print(f"Confidence Threshold: {conf_thres}, IOU Threshold for NMS/WBF: {iou_thres}")


    for img_path in tqdm(file_list, desc="evaluating"):
        preds = run_single_image(
            img_path, yolos, weights, device, names={},  # names not needed here
            input_sz=input_sz, conf_thres=conf_thres, iou_thres=iou_thres
        )

        if preds:
            num_images_with_dets += 1

        # ground-truth labels ---------------------------------------------
        # MODIFIED: Use the same label path logic as in main.py's Dataset class
        image_folder_str = f"{os.sep}images{os.sep}"
        label_folder_str = f"{os.sep}labels{os.sep}"

        lbl_path = img_path # Start with the image path
        if image_folder_str in lbl_path: # Check if the image folder pattern exists
            # Split and rejoin to swap 'images' for 'labels'
            split_parts = lbl_path.rsplit(image_folder_str, 1)
            lbl_path = label_folder_str.join(split_parts)

        # Ensure it has a .txt extension
        lbl_path = os.path.splitext(lbl_path)[0] + '.txt'


        gts = []
        if os.path.isfile(lbl_path):
            with open(lbl_path) as f:
                lines = [line.strip() for line in f if line.strip()]
                if lines: # Check if there are any lines after stripping
                    num_images_with_gts += 1
                    # Only read img0 if there are GTs to process (efficiency)
                    img0 = cv2.imread(img_path)
                    if img0 is None:
                        print(f"Warning: Could not read image {img_path} for GT processing. Skipping GT for this image.")
                        continue
                    h0, w0 = img0.shape[:2]
                    for line in lines:
                        try:
                            c, xc, yc, w, h = map(float, line.split()[:5])
                            x1 = (xc - w / 2) * w0
                            y1 = (yc - h / 2) * h0
                            x2 = (xc + w / 2) * w0
                            y2 = (yc + h / 2) * h0
                            gts.append((int(c), x1, y1, x2, y2))
                        except ValueError as e:
                            print(f"Warning: Error parsing GT line '{line}' in {lbl_path}: {e}. Skipping this line.")
                else:
                    pass # GT file exists but is empty


        gt_boxes = torch.tensor([g[1:] for g in gts], device=device) if gts else torch.zeros((0, 4), device=device)
        gt_cls   = [g[0] for g in gts]

        if preds:
            pred_boxes = torch.tensor([p[2:] for p in preds], device=device)
            pred_confs = torch.tensor([p[1] for p in preds], device=device)
            pred_clses = [p[0] for p in preds]

            # Ensure that ious calculation handles cases where pred_boxes or gt_boxes are empty
            if pred_boxes.numel() == 0 or gt_boxes.numel() == 0:
                ious = torch.zeros((len(preds), len(gts)), device=device)
            else:
                ious = util.box_iou(pred_boxes, gt_boxes)


            assigned = set()
            for idx, (box, conf, cls_) in enumerate(zip(pred_boxes, pred_confs, pred_clses)):
                tpf = 0
                if gt_boxes.numel(): # Only try to find true positives if there are ground truths
                    c_mask = [i for i, c in enumerate(gt_cls) if c == cls_ and i not in assigned]
                    if c_mask:
                        # Ensure iou_vals is not empty before argmax/max
                        if ious.shape[1] > 0: # Check if there are columns in ious for this GT
                            iou_vals = ious[idx, c_mask]
                            if iou_vals.numel() > 0: # Check if there are actual iou values for current class
                                best_iou, best_gt = (iou_vals.max().item(), c_mask[int(iou_vals.argmax())])
                                if best_iou > iou_thres: # This iou_thres is for TP assignment
                                    tpf = 1
                                    assigned.add(best_gt)
                tps.append(tpf)
                confs.append(conf.item())
                pred_cls.append(cls_)

        # record all GT classes for AP calculation
        target_cls.extend(gt_cls)

    print(f"\n--- Evaluation Summary ---")
    print(f"Total images processed: {len(file_list)}")
    print(f"Images with at least one detection: {num_images_with_dets}")
    print(f"Images with at least one ground truth: {num_images_with_gts}")
    print(f"Total detections collected (before filtering for AP): {len(pred_cls)}")
    print(f"Total ground truths collected: {len(target_cls)}")
    print(f"Total True Positives collected: {sum(tps)}")

    if len(pred_cls) == 0:
        print("⚠️  No detections found across the entire validation set. Cannot compute metrics.")
        return

    if len(target_cls) == 0:
        print("⚠️  No ground truth labels found across the entire validation set. Cannot compute Recall or mAP.")
        import numpy as np
        tp_array   = np.array(tps).reshape(-1, 1)
        conf_array = np.array(confs)
        pred_array = np.array(pred_cls)
        print(f"\nPrecision: {np.mean(tp_array) if tp_array.size > 0 else 0:.4f}")
        return


    import numpy as np
    tp_array   = np.array(tps).reshape(-1, 1)
    conf_array = np.array(confs)
    pred_array = np.array(pred_cls)
    tgt_array  = np.array(target_cls)

    if tp_array.size == 0 or conf_array.size == 0 or pred_array.size == 0:
        print("⚠️  Insufficient prediction data to compute AP. Check confidence thresholds or model performance.")
        print(f"tp_array size: {tp_array.size}, conf_array size: {conf_array.size}, pred_array size: {pred_array.size}")
        return

    _, _, m_pre, m_rec, map50, map_all = util.compute_ap(
        tp_array, conf_array, pred_array, tgt_array
    )

    print(f"\nPrecision: {m_pre:.4f}"
          f" | Recall: {m_rec:.4f}"
          f" | mAP@0.5: {map50:.4f}"
          f" | mAP@0.5:0.95: {map_all:.4f}")


# --------------------------------------------------------------------------
# main ---------------------------------------------------------------------
# --------------------------------------------------------------------------
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hyp = load_hyp(args.hyp)

    yolos   = load_three_yolos(args.weights, num_classes=len(hyp["names"]), device=device)
    weights = get_uniform_weights(device=device)

    if args.image:
        dets = run_single_image(
            args.image, yolos, weights, device, hyp["names"],
            input_sz=args.input_size, conf_thres=args.conf, iou_thres=args.iou
        )
        print(f"\nDetections on {args.image}:")
        for cls, sc, x1, y1, x2, y2 in dets:
            print(f"  {hyp['names'][cls]:15s}  {sc:5.2f}  "
                  f"({x1:4.0f},{y1:4.0f}) – ({x2:4.0f},{y2:4.0f})")

        # Load the image again for drawing, this time in color if it's grayscale
        # or just in its original format.
        img_to_draw = cv2.imread(args.image)
        if img_to_draw is None:
            print(f"Error: Could not load image for drawing: {args.image}")
            return

        # Ensure it's a 3-channel image for drawing colored boxes
        if len(img_to_draw.shape) == 2: # if grayscale
            img_to_draw = cv2.cvtColor(img_to_draw, cv2.COLOR_GRAY2BGR)


        for cls, sc, x1, y1, x2, y2 in dets:
            # Generate a consistent color for each class
            # You can customize these colors or use a more sophisticated color mapping
            color = [(c * 50) % 255 for c in (cls, cls + 1, cls + 2)] # Simple color generation
            color = (int(color[0]), int(color[1]), int(color[2]))

            cv2.rectangle(img_to_draw, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(img_to_draw, f"{hyp['names'][cls]} {sc:.2f}",
                        (int(x1), int(y1) - 10 if int(y1) - 10 > 10 else int(y1) + 15), # Adjust text position
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2) # Larger font, thicker line

        if args.save:
            out_name = Path(args.image).stem + "_baseline.jpg"
            cv2.imwrite(out_name, img_to_draw)
            print(f"\n🖼  Saved visualised output to {out_name}")

        if args.show: # New block for displaying image
            window_name = f"Detections on {Path(args.image).name}"
            cv2.imshow(window_name, img_to_draw)
            print("Press any key to close the displayed image.")
            cv2.waitKey(0)  # Wait indefinitely until a key is pressed
            cv2.destroyAllWindows() # Close all OpenCV windows

    else:  # --val ----------------------------------------------------------
        if os.path.isdir(args.val):
            files = sorted(
                f for ext in ("*.jpg", "*.png", "*.jpeg", "*.bmp")
                for f in glob.glob(os.path.join(args.val, ext))
            )
        else:  # txt file with one image per line
            with open(args.val) as f:
                files = [ln.strip() for ln in f if ln.strip()]
        if not files:
            raise FileNotFoundError(f"No images found in {args.val}")

        evaluate_dataset(
            files, yolos, weights, device,
            input_sz=args.input_size, conf_thres=args.conf, iou_thres=args.iou
        )


if __name__ == "__main__":
    main()