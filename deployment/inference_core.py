# yolo_web/inference_core.py
"""
Reusable MoCo ➜ MLP ➜ YOLO‑ensemble predictor.
Call   >>> predictor = EnsemblePredictor(...)
       >>> dets, jpg_bytes = predictor.predict(image_bytes)
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple

import io
import cv2
import numpy as np
import torch
import yaml

# — project imports -----------------------------------------------------------
import program.models.MoCo_inference as fv           # your MoCo helpers
from program.nets.mlp_net import MLP
from program.nets import nn                          # YOLO back‑bones
from program.utils import util
from program.utils.WBF import weighted_boxes_fusion
from program.utils.dataset import resize_static

# -----------------------------------------------------------------------------


class EnsemblePredictor:
    """
    Loads MoCo‑v2 encoder, the temperature‑softmax MLP and three YOLO‑v8‑n
    detectors once.  After that `.predict()` is ~20 ms on a V100 for 384 px.
    """

    def __init__(
        self,
        weights_ckpt: str | Path,
        moco_ckpt: str | Path,
        hyp_yaml: str | Path,
        device: str | torch.device = "cuda",
        input_size: int = 384,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.input_size = input_size

        # --------------- classes / names ----------------------------------
        with open(hyp_yaml, "r") as f:
            hyp: Dict = yaml.safe_load(f)
        self.names: Dict[int, str] = hyp["names"]

        # --------------- MoCo encoder ------------------------------------
        self.moco = fv.load_moco_model(moco_ckpt, self.device)
        self.moco.eval()

        # --------------- MLP + 3×YOLO ------------------------------------
        ckpt = torch.load(weights_ckpt, map_location=self.device)
        self.mlp = MLP().to(self.device).eval()
        self.yolos = [nn.yolo_v8_n(len(self.names)).to(self.device).eval() for _ in range(3)]

        self._load_state(self.mlp, ckpt["mlp"])
        for i, tag in enumerate(("yolo1", "yolo2", "yolo3")):
            self._load_state(self.yolos[i], ckpt[tag])

        print("✅  Ensemble models loaded & ready.")

    # ---------------------------------------------------------------------

    @staticmethod
    def _strip_module(state_dict):
        return {k.removeprefix("module."): v for k, v in state_dict.items()}

    def _load_state(self, model: torch.nn.Module, state_dict):
        model.load_state_dict(self._strip_module(state_dict), strict=True)

    # ---------------------------------------------------------------------

    @torch.inference_mode()
    def predict(
        self,
        image_bytes: bytes,
        conf: float = 0.25,
        iou: float = 0.65,
        temperature: float = 0.1,
    ) -> Tuple[List[Tuple[int, float, float, float, float, float]], bytes]:
        """
        Args
        ----
        image_bytes : raw bytes from an uploaded file.
        Returns
        -------
        detections : List[(cls_id, conf, x1, y1, x2, y2)]
        jpg_bytes  : ready‑to‑send JPEG with drawn boxes.
        """
        # ---------- decode & letterbox -----------------------------------
        img0 = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
        if img0 is None:
            raise ValueError("Could not decode image.")
        img_l, ratio, pad = resize_static(img0, self.input_size)
        tensor = (
            torch.from_numpy(img_l.astype(np.float32) / 255.0)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(self.device)
        )

        # ---------- MoCo ➜ MLP weights -----------------------------------
        x3 = tensor.repeat(1, 3, 1, 1)  # (1,3,H,W) because MoCo is RGB
        _, feats = self.moco(x3)
        weights = torch.softmax(self.mlp(feats) / temperature, dim=1)[0]  # (3,)

        # ---------- YOLO forward (+NMS) ----------------------------------
        preds = [y(tensor).float() for y in self.yolos]
        preds_nms = [util.non_max_suppression(p, conf, iou)[0] for p in preds]

        # ---------- convert to WBF format --------------------------------
        boxes_l, scores_l, labels_l = [], [], []
        for det in preds_nms:
            if det is None or det.numel() == 0:
                boxes_l.append(np.zeros((0, 4), np.float32))
                scores_l.append(np.zeros((0,), np.float32))
                labels_l.append(np.zeros((0,), np.float32))
                continue
            b = (det[:, :4] / self.input_size).cpu().numpy()
            boxes_l.append(b)
            scores_l.append(det[:, 4].cpu().numpy())
            labels_l.append(det[:, 5].cpu().numpy())

        if sum(len(b) for b in boxes_l) == 0:  # nothing found
            return [], image_bytes

        boxes_f, scores_f, labels_f = weighted_boxes_fusion(
            boxes_l,
            scores_l,
            labels_l,
            weights=weights.cpu().tolist(),
            iou_thr=0.55,
            skip_box_thr=conf,
            conf_type="max",
        )

        # ---------- back‑project to original coords ----------------------
        boxes_f *= self.input_size
        boxes_f[:, [0, 2]] -= pad[0]
        boxes_f[:, [1, 3]] -= pad[1]
        boxes_f /= ratio[0]

        detections = []
        for (x1, y1, x2, y2), sc, lb in zip(boxes_f, scores_f, labels_f):
            if sc < conf or not np.isfinite([x1, y1, x2, y2]).all():
                continue
            detections.append((int(lb), float(sc), float(x1), float(y1), float(x2), float(y2)))

        # ---------- draw --------------------------------------------------
        img_bgr = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)
        for cls, conf_, x1, y1, x2, y2 in detections:
            p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
            cv2.rectangle(img_bgr, p1, p2, (0, 255, 0), 2)
            label = f"{self.names.get(cls, cls)} {conf_:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img_bgr, (p1[0], p1[1] - th - 4), (p1[0] + tw, p1[1]), (0, 255, 0), -1)
            cv2.putText(
                img_bgr,
                label,
                (p1[0], p1[1] - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        ok, buf = cv2.imencode(".jpg", img_bgr)
        if not ok:
            raise RuntimeError("cv2.imencode failed!")
        return detections, buf.tobytes()
