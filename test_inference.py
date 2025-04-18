# test_inference.py
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parent / "program"))

from deployment.inference_core import EnsemblePredictor

pred = EnsemblePredictor(
    weights_ckpt="program/weights/checkpoints/epoch_2.pt",
    moco_ckpt="program/data/moco_epoch_100.pt",
    hyp_yaml="program/utils/args.yaml",
    device="cuda"   # or "cpu"
)

with open("some_test.jpg", "rb") as f:
    dets, jpg = pred.predict(f.read())

print("Detections:", dets)
with open("out.jpg", "wb") as f:
    f.write(jpg)
print("Annotated image saved ➜ out.jpg")
