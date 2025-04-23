# deployment/app.py
"""
FastAPI wrapper around EnsemblePredictor.

▪ start:   uvicorn deployment.app:app --port 8000 --reload
▪ browse:  http://localhost:8000
"""

import io, asyncio, concurrent.futures
from functools import partial
from pathlib import Path
from typing import Tuple, List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse

# ── local imports ─────────────────────────────────────────────────────────────
from deployment.inference_core import EnsemblePredictor

# ── paths – adjust if your project layout differs ────────────────────────────
ROOT = Path(__file__).resolve().parents[1]          # project‑root
CKPT   = ROOT / "program/weights/checkpoints/epoch_2.pt"
MOCO   = ROOT / "program/weights/checkpoints/moco_epoch_100.pt"
HYP    = ROOT / "program/utils/args.yaml"

# ── app & globals ─────────────────────────────────────────────────────────────
app      = FastAPI(title="YOLO‑Ensemble Demo", version="0.1")
predictor: EnsemblePredictor | None = None           # set in startup
executor  = concurrent.futures.ThreadPoolExecutor(max_workers=1)


# -----------------------------------------------------------------------------
# 1. life‑cycle: load the heavy models *once*
# -----------------------------------------------------------------------------
@app.on_event("startup")
async def _load_models():
    global predictor
    predictor = EnsemblePredictor(
        weights_ckpt=str(CKPT),
        moco_ckpt=str(MOCO),
        hyp_yaml=str(HYP),
        device="cpu",           # change to "cpu" if no GPU
        input_size=384,
    )


# -----------------------------------------------------------------------------
# 2. tiny HTML upload page (no JS framework needed for a demo)
# -----------------------------------------------------------------------------
UPLOAD_FORM = """
<!doctype html>
<title>YOLO‑Ensemble demo</title>
<h2>Upload an image</h2>
<form action="/predict" method="post" enctype="multipart/form-data">
  <input type="file" name="file" accept="image/*" required>
  <button type="submit">Detect</button>
</form>
"""

@app.get("/", response_class=HTMLResponse)
async def index():
    return UPLOAD_FORM


# -----------------------------------------------------------------------------
# 3. POST /predict  → returns jpeg with boxes (or JSON if ?json)
# -----------------------------------------------------------------------------
async def _run_inference(img_bytes: bytes) -> Tuple[List, bytes]:
    """Run heavy predictor in a background thread so event loop stays free."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        executor, partial(predictor.predict, img_bytes)
    )

@app.get("/healthz")
async def healthz():
    if predictor is None:
        return JSONResponse(status_code=503, content={"status": "loading"})
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...), json: bool = False):
    if predictor is None:  # should never happen
        raise HTTPException(status_code=503, detail="Model not ready yet ¯\\_(ツ)_/¯")

    img_bytes = await file.read()
    if len(img_bytes) > 8 * 1024 * 1024:           # 8 MB guard
        raise HTTPException(413, "File too large")

    dets, jpeg = await _run_inference(img_bytes)

    if json:
        # Return *only* detections (useful for JS front‑end)
        return JSONResponse(
            [{"cls": int(c), "conf": float(p), "x1": x1, "y1": y1, "x2": x2, "y2": y2}
             for c, p, x1, y1, x2, y2 in dets]
        )

    # Otherwise stream back the annotated jpeg
    return StreamingResponse(io.BytesIO(jpeg), media_type="image/jpeg")
