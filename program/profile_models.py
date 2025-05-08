#!/usr/bin/env python
# ──────────────────────────────────────────────────────────────────────────────
# profile_models.py
# Compute trainable‑parameter counts and FLOPs (no extra pip dependencies)
# ──────────────────────────────────────────────────────────────────────────────

import argparse
import torch

# --------------------------------------------------------------------------
# Re‑use the helper factories defined in your training script
# (make sure train.py is discoverable on PYTHONPATH)
# --------------------------------------------------------------------------
from train import init_moco, init_mlp, init_yolos   # noqa: E402


# ──────────────────────────────────────────────────────────────────────────────
# Bare‑bones helpers
# ──────────────────────────────────────────────────────────────────────────────
def count_trainable(model: torch.nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def quick_profile(model: torch.nn.Module,
                  sample: torch.Tensor,
                  device: torch.device) -> float:
    """
    Run a single forward pass under torch.profiler and return total FLOPs.
    Requires PyTorch ≥ 2.2 built with profiler support.
    """
    model = model.to(device).eval()
    sample = sample.to(device)

    with torch.no_grad():
        with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU],
                with_flops=True,
                record_shapes=True) as prof:
            model(sample)

    total_flops = sum(ev.flops for ev in prof.key_averages())
    return total_flops


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print #params and FLOPs of MoCo‑gate YOLO ensemble.")
    parser.add_argument("--input-size", type=int, default=384,
                        help="input resolution (H=W) used for profiling")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    # ------------------------------------------------------------------
    # Build the models exactly as training would
    # ------------------------------------------------------------------
    moco = init_moco(device)
    mlp = init_mlp(device=device)
    yolos = init_yolos(num_classes=5, device=device)   # adjust if classes differ

    dummy = torch.randn(1, 3, args.input_size, args.input_size)

    # ------------------------------------------------------------------
    # Report numbers
    # ------------------------------------------------------------------
    print("\n• TRAINABLE PARAMETERS •")
    for name, model in [
        ("MoCo‑enc", moco),
        ("MLP‑gate", mlp),
        *[(f"YOLO‑{i}", y) for i, y in enumerate(yolos)]
    ]:
        print(f"{name:10s}: {count_trainable(model):,}")

    print("\n• FLOPs per forward pass •")
    # One YOLO is representative; multiply by 3 for the ensemble.
    flops = quick_profile(yolos[0], dummy, device)
    print(f"YOLO‑v8‑n @ {args.input_size}×{args.input_size}: "
          f"{flops/1e9:,.2f} GFLOPs "
          f"(≈ {flops*3/1e9:,.2f} GFLOPs for 3×YOLO)\n")


if __name__ == "__main__":
    main()
