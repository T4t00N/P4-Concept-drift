#!/usr/bin/env python3
# train.py – frozen-YOLO, frozen-MoCo, batch-size 1.  Only the MLP learns.
# ------------------------------------------------------------------------------
import argparse
import copy
import glob
import os
import time
from pathlib import Path
from contextlib import nullcontext

import numpy as np
import torch
from torch.utils import data
import tqdm
import yaml

# ---------------- third-party ---------------------------------------------------
import wandb  # experiment tracking

# ---------------- local project imports ----------------------------------------
import models.MoCo_inference as fv
from nets.mlp_net import MLP
from nets import nn  # YOLO back-bones
from utils import util
from utils.dataset import Dataset

# ---------------- wandb settings ------------------------------------------------
WANDB_API_KEY = "b881bb0c188ba3a391651a118e8cbcd3fc00a212"
WANDB_PROJECT = "P4_model"

# ------------------------------------------------------------------------------
# 1. helpers
# ------------------------------------------------------------------------------
import warnings, logging

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("py.warnings").setLevel(logging.ERROR)
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"


def init_moco(device: str):
    """Load a **frozen** MoCo-v2 encoder checkpoint."""
    ckpt = "/ceph/project/P4-concept-drift/YOLOv8-Anton/data/moco_epoch_100.pt"
    model = fv.load_moco_model(ckpt, device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def init_mlp(input_dim=128, hidden_dim=512, num_experts=3, device="cpu"):
    return MLP(input_dim, hidden_dim, num_experts).to(device)


def init_yolos(num_classes: int, device: str):
    """Return frozen YOLO-v8-n models (one per cluster) whose heads stay in train mode."""
    models = [nn.yolo_v8_n(num_classes).to(device) for _ in range(3)]
    for m in models:
        m.eval()  # freeze BN statistics etc.
        m.head.train()  # but we still need raw (DFL) outputs
        for p in m.parameters():  # keep all weights frozen
            p.requires_grad_(False)
    return models


def load_yolo_weights(model: torch.nn.Module, ckpt_path: str, device: str):
    """Load a YOLO checkpoint, copying only shape-compatible tensors."""
    obj = torch.load(ckpt_path, map_location=device)
    if isinstance(obj, dict) and "model" in obj:
        obj = obj["model"]
    if isinstance(obj, torch.nn.Module):
        obj = obj.state_dict()
    if not isinstance(obj, dict):
        raise TypeError(f"{ckpt_path}: unsupported checkpoint type {type(obj)}")

    model_state = model.state_dict()
    compatible = {k: v for k, v in obj.items()
                  if k in model_state and v.shape == model_state[k].shape}
    skipped = [k for k in obj.keys() if k not in compatible]

    model.load_state_dict(compatible, strict=False)
    print(f"[YOLO] {os.path.basename(ckpt_path)} → "
          f"copied {len(compatible):4d} tensors, skipped {len(skipped):2d} (shape mismatch)")


# ------------------------------------------------------------------------------
# 2. loss wrapper
# ------------------------------------------------------------------------------

def compute_weighted_loss(
        moco_model: torch.nn.Module,
        mlp: torch.nn.Module,
        yolo_models: list[torch.nn.Module],
        images: torch.Tensor,
        targets: torch.Tensor,
        criterion,
        temperature: float,  # No default here, passed from hyp
        ent_weight: float,  # No default here, passed dynamically
        elastic_alpha: float = 0.0,  # New: Elasticity penalty weight
        return_weights: bool = False  # Added argument to return weights
):
    """
    MoCo → MLP → soft weights → weighted YOLO loss (YOLOs are frozen).
    Adds an entropy term –λ·H(w) to avoid weight collapse.
    Adds an elastic penalty to prevent extreme weight deviation.
    """
    # ------------------------------------------------------------------ #
    # 1. Features from **frozen** MoCo
    img3 = images.expand(-1, 3, -1, -1) if images.shape[1] == 1 else images
    with torch.no_grad():
        _, feats = moco_model(img3)  # (B, 128)

    # 2. Gating weights from trainable MLP
    logits = mlp(feats)  # (B, 3)
    weights = torch.softmax(logits / temperature, dim=1)

    # Guarantee minimum weight per expert
    min_w = 0.2
    n_exp = weights.size(1)
    weights = weights * (1.0 - n_exp * min_w) + min_w  # (B, 3)

    # 3. Expert losses – forward pass inside no-grad (YOLOs are frozen)
    batch_losses = []
    with torch.no_grad():
        for m in yolo_models:
            outputs = m(images, return_feats=True)  # Get both feats and outputs
            batch_losses.append(criterion(outputs, targets))  # scalar
    losses = torch.stack(batch_losses)  # (3,)

    # 4. Weighted sum (average weights across batch) DOES not matter with batch size 1
    avg_weights = weights.mean(0)  # (3,)
    main_loss = (avg_weights * losses).sum()  # scalar

    # 5. Entropy bonus
    entropy = -(weights * (weights + 1e-8).log()).sum(1).mean()
    entropy_loss = ent_weight * entropy

    # 6. Elasticity penalty: Forces weights to stay closer to average distribution
    elastic_loss = 0.0
    if elastic_alpha > 0:
        target_avg_weight = 1.0 / n_exp  # e.g., 1/3 for 3 experts
        deviation = (weights - target_avg_weight).pow(2).sum(1).mean()
        elastic_loss = elastic_alpha * deviation

    outputs = m(images)
    if isinstance(outputs, tuple):
        feats, x = outputs

    total_loss = main_loss - entropy_loss + elastic_loss  # Note: entropy is maximized, so its term is subtracted

    if return_weights:
        # Now, return the per-sample weights as well for inspection
        return total_loss, avg_weights, weights
    else:
        return total_loss


# ------------------------------------------------------------------------------
# 3. LR schedule helper
# ------------------------------------------------------------------------------

def lr_lambda(epoch: int, epochs: int, lrf: float):
    return (1 - epoch / epochs) * (1.0 - lrf) + lrf


# ------------------------------------------------------------------------------
# 4. main training routine – only the MLP learns
# ------------------------------------------------------------------------------

def train(args, hyp):
    rank, world, device = args.local_rank, args.world_size, args.device

    # ---------------- wandb initialisation ------------------------------
    if rank == 0:
        wandb.login(key=WANDB_API_KEY, relogin=True)
        cfg_serializable = {**vars(args),
                            **{k: v for k, v in hyp.items()
                               if isinstance(v, (int, float, str, bool))}}
        wandb_run = wandb.init(project=WANDB_PROJECT,
                               name=f"run_{int(time.time())}",
                               config=cfg_serializable,
                               reinit=True)
    else:
        wandb_run = None

    # ---------------- dataset -------------------------------------------
    train_list = Path(hyp["train_list"]).expanduser()
    assert train_list.is_file(), f"train_list file not found: {train_list}"
    with open(train_list) as f:
        filenames = [p.strip() for p in f if p.strip()]

    ds = Dataset(filenames, args.input_size, hyp, augment=True)
    sampler = data.distributed.DistributedSampler(ds) if world > 1 else None
    dl = data.DataLoader(
        ds,
        batch_size=1,  # <-- forced batch-size = 1
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        collate_fn=Dataset.collate_fn,
    )

    # ---------------- models --------------------------------------------
    moco = init_moco(device)
    mlp = init_mlp(device=device)
    yolos = init_yolos(num_classes=len(hyp["names"]), device=device)

    # load checkpoints for the three YOLO experts
    for model, ck in zip(yolos, args.yolo_ckpts):
        load_yolo_weights(model, ck, device)

    criterion = util.ComputeLoss(yolos[0], hyp)

    # ---------------- DDP (only the MLP needs it) ------------------------
    if world > 1:
        mlp = torch.nn.parallel.DistributedDataParallel(
            mlp, device_ids=[rank], output_device=rank)

    # ---------------- optimizer / scheduler -----------------------------
    hyp["weight_decay"] *= 1 / 64  # small net
    opt = torch.optim.SGD(mlp.parameters(),
                          hyp["lr0"],
                          hyp["momentum"],
                          nesterov=True,
                          weight_decay=hyp["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda ep: lr_lambda(ep, args.epochs, hyp["lrf"]))

    # AMP helpers ---------------------------------------------------------
    if device == "cuda" and torch.cuda.is_available():
        scaler, autocast = torch.cuda.amp.GradScaler(), torch.cuda.amp.autocast
    else:
        scaler, autocast = torch.amp.GradScaler(enabled=False), nullcontext

    util.setup_multi_processes()

    # model-saving helper
    def save_model(mlp, save_path):
        # Handle DDP-wrapped model
        model_state = mlp.module.state_dict() if world > 1 else mlp.state_dict()
        torch.save({"mlp": model_state}, save_path)

    # --- Dynamic Entropy Weight Setup ---
    entropy_start_value = hyp.get("entropy_start_value", 50.0)
    entropy_end_value = hyp.get("entropy_end_value", 0.5)
    entropy_decay_iterations = max(1, hyp.get("entropy_decay_iterations", 1000))

    # --- Elastic Bounds Setup ---
    elastic_alpha = hyp.get("elastic_alpha", 1.0)  # Start with a reasonable value like 1.0 or 0.5

    # ---------------- training loop -------------------------------------
    num_batches = len(dl)
    num_warmup = max(round(hyp["warmup_epochs"] * num_batches), 100)  # shorter
    accumulate = 1  # no grad-accum

    for ep in range(args.epochs):
        if sampler:
            sampler.set_epoch(ep)
        mlp.train()  # YOLOs / MoCo are frozen

        pbar = tqdm.tqdm(enumerate(dl),
                         total=num_batches,
                         disable=(rank != 0),
                         desc=f"Epoch {ep + 1}/{args.epochs}")
        avg = util.AverageMeter()

        opt.zero_grad(set_to_none=True)
        for i, (samples, targets, _) in pbar:
            seen_batches = i + num_batches * ep
            samples = samples.to(device).float() / 255
            targets = targets.to(device)

            # --- Calculate current entropy weight dynamically ---
            if seen_batches < entropy_decay_iterations:
                current_ent_weight = entropy_start_value + \
                                     (entropy_end_value - entropy_start_value) * \
                                     (seen_batches / entropy_decay_iterations)
            else:
                current_ent_weight = entropy_end_value

            # warm-up (LR only, no accumulate scaling) -------------------
            if seen_batches <= num_warmup:
                xi = [0, num_warmup]
                for j, pg in enumerate(opt.param_groups):
                    # In this setup, there's only one parameter group, so we simplify
                    pg["lr"] = np.interp(
                        seen_batches, xi,
                        [hyp["warmup_bias_lr"],
                         pg["initial_lr"] * lr_lambda(ep, args.epochs, hyp["lrf"])])

            with autocast():
                # Pass the dynamically calculated current_ent_weight and temperature from hyp,
                # plus the new elastic_alpha.
                # Now, compute_weighted_loss returns total_loss, avg_weights, AND weights
                loss, avg_weights, sample_weights = compute_weighted_loss(
                    moco, mlp, yolos, samples, targets, criterion,
                    temperature=hyp["temperature"],
                    ent_weight=current_ent_weight,
                    elastic_alpha=elastic_alpha,
                    return_weights=True  # Make sure this is True to get the weights
                )

            # --- COLLAPSE AVOIDANCE MECHANISM [NEW] ---
            # Checks if the average weights have collapsed into a 0.6, 0.2, 0.2 permutation.
            # If so, it randomizes the MLP weights to escape this state.

            # Target weights are sorted to make the check order-invariant.
            target_collapse_state = torch.tensor([0.2, 0.2, 0.6], device=device)
            sorted_avg_weights, _ = torch.sort(avg_weights)

            # Check if current weights are very close to the collapse state.
            if torch.allclose(sorted_avg_weights, target_collapse_state, atol=1e-2):
                if rank == 0:  # Log only on the main process
                    print("\n--- COLLAPSE CONDITION DETECTED ---")
                    print(f"  Weights ~{avg_weights.cpu().numpy().round(3)}. Randomizing MLP to escape.")
                    print("-----------------------------------\n")

                # Access the underlying model, handling the DDP wrapper if present.
                model_to_randomize = mlp.module if world > 1 else mlp

                # Re-initialize the MLP parameters with new random values.
                with torch.no_grad():
                    for param in model_to_randomize.parameters():
                        # A simple randomization from a uniform distribution.
                        param.data.uniform_(-0.1, 0.1)

            scaler.scale(loss).backward()

            if (seen_batches + 1) % accumulate == 0:
                scaler.unscale_(opt)
                util.clip_gradients(mlp)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

            avg.update(loss.item(), samples.size(0))


            # -------- Save model at 3000 iterations --------
            if rank == 0 and (seen_batches + 1) == 3000:
                ckpt_dir = Path("weights/checkpoints")
                ckpt_dir.mkdir(exist_ok=True, parents=True)
                ckpt_path = ckpt_dir / "iter_3000.pt"
                save_model(mlp, ckpt_path)
                if wandb_run:  # Check if wandb is initialized
                    wandb.save(str(ckpt_path))
                print(f"\nCheckpoint saved at iteration 3000 to {ckpt_path}")

            if (i + 1) % 100 == 0 and rank == 0:  # only in main process
                current = i + 1
                pbar.set_description(
                    f"Epoch {ep + 1}/{args.epochs} | "
                    f"Iter {current:>5}/{num_batches} | "
                    f"Loss {loss.item():.4f}"
                )

                # --- Print MLP's output weights for YOLO Models every 100 iterations ---
                if (i + 1) % 100 == 0:
                    print("\n--- MLP Output Weights for YOLO Models (Iteration {}) ---".format(current))
                    print(f"  Average Weights per Expert: {[f'{w:.4f}' for w in avg_weights.tolist()]}")
                    # Sample and print 3 random weights per expert
                    num_experts = sample_weights.size(1)
                    num_samples_to_show = min(3, sample_weights.size(0))  # Show up to 3 samples if batch size allows

                    # Randomly select indices for the samples
                    if sample_weights.size(0) > 0:  # Ensure there are samples in the batch
                        sample_indices = torch.randperm(sample_weights.size(0))[:num_samples_to_show]
                        for idx in sample_indices:
                            print(
                                f"    Sampled Weights (Batch Item {idx.item()}): {[f'{w:.4f}' for w in sample_weights[idx].tolist()]}")
                    else:
                        print("    No samples in batch to display individual weights.")

                    print(f"  Current Entropy Weight: {current_ent_weight:.4f}")
                    print(f"  Elastic Alpha (Penalty): {elastic_alpha:.4f}")
                    print("----------------------------------------------------\n")

        sched.step()

        # -------- checkpoint every 2 epochs ------------------------------
        if rank == 0 and (ep + 1) % 2 == 0:
            ckpt_dir = Path("weights/checkpoints")
            ckpt_dir.mkdir(exist_ok=True, parents=True)
            ckpt_path = ckpt_dir / f"epoch_{ep + 1}.pt"
            save_model(mlp, ckpt_path)
            wandb.save(str(ckpt_path))
            print(f"Checkpoint saved at epoch {ep + 1}")

        # -------- wandb logging -----------------------------------------
        if rank == 0:
            wandb.log({"epoch": ep + 1,
                       "loss": avg.avg,
                       "lr": sched.get_last_lr()[0],
                       "current_ent_weight": current_ent_weight,
                       "elastic_alpha": elastic_alpha})

    # ---------------- final save ----------------------------------------
    if rank == 0:
        save_dir = Path("weights")
        save_dir.mkdir(exist_ok=True)
        save_path = save_dir / "final.pt"
        save_model(mlp, save_path)
        wandb.save(str(save_path))
        wandb_run.finish()


# ------------------------------------------------------------------------------
# 5. CLI
# ------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=1,  # <-- forced 1
                        help="*Ignored*: training is hard-wired to batch-size 1")
    parser.add_argument("--epochs", type=int, default=4)  # longer – tiny net
    parser.add_argument("--train", action="store_true", default=True)
    parser.add_argument("--local_rank", type=int, default=int(os.getenv("LOCAL_RANK", 0)))
    parser.add_argument("--world_size", type=int, default=int(os.getenv("WORLD_SIZE", 1)))
    parser.add_argument(
        "--yolo_ckpts",
        nargs=3,
        metavar=("CKPT0", "CKPT1", "CKPT2"),
        default=["last_1.pt",
                 "last_2.pt",
                 "last_3.pt"],
        help="paths to the three cluster-specific YOLO checkpoints",
    )
    args, _ = parser.parse_known_args()

    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.world_size > 1 and args.device == "cuda":
        torch.cuda.set_device(args.local_rank % torch.cuda.device_count())
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

    hyp_path = Path("utils") / "args.yaml"
    with open(hyp_path) as f:
        hyp = yaml.safe_load(f)

    # --- Hyperparameters for dynamic entropy and new elastic bounds ---
    hyp["temperature"] = hyp.get("temperature", 10.0)

    # Entropy decay parameters
    hyp["entropy_start_value"] = hyp.get("entropy_start_value", 50.0)
    hyp["entropy_end_value"] = hyp.get("entropy_end_value", 0.5)
    hyp["entropy_decay_iterations"] = hyp.get("entropy_decay_iterations", 1000)

    # New: Elasticity penalty weight
    hyp["elastic_alpha"] = hyp.get("elastic_alpha", 1.0)  # Initial value for the elastic penalty

    if args.train:
        train(args, hyp)


if __name__ == "__main__":
    main()