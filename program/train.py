import argparse
import copy
import glob
import os
from pathlib import Path
from contextlib import nullcontext

import numpy as np
import torch
from torch.utils import data
import tqdm
import yaml

import os
import warnings
import logging

# ---------------------------------------------------------------------------
# 0. local project imports ---------------------------------------------------
# ---------------------------------------------------------------------------
# Ensure these modules are resolvable in PYTHONPATH.
import models.MoCo_inference as fv
from nets.mlp_net import MLP
from nets import nn  # YOLO back‑bones
from utils import util
from utils.dataset import Dataset

# ---------------------------------------------------------------------------
# 1. helpers ---------------------------------------------------------------
# ---------------------------------------------------------------------------

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Torch and tqdm sometimes log through the logging module – keep those quiet too
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("py.warnings").setLevel(logging.ERROR)

# If you ever launch through torchrun/torch.distributed, this env‑var
# suppresses the extra “W04…” banner lines printed by torchrun itself.
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"

def init_moco(device: str):
    """Load a frozen MoCo‑v2 encoder checkpoint."""
    ckpt = '/ceph/project/P4-concept-drift/YOLOv8-Anton/data/moco_epoch_100.pt'
    return fv.load_moco_model(ckpt, device)


def init_mlp(input_dim=128, hidden_dim=512, num_experts=3, device='cpu'):
    return MLP(input_dim, hidden_dim, num_experts).to(device)


def init_yolos(num_classes: int, device: str):
    """Return three independent YOLO‑v8‑n models."""
    return [nn.yolo_v8_n(num_classes).to(device) for _ in range(3)]


# ---------------------------------------------------------------------------
# 2. loss wrapper -----------------------------------------------------------
# ---------------------------------------------------------------------------

def compute_weighted_loss(
    moco_model: torch.nn.Module,
    mlp: torch.nn.Module,
    yolo_models: list,
    images: torch.Tensor,  # B×1×H×W in this project
    targets: torch.Tensor,
    criterion,
    temperature: float = 0.1,
):
    """Compute ensemble loss with differentiable MoCo‑driven weighting.

    The YOLO criterion returns a *scalar* batch loss, so we collect three
    scalars (one per model).  Broadcasting lets us still weight them with the
    per‑image softmax scores coming from the MLP.
    """
    # ---------------- 1. MoCo features ------------------------------------
    with torch.no_grad():
        img3 = images.expand(-1, 3, -1, -1) if images.shape[1] == 1 else images
        _, feats = moco_model(img3)  # (B, 128)

    # ---------------- 2. Soft weights -------------------------------------
    logits = mlp(feats)                 # (B, 3)
    weights = torch.softmax(logits / temperature, dim=1)

    # ---------------- 3. YOLO losses --------------------------------------
    # criterion returns scalar → list of three scalars
    loss_list = [criterion(model(images), targets) for model in yolo_models]
    losses = torch.stack(loss_list)     # (3,)

    # Broadcast (B,3) * (3,) → (B,3)
    weighted = (weights * losses).sum(dim=1).mean()
    return weighted


# ---------------------------------------------------------------------------
# 3. LR schedule helper -----------------------------------------------------
# ---------------------------------------------------------------------------

def lr_lambda(epoch: int, epochs: int, lrf: float):
    return (1 - epoch / epochs) * (1.0 - lrf) + lrf


# ---------------------------------------------------------------------------
# 4. main training routine --------------------------------------------------
# ---------------------------------------------------------------------------

def train(args, hyp):
    rank, world, device = args.local_rank, args.world_size, args.device

    # ---------------- dataset ---------------------------------------------
    train_list = Path(hyp['train_list']).expanduser()
    assert train_list.is_file(), f"train_list file not found: {train_list}"
    with open(train_list) as f:
        filenames = [p.strip() for p in f if p.strip()]

    ds = Dataset(filenames, args.input_size, hyp, augment=True)
    sampler = data.distributed.DistributedSampler(ds) if world > 1 else None
    dl = data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=8,
        pin_memory=True,
        collate_fn=Dataset.collate_fn,
    )

    # ---------------- models ----------------------------------------------
    moco = init_moco(device)
    moco.eval()  # keep frozen BN stats

    mlp = init_mlp(device=device)
    yolos = init_yolos(num_classes=len(hyp['names']), device=device)

    criterion = util.ComputeLoss(yolos[0], hyp)

    # ---------------- DDP --------------------------------------------------
    if world > 1:
        yolos = [torch.nn.SyncBatchNorm.convert_sync_batchnorm(y) for y in yolos]
        yolos = [torch.nn.parallel.DistributedDataParallel(y, device_ids=[rank], output_device=rank) for y in yolos]
        mlp = torch.nn.parallel.DistributedDataParallel(mlp, device_ids=[rank], output_device=rank)

    # ---------------- opt & sched -----------------------------------------
    hyp['weight_decay'] *= args.batch_size * world / 64
    trainable = list(mlp.parameters()) + [p for y in yolos for p in y.parameters()]
    opt = torch.optim.SGD(trainable, hyp['lr0'], hyp['momentum'], nesterov=True, weight_decay=hyp['weight_decay'])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda ep: lr_lambda(ep, args.epochs, hyp['lrf']))

    # AMP helpers -----------------------------------------------------------
    if device == 'cuda' and torch.cuda.is_available():
        scaler = torch.cuda.amp.GradScaler()
        autocast = torch.cuda.amp.autocast
    else:
        scaler = torch.amp.GradScaler(enabled=False)
        autocast = nullcontext  # no‑op on CPU

    util.setup_seed()
    util.setup_multi_processes()

    # ---------------- loop -------------------------------------------------
    num_batches = len(dl)
    num_warmup = max(round(hyp['warmup_epochs'] * num_batches), 1000)
    accumulate = 1  # dynamic gradient‑accumulation factor

    for ep in range(args.epochs):
        if sampler:
            sampler.set_epoch(ep)
        mlp.train();  [y.train() for y in yolos]

        pbar = tqdm.tqdm(enumerate(dl), total=num_batches, disable=(rank != 0), desc=f"Epoch {ep+1}/{args.epochs}")
        avg = util.AverageMeter()

        opt.zero_grad(set_to_none=True)
        for i, (samples, targets, _) in pbar:
            seen_batches = i + num_batches * ep
            samples = samples.to(device).float() / 255
            targets = targets.to(device)

            # warm‑up -------------------------------------------------------
            if seen_batches <= num_warmup:
                xi = [0, num_warmup]
                accumulate = max(1, np.interp(seen_batches, xi, [1, 64 / (args.batch_size * world)]).round())
                for j, pg in enumerate(opt.param_groups):
                    if j == 0:  # biases
                        pg['lr'] = np.interp(seen_batches, xi, [hyp['warmup_bias_lr'], pg['initial_lr'] * lr_lambda(ep, args.epochs, hyp['lrf'])])
                    else:
                        pg['lr'] = np.interp(seen_batches, xi, [0.0, pg['initial_lr'] * lr_lambda(ep, args.epochs, hyp['lrf'])])

            with autocast():
                loss = compute_weighted_loss(moco, mlp, yolos, samples, targets, criterion)

            scaler.scale(loss).backward()
            if (seen_batches + 1) % accumulate == 0:
                scaler.unscale_(opt)
                util.clip_gradients(mlp)
                [util.clip_gradients(y) for y in yolos]
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

            avg.update(loss.item(), samples.size(0))
            if rank == 0:
                pbar.set_description(f"Epoch {ep+1}/{args.epochs}  loss {avg.avg:.4f}")

        sched.step()

    if rank == 0:
        print('Training finished → saving weights …')
        save_dir = Path('weights'); save_dir.mkdir(exist_ok=True)
        torch.save({'mlp': mlp.state_dict(),
                    'yolo1': yolos[0].state_dict() if not isinstance(yolos[0], torch.nn.parallel.DistributedDataParallel) else yolos[0].module.state_dict(),
                    'yolo2': yolos[1].state_dict() if not isinstance(yolos[1], torch.nn.parallel.DistributedDataParallel) else yolos[1].module.state_dict(),
                    'yolo3': yolos[2].state_dict() if not isinstance(yolos[2], torch.nn.parallel.DistributedDataParallel) else yolos[2].module.state_dict()}, save_dir / 'final.pt')


# ---------------------------------------------------------------------------
# 5. CLI --------------------------------------------------------------------
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-size', type=int, default=384)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--train', action='store_true', default=True)
    parser.add_argument('--local_rank', type=int, default=int(os.getenv('LOCAL_RANK', 0)))
    parser.add_argument('--world_size', type=int, default=int(os.getenv('WORLD_SIZE', 1)))
    args, _ = parser.parse_known_args()

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.world_size > 1 and args.device == 'cuda':
        torch.cuda.set_device(args.local_rank % torch.cuda.device_count())
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    hyp_path = Path('utils') / 'args.yaml'
    with open(hyp_path) as f:
        hyp = yaml.safe_load(f)

    if args.train:
        train(args, hyp)


if __name__ == '__main__':
    main()
