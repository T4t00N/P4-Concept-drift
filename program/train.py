import torch
import argparse
import os, yaml, glob
from torch.utils import data
import tqdm, numpy, copy

# ----- local imports ------
import models.MoCo_inference as fv
from nets.mlp_net import MLP
from nets import nn                     # YOLO backbone
from utils import util                  # same helpers as yolo.py
from utils.dataset import Dataset


def init_moco(args):
    """
    Initialize MoCo and feed image

    """
    #Select MoCo model
    moco_model_path = '/ceph/project/P4-concept-drift/YOLOv8-Anton/data/moco_epoch_100.pt'
    # Load the model
    moco_model = fv.load_moco_model(moco_model_path, args.device)
    return moco_model

def init_mlp(input_dim=128, hidden_dim=512, num_experts=3, device='cpu'):
    mlp = MLP(input_dim, hidden_dim, num_experts).to(device)
    return mlp

def init_yolos(num_classes, device):
    # three independent YOLO‑v8‑n models
    return [nn.yolo_v8_n(num_classes).to(device) for _ in range(3)]


def compute_weighted_loss(moco_model, mlp, yolo_models, image, targets,
                          criterion,  # your util.ComputeLoss
                          device='cpu', temperature=0.1):
    # 1) feature
    with torch.no_grad():  # (freeze MoCo)
        _, moco_features = moco_model(image)  # (B, 128)
    # 2) soft weights
    logits = mlp(moco_features)  # (B, 3)
    weights = torch.softmax(logits / temperature, dim=1)  # (B, 3)

    # 3) YOLO losses (run each model, same targets)
    losses = []
    for ym in yolo_models:
        y_out = ym(image)  # forward
        loss_y = criterion(y_out, targets)  # scalar
        losses.append(loss_y)

    losses = torch.stack(losses, dim=1)  # (B, 3)
    weighted = (weights * losses).sum(dim=1).mean()  # mean over batch
    return weighted


def lr_lambda(epoch, epochs, lrf):
    return (1 - epoch / epochs) * (1.0 - lrf) + lrf


def train(args, params):
    rank, world = args.local_rank, args.world_size
    device      = args.device

    # ---------------- dataset ----------------------------------------------
    with open(params['train_list']) as f:             # e.g. '/.../train.txt'
        filenames = [p.strip() for p in f]
    ds   = Dataset(filenames, args.input_size, params, augment=True)
    smp  = data.distributed.DistributedSampler(ds) if world > 1 else None
    dl   = data.DataLoader(ds,
                           batch_size=args.batch_size,
                           shuffle=(smp is None),
                           sampler=smp,
                           num_workers=8,
                           pin_memory=True,
                           collate_fn=Dataset.collate_fn)

    # --------------- models -------------------------------------------------
    moco = init_moco(args)                            # eval/frozen
    mlp  = init_mlp(device)
    yolos = init_yolos(num_classes=len(params['names']), device=device)

    # For ComputeLoss we only need the structure of one YOLO
    criterion = util.ComputeLoss(yolos[0], params)

    # --------------- DDP wrap ----------------------------------------------
    if world > 1:
        yolos = [torch.nn.SyncBatchNorm.convert_sync_batchnorm(y) for y in yolos]
        yolos = [torch.nn.parallel.DistributedDataParallel(
                    module=y, device_ids=[rank], output_device=rank) for y in yolos]
        mlp   = torch.nn.parallel.DistributedDataParallel(
                    module=mlp, device_ids=[rank], output_device=rank)

    # --------------- optimiser & sched -------------------------------------
    params['weight_decay'] *= args.batch_size * world / 64
    trainable   = list(mlp.parameters()) + [p for y in yolos for p in y.parameters()]
    opt         = torch.optim.SGD(trainable, params['lr0'], params['momentum'],
                                  nesterov=True, weight_decay=params['weight_decay'])
    scheduler   = torch.optim.lr_scheduler.LambdaLR(
                        opt, lambda ep: lr_lambda(ep, args.epochs, params['lrf']))

    amp_scaler  = torch.cuda.amp.GradScaler()
    util.setup_seed()
    util.setup_multi_processes()

    # --------------- training loop -----------------------------------------
    num_batches = len(dl)
    num_warmup  = max(round(params['warmup_epochs'] * num_batches), 1000)
    for ep in range(args.epochs):
        if smp: smp.set_epoch(ep)
        mlp.train();  [y.train() for y in yolos]

        avg = util.AverageMeter()
        pbar = tqdm.tqdm(enumerate(dl), total=num_batches,
                         disable=(rank!=0), desc=f'Epoch {ep+1}/{args.epochs}')

        opt.zero_grad(set_to_none=True)
        for i,(samples, targets, _) in pbar:
            x = i + num_batches*ep           # number of seen batches
            samples = samples.to(device).float()/255
            targets = targets.to(device)

            # warm‑up --------------------------------------------------------
            if x <= num_warmup:
                xi = [0, num_warmup]
                accumulate = max(1, numpy.interp(x, xi, [1, 64/(args.batch_size*world)]).round())
                for j, pg in enumerate(opt.param_groups):
                    if j == 0:  # biases
                        pg['lr'] = numpy.interp(x, xi,
                                   [params['warmup_bias_lr'], pg['initial_lr']*scheduler.ep+1])
                    else:
                        pg['lr'] = numpy.interp(x, xi,
                                   [0.0, pg['initial_lr']*scheduler.ep+1])

            with torch.cuda.amp.autocast():
                loss = compute_weighted_loss(moco, mlp, yolos,
                                             samples, targets, criterion)

            amp_scaler.scale(loss).backward()
            if (x+1) % accumulate == 0:
                amp_scaler.unscale_(opt)
                util.clip_gradients(mlp)      # (simple clipping helper)
                [util.clip_gradients(y) for y in yolos]
                amp_scaler.step(opt)
                amp_scaler.update()
                opt.zero_grad(set_to_none=True)

            avg.update(loss.item(), samples.size(0))
            if rank==0:
                pbar.set_description(f'Epoch {ep+1}/{args.epochs}  '
                                     f'loss {avg.avg:.4f}')

        scheduler.step()
        # --- (insert evaluation / checkpoint logic here if required) -----

    if rank==0:
        print('Training finished. Save final weights …')
        torch.save({'mlp': mlp.state_dict(),
                    'yolo1': yolos[0].state_dict(),
                    'yolo2': yolos[1].state_dict(),
                    'yolo3': yolos[2].state_dict()}, 'weights/final.pt')

# ---------------------------------------------------------------------------
# 3.  CLI / entry‑point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # mirror the flags present in yolo.py
    parser.add_argument('--input-size', default=640, type=int)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--epochs', default=5,   type=int)
    parser.add_argument('--train', action='store_true', default=True)
    parser.add_argument('--local_rank', type=int, default=int(os.getenv('LOCAL_RANK', 0)))
    parser.add_argument('--world_size', type=int, default=int(os.getenv('WORLD_SIZE', 1)))
    args,_ = parser.parse_known_args()

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.world_size > 1:
        torch.cuda.set_device(args.local_rank % torch.cuda.device_count())
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    with open(os.path.join('utils', 'args.yaml')) as f:
        params = yaml.safe_load(f)

    if args.train:
        train(args, params)