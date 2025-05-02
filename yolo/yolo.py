import argparse
import copy
import csv
import os
import warnings
import numpy
import torch
import tqdm
import yaml
from torch.utils import data
from nets import nn
from utils import util
from utils.dataset import Dataset
from datetime import timedelta
warnings.filterwarnings("ignore")


def learning_rate(args, params):
    def fn(x):
        return (1 - x / args.epochs) * (1.0 - params['lrf']) + params['lrf']
    return fn


def train(args, params):
    model = nn.yolo_v8_n(len(params['names'])).cuda()

    # Optimizer
    accumulate = max(round(64 / (args.batch_size * args.world_size)), 1)
    params['weight_decay'] *= args.batch_size * args.world_size * accumulate / 64

    p = ([], [], [])
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, torch.nn.Parameter):
            p[2].append(v.bias)
        if isinstance(v, torch.nn.BatchNorm2d):
            p[1].append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, torch.nn.Parameter):
            p[0].append(v.weight)

    optimizer = torch.optim.SGD(p[2], params['lr0'], params['momentum'], nesterov=True)
    optimizer.add_param_group({'params': p[0], 'weight_decay': params['weight_decay']})
    optimizer.add_param_group({'params': p[1]})
    del p

    # Scheduler
    lr = learning_rate(args, params)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr, last_epoch=-1)

    # EMA
    ema = util.EMA(model) if args.local_rank == 0 else None

    # Load training filenames
    filenames = []
    path = r"/ceph/project/P4-concept-drift/final_yolo_data_format/YOLOv8-pt/Dataset"
    with open(f'{path}/train_name.txt') as reader:
        for filepath in reader.readlines():
            filenames.append(filepath.strip())

    # Create Dataset (pass augment=False because we removed augmentation logic)
    dataset = Dataset(filenames, args.input_size, params, augment=False)

    # Sampler for distributed training if needed
    if args.world_size <= 1:
        sampler = None
    else:
        sampler = data.distributed.DistributedSampler(dataset)

    loader = data.DataLoader(dataset,
                             batch_size=args.batch_size,
                             shuffle=(sampler is None),
                             sampler=sampler,
                             num_workers=16,
                             pin_memory=True,
                             collate_fn=Dataset.collate_fn)

    # DDP (DistributedDataParallel) if needed
    if args.world_size > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(module=model,
                                                          device_ids=[args.local_rank],
                                                          output_device=args.local_rank)

    best = 0
    num_batch = len(loader)
    amp_scale = torch.cuda.amp.GradScaler()
    criterion = util.ComputeLoss(model.module if args.world_size > 1 else model, params)
    num_warmup = max(round(params['warmup_epochs'] * num_batch), 1000)

    print(f"Starting training loop for {args.epochs} epochs...")


    # CSV logging
    with open('weights/step.csv', 'w') as f:
        if args.local_rank == 0:
            writer = csv.DictWriter(f, fieldnames=['epoch', 'mAP@50', 'mAP'])
            writer.writeheader()

        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch + 1}/{args.epochs} started.")
            model.train()

            if args.world_size > 1:
                sampler.set_epoch(epoch)

            m_loss = util.AverageMeter()
            p_bar = enumerate(loader)

            if args.local_rank == 0:
                print(('\n' + '%10s' * 3) % ('epoch', 'memory', 'loss'))
                p_bar = tqdm.tqdm(p_bar, total=num_batch)

            optimizer.zero_grad()

            for i, (samples, targets, _) in p_bar:
                x = i + num_batch * epoch
                samples = samples.cuda().float() / 255
                targets = targets.cuda()

                # Warmup
                if x <= num_warmup:
                    xp = [0, num_warmup]
                    fp = [1, 64 / (args.batch_size * args.world_size)]
                    accumulate = max(1, numpy.interp(x, xp, fp).round())
                    for j, y in enumerate(optimizer.param_groups):
                        if j == 0:
                            fp = [params['warmup_bias_lr'], y['initial_lr'] * lr(epoch)]
                        else:
                            fp = [0.0, y['initial_lr'] * lr(epoch)]
                        y['lr'] = numpy.interp(x, xp, fp)
                        if 'momentum' in y:
                            fp = [params['warmup_momentum'], params['momentum']]
                            y['momentum'] = numpy.interp(x, xp, fp)

                # Forward: Compute loss for all three models and sum them
                with torch.cuda.amp.autocast():
                    outputs = model(samples)  # was outputs_list
                    loss = criterion(outputs, targets)

                m_loss.update(loss.item(), samples.size(0))

                loss *= args.batch_size
                loss *= args.world_size

                # Backward
                amp_scale.scale(loss).backward()

                # Optimize
                if x % accumulate == 0:
                    amp_scale.unscale_(optimizer)
                    util.clip_gradients(model)
                    amp_scale.step(optimizer)
                    amp_scale.update()
                    optimizer.zero_grad()
                    if ema:
                        ema.update(model)

                # Logging
                if args.local_rank == 0:
                    memory = f'{torch.cuda.memory_reserved() / 1E9:.3g}G'
                    s = ('%10s' * 2 + '%10.4g') % (f'{epoch + 1}/{args.epochs}', memory, m_loss.avg)
                    p_bar.set_description(s)

                del loss, outputs

            # Scheduler step
            scheduler.step()

            # Validation
            last = None
            VAL_INTERVAL = 1  # validate every 5 epochs

            if (epoch + 1) % VAL_INTERVAL == 0 and not args.no_test:
                eval_model = ema.ema if ema else model  # ← new
                last = test(args, params, eval_model) # ← every rank now runs this

            # best-mAP bookkeeping only on rank-0
            if args.local_rank == 0 and last is not None and last[1] > best:
                best = last[1]

            if args.local_rank == 0:
                os.makedirs('./weights', exist_ok=True)
                torch.save({'model': copy.deepcopy(ema.ema if ema else model).half()},
                           './weights/last.pt')
                if last is not None and last[1] >= best:
                    torch.save({'model': copy.deepcopy(ema.ema if ema else model).half()},
                               './weights/best.pt')

    torch.cuda.empty_cache()


@torch.no_grad()
def test(args, params, model=None):
    filenames = []
    path = r"/ceph/project/P4-concept-drift/final_yolo_data_format/YOLOv8-pt/Dataset" # Consider making path an argument
    with open(f'{path}/val_name.txt') as reader:
        for filepath in reader.readlines():
            filenames.append(filepath.strip())

    # Create Dataset
    dataset = Dataset(filenames, args.input_size, params, augment=False)

    # NEW: shard val set across GPUs
    if args.world_size > 1:
        val_sampler = data.distributed.DistributedSampler(dataset,
                                                          shuffle=False)
    else:
        val_sampler = None

    loader = data.DataLoader(dataset,
                             batch_size=max(1, args.batch_size // args.world_size),
                             sampler=val_sampler,
                             shuffle=(val_sampler is None),
                             num_workers=16,
                             pin_memory=True,
                             collate_fn=Dataset.collate_fn)

    if model is None:
        print("Loading single-model weights…")
        model = nn.yolo_v8_n(len(params['names'])).cuda()
        ckpt_path = './weights/best.pt'  # will be created by the new save code below
        if os.path.isfile(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location='cuda')['model'])
            print("Weights loaded.")


    model.half() # Use FP16
    model.eval()

    # iou vector for mAP@0.5:0.95
    iou_v = torch.linspace(0.5, 0.95, 10).cuda() # Use half() if metrics support it, else float()
    n_iou = iou_v.numel()

    m_pre, m_rec, map50, mean_ap = 0., 0., 0., 0.
    metrics = []
    p_bar = tqdm.tqdm(loader, desc=('%10s' * 3) % ('precision', 'recall', 'mAP'))

    # WBF parameters (adjust as needed)
    wbf_iou_thr = 0.55
    wbf_skip_box_thr = 0.001
    wbf_weights = [1, 1, 1] # Equal weights for the three models

    for samples, targets, shapes in p_bar:
        samples = samples.cuda().half() / 255 # Normalize to [0, 1]
        _, _, height, width = samples.shape # Input size height/width
        targets = targets.cuda()
        targets[:, 2:] *= torch.tensor((width, height, width, height), device=targets.device) # Scale targets to input size

        # Forward pass: get outputs from all models
        with torch.cuda.amp.autocast():
            preds = model(samples)                         # tensor [B, …, 85]
        outputs = util.non_max_suppression(
            preds, 0.001, 0.65)

        # Iterate over each image in the batch
        for i, output in enumerate(outputs):
            if output is None:
                output = torch.zeros((0, 6), device=samples.device)

            labels = targets[targets[:, 0] == i, 1:]        # ground-truth for image i
            correct = torch.zeros(output.shape[0], n_iou,
                                   dtype=torch.bool, device=output.device)

            # --- Start of original metrics calculation logic ---
            # Use the fused 'output' tensor instead of single-model output

            labels = targets[targets[:, 0] == i, 1:] # Ground truth for image i
            correct = torch.zeros(output.shape[0], n_iou, dtype=torch.bool, device=output.device)

            if output.shape[0] == 0:
                if labels.shape[0]: # If no detections but there are labels
                    metrics.append((
                        correct.cpu(),  # (0, nIoU)
                        torch.zeros(0).cpu(),  # confidences  (0,)
                        torch.zeros(0, dtype=torch.float32).cpu(),  # pred classes (0,)
                        labels[:, 0].cpu()  # gt classes   (nlabels,)
                    ))# Ensure on CPU for numpy conversion later
                continue # Go to next image

            detections = output.clone()
            # Scale fused detections to original image size
            util.scale(detections[:, :4], samples[i].shape[1:], shapes[i][0], shapes[i][1])

            if labels.shape[0]:
                # Ground truth boxes are already scaled to input size, scale them to original size
                tbox = util.wh2xy(labels[:, 1:5]) # Convert target format if necessary
                util.scale(tbox, samples[i].shape[1:], shapes[i][0], shapes[i][1])
                t_tensor = torch.cat((labels[:, 0:1], tbox), 1) # Target tensor [class, x1, y1, x2, y2]

                # Calculate IoU between fused detections and ground truth
                iou = util.box_iou(t_tensor[:, 1:], detections[:, :4]) # [n_targets, n_detections]
                # Find correct matches for each IoU threshold
                correct_class = t_tensor[:, 0:1] == detections[:, 5] # Check class match [n_targets, n_detections]

                for j in range(n_iou): # For each IoU threshold
                    # Find matches: IoU >= threshold AND Class matches
                    x = torch.where((iou >= iou_v[j]) & correct_class)
                    if x[0].shape[0]:
                        # matches = [target_idx, detection_idx, iou_value]
                        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
                        if x[0].shape[0] > 1: # If more than one match
                             # Sort by IoU descending
                            matches = matches[matches[:, 2].argsort()[::-1]]
                            # Keep only best detection match for each target
                            matches = matches[numpy.unique(matches[:, 0], return_index=True)[1]]
                            # Keep only best target match for each detection
                            matches = matches[numpy.unique(matches[:, 1], return_index=True)[1]]
                        # Mark the matched detections as correct for this IoU threshold
                        correct[matches[:, 1].astype(int), j] = True

            # Append metrics: (correct_flags, detection_confidence, detection_class, target_class)
            # Ensure all tensors are moved to CPU before appending if they will be concatenated later into NumPy arrays
            metrics.append((correct.cpu(), output[:, 4].cpu(), output[:, 5].cpu(), labels[:, 0].cpu()))
            # --- End of original metrics calculation logic ---

    # Compute final metrics
    if not metrics: # Handle case where metrics list is empty
        print("No metrics recorded.")
        return 0.0, 0.0

    try:
        # Ensure all elements in metrics are tuples of tensors before concatenating
        metrics_cat = [torch.cat([m[j] for m in metrics], 0) for j in range(4)]
        # Convert to numpy for compute_ap
        metrics_np = [m.numpy() for m in metrics_cat]

        if args.local_rank == 0:
            tp, fp, m_pre, m_rec, map50, mean_ap = util.compute_ap(*metrics_np)
        else:
            tp = fp = m_pre = m_rec = map50 = mean_ap = 0.0

    except Exception as e:
         print(f"Error during metric computation: {e}")
         print(f"Metrics list length: {len(metrics)}")
         # Optionally print shapes or types of elements in metrics for debugging
         # for idx, item in enumerate(metrics):
         #     print(f"Metric item {idx}: {[t.shape if hasattr(t,'shape') else type(t) for t in item]}")
         m_pre, m_rec, map50, mean_ap = 0., 0., 0., 0. # Default values on error


    # Print metrics
    results = torch.tensor([map50, mean_ap], device='cuda')
    if args.world_size > 1:
        torch.distributed.broadcast(results, src=0)

    # Only rank-0 prints
    if args.local_rank == 0:
        print('%10s %10s %10s' % ('precision', 'recall', 'mAP_0.5:0.95'))
        print('%10.3g %10.3g %10.3g' % (m_pre, m_rec, results[1].item()))

    model.float()
    return results[0].item(), results[1].item()


def main():
    print("Starting main function")
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-size', default=384, type=int)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--no_test', action='store_true', help='Skip testing between epochs during training')

    args, _ = parser.parse_known_args()

    args.local_rank = int(os.environ.get('LOCAL_RANK', 0))
    args.world_size = int(os.getenv('WORLD_SIZE', 1))
    print(f"Args parsed: {args}")
    print(f"Process {args.local_rank} using GPU {torch.cuda.current_device()}")

    print(f"WORLD_SIZE: {os.environ.get('WORLD_SIZE', 'not set')}")
    if args.world_size > 1:
        torch.cuda.set_device(device=args.local_rank % torch.cuda.device_count())
        print(f"Process {args.local_rank} using GPU {torch.cuda.current_device()} "
              f"out of {torch.cuda.device_count()} GPUs")
        torch.distributed.init_process_group(
            backend='nccl',
            init_method='env://',
            timeout=timedelta(minutes=60))

    if args.local_rank == 0:
        if not os.path.exists('weights'):
            os.makedirs('weights')

    util.setup_seed()
    util.setup_multi_processes()

    with open(os.path.join('utils', 'args.yaml'), errors='ignore') as f:
        params = yaml.safe_load(f)

    if args.train:
        train(args, params)
    if args.test:
        test(args, params)
    if args.world_size > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()