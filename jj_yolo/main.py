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

warnings.filterwarnings("ignore")


def learning_rate(args, params):
    def fn(x):
        return (1 - x / args.epochs) * (1.0 - params['lrf']) + params['lrf']
    return fn


def train(args, params):
    # Model
    model = nn.yolo_v8_n(len(params['names'].values())).cuda()

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
    with open(f'{path}/train.txt') as reader:
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
                             num_workers=32,
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
    criterion = util.ComputeLoss(model, params)
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

                # Forward
                with torch.cuda.amp.autocast():
                    outputs = model(samples)
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
            if args.local_rank == 0:
                print(f"Epoch {epoch + 1}/{args.epochs} completed. Testing model...")
                last = test(args, params, ema.ema)
                writer.writerow({
                    'mAP': f'{last[1]:.3f}',
                    'epoch': str(epoch + 1).zfill(3),
                    'mAP@50': f'{last[0]:.3f}'
                })
                f.flush()

                # Update best
                if last[1] > best:
                    best = last[1]

                # Save model
                ckpt = {'model': copy.deepcopy(ema.ema).half()}
                torch.save(ckpt, './weights/last.pt')
                if best == last[1]:
                    torch.save(ckpt, './weights/best.pt')
                del ckpt

    # Cleanup
    if args.local_rank == 0:
        print("Finalizing training and stripping optimizer...")
        util.strip_optimizer('./weights/best.pt')
        util.strip_optimizer('./weights/last.pt')

    torch.cuda.empty_cache()


@torch.no_grad()
def test(args, params, model=None):
    filenames = []
    path = r"/ceph/project/P4-concept-drift/final_yolo_data_format/YOLOv8-pt/Dataset"
    with open(f'{path}/test.txt') as reader:
        for filepath in reader.readlines():
            filenames.append(filepath.strip())

    # Create Dataset (again, augment=False, no mosaic)
    dataset = Dataset(filenames, args.input_size, params, augment=False)
    loader = data.DataLoader(dataset, 8, shuffle=False,
                             num_workers=32, pin_memory=True,
                             collate_fn=Dataset.collate_fn)

    if model is None:
        model = torch.load('./weights/best.pt', map_location='cuda')['model'].float()

    model.half()
    model.eval()

    # iou vector for mAP@0.5:0.95
    iou_v = torch.linspace(0.5, 0.95, 10).cuda()
    n_iou = iou_v.numel()

    m_pre, m_rec, map50, mean_ap = 0., 0., 0., 0.
    metrics = []
    p_bar = tqdm.tqdm(loader, desc=('%10s' * 3) % ('precision', 'recall', 'mAP'))
    for samples, targets, shapes in p_bar:
        samples = samples.cuda().half() / 255
        _, _, height, width = samples.shape
        targets = targets.cuda()

        # Forward
        outputs = model(samples)

        # NMS
        targets[:, 2:] *= torch.tensor((width, height, width, height), device=targets.device)
        outputs = util.non_max_suppression(outputs, 0.001, 0.65)

        # Metrics
        for i, output in enumerate(outputs):
            labels = targets[targets[:, 0] == i, 1:]
            correct = torch.zeros(output.shape[0], n_iou, dtype=torch.bool).cuda()
            if output.shape[0] == 0:
                if labels.shape[0]:
                    metrics.append((correct, *torch.zeros((3, 0)).cuda()))
                continue

            detections = output.clone()
            util.scale(detections[:, :4], samples[i].shape[1:], shapes[i][0], shapes[i][1])
            if labels.shape[0]:
                tbox = labels[:, 1:5].clone()
                tbox[:, 0] = labels[:, 1] - labels[:, 3] / 2
                tbox[:, 1] = labels[:, 2] - labels[:, 4] / 2
                tbox[:, 2] = labels[:, 1] + labels[:, 3] / 2
                tbox[:, 3] = labels[:, 2] + labels[:, 4] / 2
                util.scale(tbox, samples[i].shape[1:], shapes[i][0], shapes[i][1])

                correct_arr = numpy.zeros((detections.shape[0], iou_v.shape[0]), dtype=bool)
                t_tensor = torch.cat((labels[:, 0:1], tbox), 1)
                iou = util.box_iou(t_tensor[:, 1:], detections[:, :4])
                correct_class = t_tensor[:, 0:1] == detections[:, 5]

                for j in range(n_iou):
                    x = torch.where((iou >= iou_v[j]) & correct_class)
                    if x[0].shape[0]:
                        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
                        # Filter duplicates
                        if x[0].shape[0] > 1:
                            matches = matches[matches[:, 2].argsort()[::-1]]  # sort by iou
                            matches = matches[numpy.unique(matches[:, 1], return_index=True)[1]]
                            matches = matches[numpy.unique(matches[:, 0], return_index=True)[1]]
                        correct_arr[matches[:, 1].astype(int), j] = True

                correct = torch.tensor(correct_arr, device=iou_v.device)

            metrics.append((correct, output[:, 4], output[:, 5], labels[:, 0]))

    # Compute metrics
    metrics = [torch.cat(x, 0).cpu().numpy() for x in zip(*metrics)]
    if len(metrics) and metrics[0].any():
        tp, fp, m_pre, m_rec, map50, mean_ap = util.compute_ap(*metrics)

    print('%10.3g' * 3 % (m_pre, m_rec, mean_ap))

    model.float()  # re-enable FP32 if needed
    return map50, mean_ap


def main():
    print("Starting main function")
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-size', default=384, type=int)
    parser.add_argument('--batch-size', default=128, type=int)
    # Remove: parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')

    # Use parse_known_args to ignore unrecognized arguments
    args, _ = parser.parse_known_args()

    # Set local_rank from environment variable
    args.local_rank = int(os.environ.get('LOCAL_RANK', 0))
    args.world_size = int(os.getenv('WORLD_SIZE', 1))
    print(f"Args parsed: {args}")
    print(f"Process {args.local_rank} using GPU {torch.cuda.current_device()}")

    print(f"WORLD_SIZE: {os.environ.get('WORLD_SIZE', 'not set')}")
    if args.world_size > 1:
        torch.cuda.set_device(device=args.local_rank % torch.cuda.device_count())
        print(f"Process {args.local_rank} using GPU {torch.cuda.current_device()} "
              f"out of {torch.cuda.device_count()} GPUs")
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

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

if __name__ == "__main__":
    main()
