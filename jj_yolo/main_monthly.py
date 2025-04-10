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

def load_filenames_from_csv(csv_path, cluster_id):
    """Reads a CSV file and returns filenames matching the cluster_id."""
    filenames = []
    print(f"Attempting to load filenames from CSV: {csv_path} for cluster: {cluster_id}")
    try:
        with open(csv_path, 'r', newline='') as csvfile:
            # Assuming CSV format: /path/to/image.jpg,CLUSTER_ID
            # Handle potential extra paths in the first column as shown in the example
            reader = csv.reader(csvfile)
            for row in reader:
                if row: # Ensure row is not empty
                    # The user example had paths concatenated before the comma
                    # Let's assume the *last* comma separates the path and the cluster ID
                    parts = ','.join(row).rsplit(',', 1)
                    if len(parts) == 2:
                        filepath = parts[0].strip()
                        cluster = parts[1].strip()
                        if cluster == cluster_id:
                            filenames.append(filepath)
                    # else: # Optional: Warn about malformed rows
                    #     print(f"Skipping malformed row: {row}")
        print(f"Loaded {len(filenames)} filenames for cluster {cluster_id} from {csv_path}")
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
    except Exception as e:
        print(f"Error reading CSV file {csv_path}: {e}")
    return filenames


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
    if args.cluster_csv and args.cluster_id:
        # Load from CSV based on cluster ID
        filenames = load_filenames_from_csv(args.cluster_csv, args.cluster_id)
        if not filenames:
            print(f"Warning: No filenames loaded for cluster {args.cluster_id} from {args.cluster_csv}. Exiting.")
            return # Or handle as appropriate
        # Decide if month_filter should still apply when using CSV
        # Option 1: CSV overrides month filter
        effective_month_filter = None
        # Option 2: Apply month filter *after* CSV loading (if desired)
        # effective_month_filter = args.train_month
        # if effective_month_filter:
        #     filenames = Dataset.filter_by_month(filenames, effective_month_filter) # Use static method from Dataset
    else:
        # Load training filenames (Original method)
        path = r"/ceph/project/P4-concept-drift/final_yolo_data_format/YOLOv8-pt/Dataset" # Make sure this path is correct
        train_txt_path = f'{path}/test.txt'
        print(f"Loading filenames from: {train_txt_path}")
        try:
            with open(train_txt_path) as reader:
                for filepath in reader.readlines():
                    filenames.append(filepath.strip())
            print(f"Loaded {len(filenames)} filenames from {train_txt_path}")
        except FileNotFoundError:
            print(f"Error: {train_txt_path} not found.")
            return # Or handle error
        # Apply month filter only if not using CSV
        effective_month_filter = args.train_month

    if not filenames:
         print("Error: No training filenames were loaded.")
         return

    # Create Dataset using the loaded filenames and potentially the month filter
    print(f"Initializing dataset with {len(filenames)} images.")
    dataset = Dataset(filenames, args.input_size, params, augment=False, month_filter=effective_month_filter) # Pass the correct filter
    # --- End of Modified Filename Loading ---

    # Sampler for distributed training if needed
    if args.world_size <= 1:
        sampler = None
    else:
        sampler = data.distributed.DistributedSampler(dataset)

    loader = data.DataLoader(dataset,
                             batch_size=args.batch_size,
                             shuffle=(sampler is None),
                             sampler=sampler,
                             num_workers=20,
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
    criterion = util.ComputeLoss(model, params) #
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
    path = r"/ceph/project/P4-concept-drift/final_yolo_data_format/YOLOv8-pt/Dataset" # Make sure path is correct
    val_txt_path = f'{path}/val.txt' # Assuming validation uses val.txt
    print(f"Loading validation filenames from: {val_txt_path}")
    try:
        with open(val_txt_path) as reader:
            for filepath in reader.readlines():
                filenames.append(filepath.strip())
        print(f"Loaded {len(filenames)} validation filenames from {val_txt_path}")
    except FileNotFoundError:
        print(f"Error: {val_txt_path} not found.")
        return 0., 0. # Return default values or handle error

    # Filter for testing month if specified (independent of training method)
    test_month_filter = args.test_month
    if test_month_filter:
         print(f"Applying test month filter: {test_month_filter}")
         filenames = Dataset.filter_by_month(filenames, test_month_filter) # Use static method
         print(f"Filtered to {len(filenames)} validation images for month: {test_month_filter}")

    if not filenames:
        print("Error: No validation filenames were loaded.")
        return 0., 0.

    print(f"Initializing validation dataset with {len(filenames)} images.")
    dataset = Dataset(filenames, args.input_size, params, augment=False, month_filter=None) # Month filter applied above

    loader = data.DataLoader(dataset, 8, shuffle=False,
                             num_workers=20, pin_memory=True,
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
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--train-month', type=str, default=None,
                        help='Filter training images by month (e.g., 01 for January)')
    parser.add_argument('--test-month', type=str, default=None,
                        help='Filter testing images by month (e.g., 01 for January)')
    parser.add_argument('--cluster-csv', type=str, default=None,
                        help='Path to the CSV file containing image paths and cluster IDs.')
    parser.add_argument('--cluster-id', type=str, default=None,
                        help='Cluster ID to filter images from the CSV (e.g., SC1). Requires --cluster-csv.')
    # --- End New Arguments ---
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
        print(f"Process {args.local_rank}: Initializing process group...")
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        print(f"Process {args.local_rank}: Process group initialized.")

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