import os, time, random, math
import numpy as np
from tqdm import tqdm

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from model import Model

from utils.ddp_utils import setup_ddp, cleanup_ddp
from utils.log_utils import log
from utils.dataset import CalibrationDataset

def train(rank, world_size, args):
    if args.enable_ddp:
        setup_ddp(rank, world_size)
    else:
        rank = 0
        world_size = 1

    # Device setup
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # Path setup
    if rank == 0:
        os.makedirs(args.checkpoint_path, exist_ok=True)

        checkpoint_dirpath = os.path.join(args.checkpoint_path, 'checkpoints' + '-{}')
        log_path = os.path.join(args.checkpoint_path, 'results.txt')
        event_path = os.path.join(args.checkpoint_path, 'tensorboard')

    '''
    Set up the model
    '''
    model = Model(
        num_extenders=args.num_extenders,
        model_name=args.model_name,
        output_reference_frame=args.output_reference_frame)

    model.to(device)
    if args.enable_ddp:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[rank])
    m = model.module if hasattr(model, 'module') else model

    for n, p in m.named_parameters():
        if not n.startswith('dex.'):
            p.requires_grad = False

    '''
    Setup training dataloader
    '''
    N = len(args.datasets)
    assert len(args.crop_shapes) == 2 * N, \
        f"--crop_shapes must have {2*N} values (H W per dataset) for {N} datasets, got {len(args.crop_shapes)}"
    for kname in ('k1_ranges', 'k2_ranges', 'k3_ranges', 'k4_ranges'):
        vals = getattr(args, kname)
        assert len(vals) == 2 * N, \
            f"--{kname} must have {2*N} values (min max per dataset) for {N} datasets, got {len(vals)}"
    assert len(args.image_txts) == N, \
        f"--image_txts must have {N} paths (one per dataset), got {len(args.image_txts)}"
    assert len(args.intrinsics_paths) == N, \
        f"--intrinsics_paths must have {N} paths (one per dataset), got {len(args.intrinsics_paths)}"
    assert len(args.learning_rates) == len(args.learning_schedule), \
        f"--learning_rates ({len(args.learning_rates)}) must match --learning_schedule ({len(args.learning_schedule)})"

    dataset_weights = args.dataset_weights if args.dataset_weights is not None else [1.0] * N
    assert len(dataset_weights) == N, \
        f"--dataset_weights must have {N} values (one per dataset), got {len(dataset_weights)}"

    # Augmentation parameters
    args.augmentation_probabilities = [
        args.aug_prob_brightness,
        args.aug_prob_contrast,
        args.aug_prob_saturation,
        args.aug_prob_hue,
        args.aug_prob_blur,
    ]
    args.augmentation_ranges = [
        args.aug_brightness_range,
        args.aug_contrast_range,
        args.aug_saturation_range,
        args.aug_hue_range,
        args.aug_blur_range,
    ]

    train_datasets = []
    dataset_names = args.datasets
    for i, name in enumerate(dataset_names):
        crop_shape = (args.crop_shapes[i * 2], args.crop_shapes[i * 2 + 1])
        distortion_ranges = [
            args.k1_ranges[i * 2], args.k1_ranges[i * 2 + 1],
            args.k2_ranges[i * 2], args.k2_ranges[i * 2 + 1],
            args.k3_ranges[i * 2], args.k3_ranges[i * 2 + 1],
            args.k4_ranges[i * 2], args.k4_ranges[i * 2 + 1],
        ]
        dataset = CalibrationDataset(
            args, patch_size=m.patch_size,
            crop_shape=crop_shape, distortion_ranges=distortion_ranges,
            image_txt=args.image_txts[i],
            intrinsics_path=args.intrinsics_paths[i])
        train_datasets.append(dataset)

    # Upsample all datasets to match the largest, then shuffle each
    max_size = max(len(d) for d in train_datasets)
    for dataset in train_datasets:
        if len(dataset) < max_size:
            # Repeat enough times until over max_size, then trim
            times = math.ceil(max_size / len(dataset))
            dataset.image_paths = (dataset.image_paths * times)[:max_size]
            if dataset._intrinsic_paths is not None:
                dataset._intrinsic_paths = (dataset._intrinsic_paths * times)[:max_size]
        # Shuffle together
        indices = list(range(len(dataset.image_paths)))
        random.shuffle(indices)
        dataset.image_paths = [dataset.image_paths[i] for i in indices]
        if dataset._intrinsic_paths is not None:
            dataset._intrinsic_paths = [dataset._intrinsic_paths[i] for i in indices]

    n_batches_per_update = args.n_batches_per_update
    assert n_batches_per_update % N == 0, f"--n_batches_per_update ({n_batches_per_update}) must be a multiple of the number of datasets ({N})"

    if rank == 0:
        print(f"Dataset sizes after upsampling: {[len(d) for d in train_datasets]}")

    train_dataloaders = []
    for i, dataset in enumerate(train_datasets):
        data_loader_opts = {
            'batch_size': args.train_batch_size,
            'shuffle': True,
            'num_workers': args.n_thread,
            'drop_last': True,
            'pin_memory': True
        }

        if args.enable_ddp:
            sampler = torch.utils.data.DistributedSampler(dataset)
            data_loader_opts['shuffle'] = False
            data_loader_opts['sampler'] = sampler

        dataloader = torch.utils.data.DataLoader(dataset, **data_loader_opts)
        train_dataloaders.append(dataloader)

    # Calculate total data batch steps per epoch (sum of all dataset steps)
    n_train_step = sum(len(loader) for loader in train_dataloaders)
    if rank == 0:
        print(f"Total steps per epoch: {n_train_step}")
        for i, loader in enumerate(train_dataloaders):
            print(f"Dataset {dataset_names[i]}: {len(loader)} steps")

    '''
    Log input paths
    '''
    if rank == 0:
        train_settings = {
            'batch_size': args.train_batch_size,
            'num_extenders': args.num_extenders,
            'output_reference_frame': args.output_reference_frame,
            'datasets': args.datasets,
            'image_txts': args.image_txts,
            'intrinsics_paths': args.intrinsics_paths,
            'crop_shapes': args.crop_shapes,
            'k1_ranges': args.k1_ranges,
            'k2_ranges': args.k2_ranges,
            'k3_ranges': args.k3_ranges,
            'k4_ranges': args.k4_ranges,
            'dataset_weights': dataset_weights,
            'n_batches_per_update': args.n_batches_per_update,
            'learning_rates': args.learning_rates,
            'learning_schedule': args.learning_schedule,
            'augmentation_probabilities': args.augmentation_probabilities,
            'checkpoint_path': args.checkpoint_path,
            'n_step_per_checkpoint': args.n_step_per_checkpoint,
            'n_step_per_summary': args.n_step_per_summary,
            'restore_path': args.restore_path,
            'device': device,
            'n_thread': args.n_thread
        }
        log('Training settings:', log_path)
        log(train_settings, log_path)

        train_summary_writer = SummaryWriter(event_path)

        m.train_summary_writer = train_summary_writer

    '''
    Train model
    '''
    learning_schedule_pos = 0
    learning_rate = args.learning_rates[0]

    dex_params = [p for name, p in m.named_parameters() if name.startswith('dex.')]

    # Set up optimizer
    optimizer = torch.optim.Adam([
        {
            'params': dex_params,
            'weight_decay': 0.0
        }],
        lr=learning_rate)

    if args.restore_path is not None:
        print(f'Loading pretrained model from {args.restore_path}')

        _, optimizer = m.restore_model(
            restore_path=args.restore_path,
            optimizer=optimizer,
            enable_ddp=args.enable_ddp,
            rank=rank)

        for g in optimizer.param_groups:
            g['lr'] = learning_rate

    time_start = time.time()

    if rank == 0:
        log('Begin training...', log_path)

    if args.enable_ddp:
        dist.barrier()

    grad_step = 0

    for epoch in range(1, args.learning_schedule[-1] + 1):
        if args.enable_ddp:
            # Set epoch for all samplers
            for i, loader in enumerate(train_dataloaders):
                if hasattr(loader, 'sampler') and hasattr(loader.sampler, 'set_epoch'):
                    loader.sampler.set_epoch(epoch)
        # Set learning rate schedule
        if epoch > args.learning_schedule[learning_schedule_pos]:
            learning_schedule_pos = learning_schedule_pos + 1
            learning_rate = args.learning_rates[learning_schedule_pos]

            print(f'Updating learning rate to {learning_rate}')

            # Update optimizer learning rates for network
            for g in optimizer.param_groups:
                g['lr'] = learning_rate

        with tqdm(total=n_train_step, desc=f"Epoch {epoch}/{args.learning_schedule[-1]}", unit="batch") as pbar:

            # Ensure different shuffling for each epoch (non-DDP mode)
            if not args.enable_ddp:
                torch.manual_seed(42 + epoch)
                np.random.seed(42 + epoch)

            # Create iterators for each dataloader
            dataloader_iterators = [iter(loader) for loader in train_dataloaders]
            n_datasets = len(train_dataloaders)

            # Initialize gradients for the epoch
            optimizer.zero_grad()

            # For logging all datasets
            dataset_losses = {}

            # Log on the first step and then every n_step_per_summary steps
            log_images_this_window = (rank == 0)

            for step_count_in_epoch in range(1, n_train_step + 1):
                dataset_idx = (step_count_in_epoch - 1) % n_datasets
                train_batch = next(dataloader_iterators[dataset_idx])

                if rank == 0:
                    pbar.update(1)

                train_image, train_fisheye, grid_tensor, for_grid_tensor, intrinsics = train_batch
                train_image = train_image.to(device)
                train_fisheye = train_fisheye.to(device)
                grid_tensor = grid_tensor.to(device)
                for_grid_tensor = for_grid_tensor.to(device)
                intrinsics = intrinsics.to(device)

                m.current_dataset = dataset_names[dataset_idx]

                # The image logging happens inside the model's loss function, so we set log_step here
                # All batches in a logging window get the same step tag so every dataset is logged
                m.log_step = (grad_step + 1) if log_images_this_window else None

                # Forward pass and loss
                standard_depth_pred, fisheye_depth_pred = model(
                    train_image, train_fisheye, intrinsics, grid_tensor, for_grid_tensor, rank)

                loss = m.direct_loss(
                    standard_depth=standard_depth_pred,
                    fisheye_depth=fisheye_depth_pred,
                    grid_tensor=grid_tensor,
                    for_grid_tensor=for_grid_tensor,
                    rank=rank)

                weighted_loss = loss * dataset_weights[dataset_idx]
                weighted_loss.backward()
                dataset_losses[dataset_names[dataset_idx]] = weighted_loss.item()

                # Gradient accumulation: optimizer step every n_batches_per_update batches
                if step_count_in_epoch % n_batches_per_update == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    grad_step += 1

                    should_log = rank == 0 and (grad_step == 1 or grad_step % args.n_step_per_summary == 0)

                    if should_log:
                        log_losses = {**dataset_losses, 'total': sum(dataset_losses.values())}
                        train_summary_writer.add_scalars('train_losses', log_losses, grad_step)
                        time_elapse = (time.time() - time_start) / 3600
                        time_remain = (n_train_step - grad_step) * time_elapse / max(grad_step, 1)
                        log('Step={:6}  Losses={}  Time Elapsed={:.2f}h  Time Remaining={:.2f}h'.format(
                            grad_step,
                            {k: f'{v:.5f}' for k, v in dataset_losses.items()},
                            time_elapse, time_remain),
                            log_path)

                    # Decide whether to log images for the next accumulation window
                    next_grad_step = grad_step + 1
                    log_images_this_window = rank == 0 and (
                        next_grad_step == 1 or next_grad_step % args.n_step_per_summary == 0)

                    # Save checkpoint
                    if grad_step % args.n_step_per_checkpoint == 0:
                        if rank == 0:
                            torch.cuda.empty_cache()
                            m.save_model(
                                checkpoint_dirpath=checkpoint_dirpath.format(grad_step),
                                step=grad_step,
                                optimizer=optimizer)

                        if args.enable_ddp:
                            dist.barrier()

            # Update any remaining accumulated gradients at epoch end
            if step_count_in_epoch % n_batches_per_update != 0:
                optimizer.step()
                optimizer.zero_grad()

    if rank == 0:
        m.save_model(
            checkpoint_dirpath=checkpoint_dirpath.format(grad_step),
            step=grad_step,
            optimizer=optimizer)

    if args.enable_ddp:
        cleanup_ddp()
