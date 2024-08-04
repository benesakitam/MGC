import os
import sys
import argparse
import time
import random
import warnings
import subprocess
import importlib
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed

from torch.nn.parallel import DistributedDataParallel as DDP
from utils.log import setup_logger
from utils import adjust_learning_rate_iter, save_checkpoint, parse_devices, AvgMeter
from utils.torch_dist import configure_nccl, synchronize
from tensorboardX import SummaryWriter
from torch.cuda.amp import autocast as autocast, GradScaler


def cleanup():
    dist.destroy_process_group()


def main():
    file_name = os.path.join(exp.output_dir, args.experiment_name)
    if args.local_rank == 0:
        if not os.path.exists(file_name):
            os.makedirs(file_name, exist_ok=True)
        writer = SummaryWriter(os.path.join(file_name, 'runs'))

    logger = setup_logger(file_name, distributed_rank=args.local_rank, filename="train_log.txt", mode="a")
    logger.info("gpuid: {}, args: {}".format(args.local_rank, args))

    train_loader = exp.get_data_loader(batch_size=args.batchsize, is_distributed=args.nr_gpu > 1, if_transformer=False)["train"]
    model = exp.get_model().to(device)
    if args.distributed:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=False)

    optimizer = exp.get_optimizer_new(model.module, args.batchsize)

    world_size = torch.distributed.get_world_size() if args.distributed else 1

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            # Map model to be loaded to specified single gpu.
            loc = device
            checkpoint = torch.load(args.resume, map_location='cpu')
            args.start_epoch = checkpoint['start_epoch']
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['start_epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # -----------------------------start training-----------------------------#
    model.train()
    ITERS_PER_EPOCH = len(train_loader)
    if args.local_rank == 0:
        logger.info("Training start...")
        logger.info("Here is the logging file"+str(file_name))
        # logger.info(str(model))

    args.lr = 10 * exp.base_lr
    args.warmup_epochs = exp.warmup_epochs
    args.total_epochs = exp.max_epoch
    iter_count = ITERS_PER_EPOCH * args.start_epoch

    scaler = GradScaler()
    for epoch in range(args.start_epoch, args.total_epochs):
        if args.nr_gpu > 1:
            train_loader.sampler.set_epoch(epoch)
        batch_time_meter = AvgMeter()

        for i, (inps, target) in enumerate(train_loader):
            iter_count += 1
            iter_start_time = time.time()

            for indx in range(len(inps)):
                inps[indx] = inps[indx].to(device, non_blocking=True)

            data_time = time.time() - iter_start_time
            with autocast():
                loss, con_c2c, con_m2m, con_c2m, con_m2c, kl_t2s, kl_m2c = model(inps, update_param=True)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            lr = adjust_learning_rate_iter(optimizer, iter_count, args, ITERS_PER_EPOCH)
            batch_time_meter.update(time.time() - iter_start_time)

            log_interval = exp.print_interval
            if args.local_rank == 0 and (i + 1) % log_interval == 0:
                remain_time = (ITERS_PER_EPOCH * exp.max_epoch - iter_count) * batch_time_meter.avg
                t_m, t_s = divmod(remain_time, 60)
                t_h, t_m = divmod(t_m, 60)
                t_d, t_h = divmod(t_h, 24)
                remain_time = "{}d.{:02d}h.{:02d}m".format(int(t_d), int(t_h), int(t_m))

                logger.info(
                    "[{}/{}], remain:{}, It:[{}/{}], Data-Time:{:.3f}, LR:{:.4f}, Loss:{:.2f}, CON_C2C:{:.2f}, "
                    "CON_M2M:{:.2f}, CON_C2M:{:.2F}, CON_M2C:{:.2F}, KL_T2S:{:.4f}, KL_M2C:{:.4f}".format(
                        epoch + 1, args.total_epochs, remain_time, i + 1, ITERS_PER_EPOCH, data_time, lr,
                        loss, con_c2c, con_m2m, con_c2m, con_m2c, kl_t2s, kl_m2c)
                )

            log_tensorboard_interval = 50
            if args.local_rank == 0 and (i + 1) % log_tensorboard_interval == 0:
                writer.add_scalar('con_c2c', con_c2c, iter_count)
                writer.add_scalar('con_m2m', con_m2m, iter_count)
                writer.add_scalar('con_c2m', con_c2m, iter_count)
                writer.add_scalar('con_m2c', con_m2c, iter_count)
                writer.add_scalar('kl_t2s', kl_t2s, iter_count)
                writer.add_scalar('kl_m2c', kl_m2c, iter_count)
                writer.add_scalar('loss', loss, iter_count)
                writer.add_scalar('lr', lr, iter_count)

        if args.local_rank == 0:
            logger.info(
                "Train-Epoch: [{}/{}], LR: {:.4f}, Unsup-Loss: {:.2f}".format(epoch + 1, args.total_epochs, lr, loss)
            )

            if epoch in [99, 199, 299, 399, 499, 599, 699, 799]:
                save_checkpoint(
                    {"start_epoch": epoch + 1, "model": model.state_dict(), "optimizer": optimizer.state_dict()},
                    False,
                    file_name,
                    str(epoch+1),
                )

    if args.local_rank == 0:
        print("Pre-training of experiment: {} is done.".format(args.experiment_name))
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser("MGC")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)

    # optimization
    parser.add_argument(
        "--scheduler",
        type=str,
        default="warmcos",
        choices=["warmcos", "cos", "linear", "multistep", "step"],
        help="type of scheduler",
    )

    # distributed
    parser.add_argument("--distributed", action='store_true')
    parser.add_argument("-n", "--n_views", type=int, default=2)
    parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
    parser.add_argument("--dist-url", default=None, type=str, help="url used to set up distributed training")
    parser.add_argument('--world-size', default=-1, type=int, help='number of nodes for distributed training')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
    parser.add_argument("-b", "--batchsize", type=int, default=256, help="batch size")
    parser.add_argument("-d", "--devices", default="0-7", type=str, help="device for training")
    parser.add_argument("--log_path", default="/workspace/MGC/results/train_new_log/", type=str, help="the path of the logging file")
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    args.nr_gpu = torch.cuda.device_count()
    if args.local_rank == 0:
        print("V1 Using", torch.cuda.device_count(), "GPUs per node!")

    from exps import mgc_exp
    exp = mgc_exp.Exp(args)

    ngpus_per_node = torch.cuda.device_count()
    if args.distributed:
        args.world_size = ngpus_per_node * args.world_size
        torch.distributed.init_process_group(backend="nccl", init_method='tcp://localhost:10001',
                                             world_size=args.world_size, rank=args.local_rank)
        if args.local_rank == 0:
            if os.path.exists("./" + args.experiment_name + "ip_add.txt"):
                os.remove("./" + args.experiment_name + "ip_add.txt")

    print("Rank {} initialization finished.".format(args.local_rank))
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)

    main()

    if args.distributed:
        cleanup()
