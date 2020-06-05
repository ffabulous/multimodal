from __future__ import print_function

import datetime
import hashlib
import json
import os
import time
import sys

import shutil
import torch
import torch.utils.data
from torch.utils.data.dataloader import default_collate
from torch import nn
import torchvision
import torchvision.datasets.video_utils
from torchvision import transforms
from torchvision.datasets.samplers import DistributedSampler, UniformClipSampler, RandomClipSampler

from dataset import VideoDataset
from scheduler import WarmupMultiStepLR
import transforms as T
import utils

try:
    from apex import amp
except ImportError:
    amp = None


def train_one_epoch(model, criterion, optimizer, lr_scheduler, data_loader, device, epoch, log_freq, apex, logger):
    model.train()
    header = "Epoch: [{}]".format(epoch)
    for video, target in logger.log_every(data_loader, log_freq, header, model.training):
        start_time = time.time()
        video, target = video.to(device), target.to(device)
        output = model(video)
        loss = criterion(output, target)

        optimizer.zero_grad()
        if apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

        acc, _ = utils.accuracy(output, target, topk=(1,))
        batch_size = video.shape[0]

        logger.meters["lr"].update(optimizer.param_groups[0]["lr"])
        logger.meters["loss/train"].update(loss.item())
        logger.meters["acc/train"].update(acc[0].item(), n=batch_size)
        # lr_scheduler.step()


def evaluate(model, criterion, data_loader, device, logger):
    model.eval()
    header = "Test:"
    logger.meters["loss/valid"].reset()
    logger.meters["acc/valid"].reset()

    with torch.no_grad():
        for video, target in logger.log_every(data_loader, -1, header, model.training):
            video = video.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(video)
            loss = criterion(output, target)

            # FIXME: need to take into account that the datasets
            # could have been padded in distributed setup
            acc, cnt = utils.accuracy(output, target, topk=(1,))

            batch_size = video.shape[0]
            logger.meters["loss/valid"].update(loss.item())
            logger.meters["acc/valid"].update(acc[0].item(), n=batch_size)  # topk=1 only
            logger.update_counter(cnt)

    # gather the stats from all processes
    logger.synchronize_between_processes()
    logger.log_tx(model.training)
    print(" * Clip Acc {:.3f}".format(logger.meters["acc/valid"].global_avg))
    logger.label_accuracy()


def get_cache_path(file_path):
    h = hashlib.sha1(file_path.encode()).hexdigest()
    cache_path = os.path.join(".", "cache", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path


# NOTE: collate_fn이 list of rows (x, y)를 columns of (X, Y)로 바꿔 주는 듯
def collate_fn(batch):
    # remove audio from the batch
    batch = [(d[0], d[2]) for d in batch]
    return default_collate(batch)


def main(args):
    if args.output_dir:
        shutil.rmtree(args.output_dir, ignore_errors=True)
        os.makedirs(args.output_dir, exist_ok=True)
    if args.log_dir:
        shutil.rmtree(args.log_dir, ignore_errors=True)
        os.makedirs(args.log_dir, exist_ok=True)
    conf = vars(args)

    utils.init_distributed_mode(args)
    print(args)
    print("torch version: ", torch.__version__)
    print("torchvision version: ", torchvision.__version__)

    device = torch.device("cpu")
    if torch.cuda.is_available():
        # TODO: multi-gpu
        torch.cuda.set_device(args.gpu)
        device = torch.device("cuda:{}".format(args.gpu))
        # https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
        torch.backends.cudnn.benchmark = True

    traindir = os.path.join(args.data_path, args.train_dir)
    valdir = os.path.join(args.data_path, args.valid_dir)
    normalize = T.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989])

    transform_train = torchvision.transforms.Compose(
        [
            T.ToFloatTensorInZeroOne(),
            T.Resize((128, 171)),
            T.RandomHorizontalFlip(),
            normalize,
            T.RandomCrop((112, 112)),
        ]
    )

    transform_test = torchvision.transforms.Compose(
        [T.ToFloatTensorInZeroOne(), T.Resize((128, 171)), normalize, T.CenterCrop((112, 112))]
    )

    print("Loading training data")
    cache_path = get_cache_path(traindir)
    if args.cache_dataset and os.path.exists(cache_path):
        print("loading dataset_train from {}".format(cache_path))
        dataset_train, _ = torch.load(cache_path)
        dataset_train.transform = transform_train
    else:
        if args.distributed:
            print("it is recommended to pre-compute the dataset cache " "on a single-gpu first, as it will be faster")
        dataset_train = VideoDataset(
            traindir,
            frames_per_clip=args.clip_len,
            step_between_clips=args.clip_step,
            transform=transform_train,
            frame_rate=15,
            extensions=("mp4",),
        )
        if args.cache_dataset:
            print("saving dataset_train to {}".format(cache_path))
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            utils.save_on_master((dataset_train, traindir), cache_path)

    print("Loading validation data"
    cache_path = get_cache_path(valdir)
    if args.cache_dataset and os.path.exists(cache_path):
        print("Loading dataset_test from {}".format(cache_path))
        dataset_test, _ = torch.load(cache_path)
        dataset_test.transform = transform_test
    else:
        if args.distributed:
            print("It is recommended to pre-compute the dataset cache " "on a single-gpu first, as it will be faster")
        dataset_test = VideoDataset(
            valdir,
            frames_per_clip=args.clip_len,
            step_between_clips=args.clip_step,
            transform=transform_test,
            frame_rate=15,
            extensions=("mp4",),
        )
        if args.cache_dataset:
            print("Saving dataset_test to {}".format(cache_path))
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            utils.save_on_master((dataset_test, valdir), cache_path)

    train_sampler = RandomClipSampler(dataset_train.video_clips, args.clips_per_video)
    test_sampler = UniformClipSampler(dataset_test.video_clips, args.clips_per_video)
    if args.distributed:
        train_sampler = DistributedSampler(train_sampler)
        test_sampler = DistributedSampler(test_sampler)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,  # NOTE: https://discuss.pytorch.org/t/when-to-set-pin-memory-to-true/19723
        collate_fn=collate_fn,
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        sampler=test_sampler,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    print("Creating model")
    model = torchvision.models.video.r2plus1d_18(pretrained=args.pretrained, progress=True)
    if not args.fine_tune:
        for param in model.parameters():
            param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, len(dataset_test.classes))
    nn.init.xavier_uniform_(model.fc.weight)

    model.to(device)
    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    criterion = nn.CrossEntropyLoss()

    lr = args.lr * args.world_size
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # TODO: AdamW + Scheduler

    if args.apex:
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.apex_opt_level)

    # convert scheduler to be per iteration, not per epoch, for warmup that lasts
    # between different epochs
    warmup_iters = args.lr_warmup_epochs * len(data_loader_train)
    lr_milestones = [len(data_loader_train) * m for m in args.lr_milestones]
    # lr_scheduler = WarmupMultiStepLR(
    #     optimizer, milestones=lr_milestones, gamma=args.lr_gamma, warmup_iters=warmup_iters, warmup_factor=1e-5
    # )

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        # lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1

    logger = utils.MetricLogger(args.log_dir, delimiter="  ")
    # logger.add_meter("clips/s", utils.SmoothedValue(window_size=10, fmt="{value:.3f}"))
    logger.add_meter("lr", utils.SmoothedValue(1, "{value}"))
    logger.add_meter("loss/train", utils.SmoothedValue(args.log_freq, "{value:.4f} ({avg:.4f})"))
    logger.add_meter("loss/valid", utils.SmoothedValue(args.log_freq, "{value:.4f} ({avg:.4f})"))

    if args.test_only:
        evaluate(model, criterion, data_loader_test, device, logger)
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(
            model, criterion, optimizer, None, data_loader_train, device, epoch, args.log_freq, args.apex, logger
        )
        evaluate(model, criterion, data_loader_test, device, logger)
        if args.output_dir:
            checkpoint = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                # "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "args": args,
            }
            epoch_acc = logger.meters["acc/valid"].global_avg
            if epoch_acc > conf.get("best_acc", 0):
                conf["best_acc"] = epoch_acc
                conf["best_epoch"] = epoch
            json.dump(conf, open(os.path.join(args.output_dir, "conf.json"), "w"), indent=2)
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, "model_{}.pth".format(epoch)))
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, "checkpoint.pth"))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Finetune R(2+1)D")

    # data, path
    parser.add_argument("--data-path", default="data", help="dataset")
    parser.add_argument("--train-dir", default="train", help="name of train dir")
    parser.add_argument("--valid-dir", default="valid", help="name of valid dir")
    # parser.add_argument("--test-dir", default="test", help="name of test dir")
    parser.add_argument(
        "--cache-dataset",
        help="Cache the datasets for quicker initialization. It also serializes the transforms",
        action="store_true",
    )
    parser.add_argument("--output-dir", help="path where to save checkpoints")
    parser.add_argument("--log-dir", help="path where to log")
    parser.add_argument("--log-freq", default=10, type=int, help="log every N steps")

    # model
    parser.add_argument("--clip-len", default=16, type=int, help="number of frames per clip")
    parser.add_argument("--clip-step", default=16, type=int, help="number of frames between clips")
    parser.add_argument("--clips-per-video", default=10, type=int, help="maximum number of clips per video to consider")
    parser.add_argument("-b", "--batch-size", default=32, type=int)
    parser.add_argument("--epochs", default=5, type=int, help="number of total epochs to run")
    parser.add_argument("-j", "--workers", default=16, type=int, help="number of data loading workers (default: 16)")
    parser.add_argument("--lr", default=0.01, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
    parser.add_argument("--weight-decay", default=1e-4, type=float, help="weight decay (default: 1e-4)")
    parser.add_argument("--lr-milestones", nargs="+", default=[20, 30, 40], type=int, help="decrease lr on milestones")
    parser.add_argument("--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma")
    parser.add_argument("--lr-warmup-epochs", default=10, type=int, help="number of warmup epochs")
    parser.add_argument("--sync-bn", dest="sync_bn", help="Use sync batch norm", action="store_true")
    parser.add_argument("--pretrained", help="Use pre-trained models from the modelzoo", action="store_true")

    # tran/eval
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument("--start-epoch", default=0, type=int, help="start epoch")  # TODO: check how it works
    parser.add_argument("--test-only", dest="test_only", help="Only test the model", action="store_true")
    parser.add_argument("--fine-tune", help="fine-tune model", action="store_true")

    # perf
    parser.add_argument("--gpu", default=0, type=int, help="gpu number")
    parser.add_argument("--apex", action="store_true", help="Use apex for mixed precision training")
    parser.add_argument(
        "--apex-opt-level", default="O1", type=str, help="https://github.com/NVIDIA/apex/tree/master/examples/imagenet"
    )
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", help="url used to set up distributed training")

    args = parser.parse_args()
    main(args)
