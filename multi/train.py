import json
import logging
import os
import shutil

import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision.datasets.samplers import RandomClipSampler, UniformClipSampler

from dataset import *
from models import DTT, MMVC
import utils


def train(model, criterion, optimizer, train_loader, eval_every, device, metric):
    model.train()
    for batch in train_loader:
        xv, xt, target = [t.to(device) for t in batch]
        output = model(xv, xt)
        logit, pred = torch.max(output, dim=1)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = utils.accuracy(pred, target)
        metric.step += 1
        metric.add_scalars("loss", {"train": loss.item()}, metric.step)
        metric.add_scalars("acc", {"train": acc.item()}, metric.step)
        logging.info("Step: %d, Loss: %.3f, Acc: %.3f", metric.step, loss.item(), acc.item())
        if metric.step % eval_every == 0:
            yield metric.step


def evaluate(model, criterion, test_loader, device, metric):
    total_loss = 0
    metric.reset_count()
    logging.info("Evaluation")
    logging.info("Batches: %d", len(test_loader))
    with torch.no_grad():
        model.eval()
        for batch in test_loader:
            xv, xt, target = [t.to(device) for t in batch]
            output = model(xv, xt)
            logit, pred = torch.max(output, dim=1)
            acc, (correct, total) = utils.accuracy(pred, target, with_count=True)
            metric.update_count(correct, total)
            loss = criterion(output, target)
            total_loss += loss.item()
    avg_acc, class_acc = metric.accuracy(class_wise=True)
    avg_loss = total_loss / len(test_loader)
    metric.add_scalars("loss", {"valid": avg_loss}, metric.step)
    metric.add_scalars("acc", {"valid": avg_acc * 1e2}, metric.step)
    logging.info("* Loss: %.3f, Acc: %.3f", avg_loss, avg_acc)
    for label, acc, correct, total in class_acc:
        logging.info("%s: %.1f (%d / %d)", label, acc * 1e2, correct, total)
    return avg_acc


def main(args):
    cached_path = json.load(open(args.cache_conf))
    video_path = cached_path["video"]
    text_path = cached_path["text"]
    map_path = cached_path["map_file"]

    logging.info("Loading Data")
    video_train = os.path.join(video_path["data_path"], video_path["train_file"])
    text_train = os.path.join(text_path["data_path"], text_path["train_file"])
    train_data = MultiModalDataset(video_train, text_train, map_path)

    video_valid = os.path.join(video_path["data_path"], video_path["valid_file"])
    text_valid = os.path.join(text_path["data_path"], text_path["valid_file"])
    valid_data = MultiModalDataset(video_valid, text_valid, map_path)

    video_test = os.path.join(video_path["data_path"], video_path["test_file"])
    text_test = os.path.join(text_path["data_path"], text_path["test_file"])
    test_data = MultiModalDataset(video_test, text_test, map_path)

    train_sampler = RandomClipSampler(train_data.video_data.video_clips, args.clips_per_video)
    valid_sampler = UniformClipSampler(valid_data.video_data.video_clips, 1)
    test_sampler = UniformClipSampler(test_data.video_data.video_clips, 1)  # NOTE: 10

    train_loader = DataLoader(
        train_data,
        args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_data,
        args.batch_size,
        sampler=valid_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_data, args.batch_size, sampler=test_sampler, num_workers=args.num_workers, pin_memory=True, drop_last=True
    )

    if args.output_dir:
        shutil.rmtree(args.output_dir, ignore_errors=True)
        os.makedirs(args.output_dir, exist_ok=True)
    if args.log_dir:
        shutil.rmtree(args.log_dir, ignore_errors=True)
        os.makedirs(args.log_dir, exist_ok=True)

    logging.info("Loading Model")
    num_classes = len(train_data.video_data.classes)

    # R(2+1)D
    video_model = torchvision.models.video.r2plus1d_18(pretrained=True, progress=True)
    video_model.fc = nn.Linear(video_model.fc.in_features, num_classes)
    video_ckpt = torch.load(os.path.join(video_path["model_path"], video_path["model_file"]), map_location="cpu")
    video_model.load_state_dict(video_ckpt["model"])
    if not args.fine_tune:
        for param in video_model.parameters():
            param.requires_grad = False
    video_out_dim = video_model.fc.in_features
    video_model.fc = nn.Identity()

    # DTT
    text_ckpt = torch.load(os.path.join(text_path["model_path"], text_path["model_file"]), map_location="cpu")
    seq_len = text_ckpt["args"].max_seq_len
    filter_sizes = text_ckpt["args"].filter_sizes
    num_filters = text_ckpt["args"].num_filters
    embedding = utils.load_word_embedding(text_path["emb_file"], add_oov=True)
    text_model = DTT(seq_len, num_classes, filter_sizes, num_filters, embedding)
    text_model.load_state_dict(text_ckpt["model"])
    if not args.fine_tune:
        for param in text_model.parameters():
            param.requires_grad = False
    text_out_dim = text_model.fc[1].in_features
    text_model.fc = nn.Identity()

    # MMVC
    model = MMVC(video_model, text_model, video_out_dim + text_out_dim, num_classes, args.drop_prob)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"])

    # optim
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    metric = utils.MetricLogger(args.log_dir)

    # device
    device = torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(args.local_rank)
        torch.backends.cudnn.benchmark = True
        device = torch.device("cuda", args.local_rank)
    model.to(device)

    if args.do_train:
        logging.info("Start Training")
        logging.info("Batches: %d", len(train_loader))
        best_acc = 0
        for epoch in range(args.epochs):
            logging.info("Epoch: %d", epoch)
            for step in train(model, criterion, optimizer, train_loader, args.eval_every, device, metric):
                step_acc = evaluate(model, criterion, valid_loader, device, metric)
                if step_acc > best_acc:
                    best_acc = step_acc
                    checkpoint = {"model": model.state_dict(), "args": args}
                    torch.save(checkpoint, os.path.join(args.output_dir, "{}.pth".format(step)))

    if args.do_test:
        logging.info("Start Evaluating")
        evaluate(model, criterion, test_loader, device, metric)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="multi-modal video classifier (MMV)")

    # resource
    parser.add_argument("--cache-conf", default="cached.json", help="cache config")
    parser.add_argument("--output-dir", help="path where to save checkpoints")
    parser.add_argument("--log-dir", default="logs/test", help="path where to log")
    parser.add_argument("--eval-every", default=100, type=int, help="evaluate every N steps")

    # model
    parser.add_argument("--clips-per-video", default=3, type=int, help="max clips to sample from a video")
    parser.add_argument("--drop-prob", type=float, default=0.2)
    parser.add_argument("--lr", default=1e-4, type=float, help="initial learning rate")

    # train
    parser.add_argument("--epochs", default=10, type=int, help="number of total epochs to run")
    parser.add_argument("-b", "--batch-size", default=32, type=int)
    parser.add_argument("-j", "--num-workers", default=10, type=int, help="number of data loading workers")
    parser.add_argument("--resume", help="resume from checkpoint")
    parser.add_argument("--fine-tune", help="fine tune", action="store_true")
    parser.add_argument("--do-train", help="train with train data", action="store_true")
    parser.add_argument("--do-test", help="evaluate on test data", action="store_true")
    parser.add_argument("--local-rank", default=0, type=int, help="gpu id")

    args = parser.parse_args()
    if not args.do_train and not args.do_test:
        parser.error("--do-{train|test} option must be specified")

    logging.basicConfig(format="%(asctime)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
    main(args)


# python train.py --eval-every 5 --epochs 1 --fine-tune --output-dir dump/test --do-train --do-test
# python train.py --resume dump/test/5.pth --do-test --local-rank 5
