import json
import logging
import os
import shutil

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import TextDataset
from models import DTT
import utils


def train(model, criterion, optimizer, train_loader, epoch, log_every, device, metric):
    model.train()
    logging.info("Epoch: {}".format(epoch))
    for video, target in train_loader:
        video, target = video.to(device), target.to(device)
        output = model(video)
        logit, pred = torch.max(output, dim=1)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = utils.accuracy(pred, target)
        metric.step += 1
        metric.add_scalars("loss", {"train": loss.item()}, metric.step)
        metric.add_scalars("acc", {"train": acc.item()}, metric.step)
        if metric.step % log_every == 0:
            logging.info("Step: {}, Loss: {:.3f}, Acc: {:.3f}".format(metric.step, loss.item(), acc.item()))


def evaluate(model, criterion, valid_loader, device, metric):
    model.eval()
    total_loss = 0
    metric.reset_count()
    logging.info("Evaluation")
    for video, target in valid_loader:
        video, target = video.to(device), target.to(device)
        output = model(video)
        logit, pred = torch.max(output, dim=1)
        acc, (correct, total) = utils.accuracy(pred, target, with_count=True)
        metric.update_count(correct, total)
        loss = criterion(output, target)
        total_loss += loss.item()
    avg_acc, class_acc = metric.accuracy(class_wise=True)
    metric.add_scalars("loss", {"valid": total_loss}, metric.step)
    metric.add_scalars("acc", {"valid": avg_acc * 1e2}, metric.step)
    logging.info("* Loss: {:.3f}, Acc: {:.3f}".format(total_loss, avg_acc))
    for label, acc, correct, total in class_acc:
        logging.info("{}: {:.1f} ({} / {})".format(label, acc * 1e2, correct, total))
    return avg_acc


def main(args):
    conf = vars(args)
    device = torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        torch.backends.cudnn.benchmark = True
        device = torch.device("cuda", args.gpu)

    if args.output_dir:
        shutil.rmtree(args.output_dir, ignore_errors=True)
        os.makedirs(args.output_dir, exist_ok=True)
    if args.log_dir:
        shutil.rmtree(args.log_dir, ignore_errors=True)
        os.makedirs(args.log_dir, exist_ok=True)

    train_file = os.path.join(args.data_dir, args.train_file)
    valid_file = os.path.join(args.data_dir, args.valid_file)
    test_file = os.path.join(args.data_dir, args.test_file)
    label_file = os.path.join(args.data_dir, args.label_file)

    logging.info("Loading Data")
    if args.do_train:
        cache_path = utils.get_cache_path(train_file, args.cache_dir)
        if args.cache_dataset and os.path.exists(cache_path):
            train_data = torch.load(cache_path)
        else:
            train_data = TextDataset(train_file, label_file, args.max_seq_len, args.min_seq_len, args.doc_stride)
            if args.cache_dataset:
                os.makedirs(args.cache_dir, exist_ok=True)
                torch.save(train_data, cache_path)
        train_loader = DataLoader(
            train_data,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=True,
        )
        cache_path = utils.get_cache_path(valid_file, args.cache_dir)
        if args.cache_dataset and os.path.exists(cache_path):
            valid_data = torch.load(cache_path)
        else:
            valid_data = TextDataset(valid_file, label_file, args.max_seq_len, args.min_seq_len, args.doc_stride)
            if args.cache_dataset:
                os.makedirs(args.cache_dir, exist_ok=True)
                torch.save(valid_data, cache_path)
        num_classes = len(valid_data.classes)
        valid_loader = DataLoader(valid_data, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)
    if args.do_test:
        cache_path = utils.get_cache_path(test_file, args.cache_dir)
        if args.cache_dataset and os.path.exists(cache_path):
            test_data = torch.load(cache_path)
        else:
            test_data = TextDataset(test_file, label_file, args.max_seq_len, args.min_seq_len)
            if args.cache_dataset:
                os.makedirs(args.cache_dir, exist_ok=True)
                torch.save(test_data, cache_path)
        num_classes = len(test_data.classes)
        test_loader = DataLoader(test_data, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)

    logging.info("Creating Model")
    W = utils.load_word_embedding(args.embedding, add_oov=True)
    model = DTT(
        seq_len=args.max_seq_len,
        num_label=num_classes,
        k_sizes=args.filter_sizes,
        num_k=args.num_filters,
        embedding=W,
        drop_prob=args.drop_prob,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    criterion = nn.CrossEntropyLoss()
    metric = utils.MetricLogger(args.log_dir)
    model.to(device)

    if args.do_train:
        logging.info("Start Training")
        for epoch in range(args.epochs):
            train(model, criterion, optimizer, train_loader, epoch, args.log_freq, device, metric)
            epoch_acc = evaluate(model, criterion, valid_loader, device, metric)
            if args.output_dir:
                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "args": args,
                }
                if epoch_acc > conf.get("best_acc", 0):
                    conf["best_acc"] = epoch_acc
                    conf["best_epoch"] = epoch
                json.dump(conf, open(os.path.join(args.output_dir, "conf.json"), "w"), indent=2)
                torch.save(checkpoint, os.path.join(args.output_dir, "model_{}.pth".format(epoch)))
        logging.info("Finished Training")
        metric.close()

    if args.do_test:
        evaluate(model, criterion, test_loader, device, metric)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="deep topic tagger (DTT)")

    # resource
    parser.add_argument("--data-dir", default="data", help="data directory")
    parser.add_argument("--cache-dir", default="cache", help="cache directory")
    parser.add_argument("--train-file", default="train.tsv", help="name of train file")
    parser.add_argument("--valid-file", default="valid.tsv", help="name of valid file")
    parser.add_argument("--test-file", default="test.tsv", help="name of test file")
    parser.add_argument("--label-file", default="label.txt", help="name of label file")
    parser.add_argument("--embedding", default="data/word2vec/sampled.npy", help="word embedding file")
    parser.add_argument("--cache-dataset", dest="cache_dataset", action="store_true")
    parser.add_argument("--output-dir", help="path where to save checkpoints")
    parser.add_argument("--log-dir", default="logs/test", help="path where to log")
    parser.add_argument("--log-freq", default=10, type=int, help="log every N steps")

    # model
    parser.add_argument("--min-seq-len", default=5, type=int, help="min sequence length (in words)")
    parser.add_argument("--max-seq-len", default=60, type=int, help="max sequence length (in words)")
    parser.add_argument("--doc-stride", default=30, type=int, help="doc stride for long documents")
    parser.add_argument("--filter-sizes", nargs="+", type=int, default=[2, 3, 4])
    parser.add_argument("--num-filters", type=int, default=100)
    parser.add_argument("--drop-prob", type=float, default=0.5)
    parser.add_argument("--lr", default=1e-4, type=float, help="initial learning rate")

    # train
    parser.add_argument("-b", "--batch-size", default=32, type=int)
    parser.add_argument("--epochs", default=10, type=int, help="number of total epochs to run")
    parser.add_argument("-j", "--workers", default=16, type=int, help="number of data loading workers")
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument("--do-train", help="train with train data", action="store_true")
    parser.add_argument("--do-test", help="evaluate on test data", action="store_true")
    parser.add_argument("--gpu", default=0, type=int, help="gpu id")

    args = parser.parse_args()
    if not args.do_train and not args.do_test:
        parser.error("--do-{train|test} option must be specified")

    logging.basicConfig(format="%(asctime)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
    main(args)
