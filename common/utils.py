from collections import Counter
import hashlib
import json
from operator import itemgetter
import os
import random

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


class MetricLogger(SummaryWriter):
    def __init__(self, logdir):
        super(MetricLogger, self).__init__(logdir)
        self.step = 0
        self.total_count = Counter()
        self.correct_count = Counter()

    def reset_count(self):
        self.total_count.clear()
        self.correct_count.clear()

    def update_count(self, correct, total):
        self.total_count.update(total)
        self.correct_count.update(correct)

    def accuracy(self, class_wise=False, sort_fn=itemgetter(0)):
        micro_avg = sum(self.correct_count.values()) / float(sum(self.total_count.values()))
        if class_wise:
            class_acc = []
            for label, total in self.total_count.items():
                correct = self.correct_count[label]
                acc = correct / float(total)
                class_acc.append((label, acc, correct, total))
            if sort_fn:
                class_acc.sort(key=sort_fn)
            return micro_avg, class_acc
        return micro_avg


def accuracy(pred, target, with_count=False):
    with torch.no_grad():
        batch_size = target.size(0)
        mask = target == pred
        correct = pred[mask]
        acc = torch.sum(mask).float() / batch_size * 1e2
        if with_count:
            return acc, (correct.tolist(), target.tolist())
        return acc


def get_cache_path(file_path, output_dir):
    h = hashlib.sha1(file_path.encode()).hexdigest()
    cache_path = os.path.join(output_dir, "{}.pt".format(h[:10]))
    return os.path.expanduser(cache_path)


def load_word_embedding(path, add_oov=True):
    W = np.load(path)
    if add_oov:
        W = np.concatenate((np.zeros([1, W.shape[1]]), W), axis=0)
    return W


def load_word_idx(path):
    word_idx = {}
    for idx, word in enumerate(open(path)):
        word_idx[word.strip()] = idx
    return word_idx


def load_list(path):
    return map(str.strip, open(path))


def load_desc(path):
    desc = []
    for l in open(path):
        if l.strip():
            desc.append(l.split("\t")[0].strip())
    return desc
