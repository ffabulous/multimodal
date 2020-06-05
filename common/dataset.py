from collections import defaultdict
import json
import os
import random

import torch
from torch.nn.functional import pad
from torch.utils.data import Dataset
import torchvision


def tensorize(ids, max_seq_len, min_seq_len, doc_stride, pad_idx):
    if len(ids) < min_seq_len:
        return []
    elif len(ids) < max_seq_len:
        return [pad(torch.LongTensor(ids), (0, max_seq_len - len(ids)), value=pad_idx)]
    T = []
    n = 0
    while True:
        idx = doc_stride * n
        t = ids[idx : idx + max_seq_len]
        if n > 0 and (doc_stride < 1 or len(t) < min_seq_len):
            break
        T.append(pad(torch.LongTensor(t), (0, max_seq_len - len(t)), value=pad_idx))
        n += 1
    return T


# TODO: consider torchtext
class TextDataset(Dataset):
    def __init__(self, file_path, label_path, max_seq_len, min_seq_len, doc_stride=0, pad_idx=0):
        self.classes = list(map(str.strip, open(label_path).readlines()))
        self.clipno_ids = defaultdict(list)
        self.labels = []
        self.indices = []
        self.seq_len = max_seq_len
        self.pad_idx = pad_idx
        idx = 0
        for l in open(file_path):
            toks = l.strip().split("\t")
            ids = json.loads(toks[2])
            for t in tensorize(ids, max_seq_len, min_seq_len, doc_stride, pad_idx):
                self.clipno_ids[toks[0]].append(idx)
                self.labels.append(int(toks[1]))
                self.indices.append(t)
                idx += 1

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.indices[idx], self.labels[idx]

    def zeros(self):
        return torch.LongTensor([self.pad_idx for _ in range(self.seq_len)])


class VideoDataset(torchvision.datasets.Kinetics400):
    def __getitem__(self, idx):
        video, audio, info, video_idx = self.video_clips.get_clip(idx)
        video_path, label = self.samples[video_idx]
        video_id = os.path.basename(video_path).split(os.extsep)[0]
        if self.transform is not None:
            video = self.transform(video)
        return video_id, video, label


class MultiModalDataset(Dataset):
    def __init__(self, video_cache_path, text_cache_path, map_path):
        self.video_data = torch.load(video_cache_path)[0]
        self.text_data = torch.load(text_cache_path)
        self.video_clip_map = {}
        for l in open(map_path):
            toks = l.strip().split("\t", 2)
            if len(toks) > 1:
                video_id, clipno = toks[:2]
                self.video_clip_map[video_id] = clipno

    def __len__(self):
        return len(self.video_data)

    def __getitem__(self, idx):
        video_id, video, label = self.video_data[idx]
        clipno = self.video_clip_map.get(video_id)
        if clipno and clipno in self.text_data.clipno_ids:
            ids = self.text_data.clipno_ids[clipno]
            text, _ = self.text_data[random.choice(ids)]
        else:
            text = self.text_data.zeros()
        return video, text, label
