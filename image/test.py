#!/bin/env python


import argparse
from collections import defaultdict
from datetime import datetime
import json
import logging
import os
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from sklearn.metrics import confusion_matrix
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageFolderWithPaths(datasets.ImageFolder):
    # https://gist.github.com/andrewjong/6b02ff237533b3b2c554701fb53d5c4d
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]    # image file path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


# argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='data')
parser.add_argument('--dump-dir', default='dump/201903130303')
parser.add_argument('--model-name', default='model-1')
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--num-workers', type=int, default=4)
parser.add_argument('--gpu', type=int, choices=range(torch.cuda.device_count()))
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()

if args.debug:
    logging.basicConfig(
        format='%(asctime)s:%(levelname)s:%(name)s: %(message)s',
        datefmt='%Y-%m-%dT%H:%M:%S'
    )
    logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)    # module name


# data
start = datetime.now()
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
image_dataset = ImageFolderWithPaths(os.path.join(args.data_dir, 'test'), transform)
data_loader = torch.utils.data.DataLoader(image_dataset, args.batch_size, num_workers=args.num_workers)
dataset_size = len(image_dataset)
classes = image_dataset.classes
num_classes = len(classes)
end = datetime.now()
logger.info('Loading: {}'.format(end - start))
logger.info('Data: {}'.format(dataset_size))


# model
start = end
model = models.resnet50()
model.fc = nn.Linear(model.fc.in_features, num_classes)
weights = torch.load(os.path.join(args.dump_dir, args.model_name + '.bin'), map_location=torch.device('cpu'))
model.load_state_dict(weights)
_ = model.eval()
# device
if torch.cuda.is_available() and args.gpu is not None:
    torch.cuda.set_device(args.gpu)    # TODO: discouraged
    device = torch.device('cuda:{}'.format(args.gpu))
else: device = torch.device('cpu')
_ = model.to(device)
end = datetime.now()
logger.info('Model: {}'.format(end - start))


# inference
start = end
correct = 0
preds = torch.LongTensor([])
trues = torch.LongTensor([])
result = defaultdict(list)
for x, y_true, imgs in data_loader:
    x = x.to(device)
    y_true = y_true.to(device)
    y_h = model(x)
    logits = torch.softmax(y_h, 1)
    y_prob, y_pred = torch.max(logits, 1)
    correct += torch.sum(y_pred == y_true.data)
    trues = torch.cat((trues, y_true.cpu()))
    preds = torch.cat((preds, y_pred.cpu()))
    l_true = [classes[x] for x in y_true]
    l_pred = [classes[x] for x in y_pred]
    for true, pred, img, prob in zip(l_true, l_pred, imgs, y_prob):
        result[','.join([true, pred])].append((img, float(prob)))
    torch.cuda.empty_cache()
end = datetime.now()
logger.info('Inference: {}'.format(end - start))


# result
print('Accuracy: {:.4f}'.format(correct.double() / dataset_size))
print(classes)
conf_mtx = confusion_matrix(trues, preds)
print(conf_mtx)
json.dump(result, open(os.path.join(args.dump_dir, args.model_name + '.json'), 'w'), indent=2)
