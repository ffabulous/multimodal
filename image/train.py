#!/bin/env python


import copy
import argparse
from datetime import datetime
import json
import logging
import os
import shutil
import warnings
warnings.filterwarnings("ignore")

from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


# argparse
parser = argparse.ArgumentParser()
parser.add_argument('--transform-resize-scale', type=float, default=0.7)
parser.add_argument('--use-pretrained', action='store_true')
parser.add_argument('--fine-tune', action='store_true')
parser.add_argument('--drop-prob', type=float, default=0.8)
parser.add_argument('--num-epochs', type=int, default=20)
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--every-n-step', type=int, default=100)    # logging
parser.add_argument('--num-workers', type=int, default=4)
parser.add_argument('--gpu', type=int, choices=range(torch.cuda.device_count()))
parser.add_argument('--data-dir', default='data')
parser.add_argument('--dump-dir', default='dump')
parser.add_argument('--log-dir', default='logs')
parser.add_argument('--exp-name', default=datetime.now().strftime('%Y%m%d%H%M'))
args = parser.parse_args()


# data
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'dev': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
splits = list(data_transforms.keys())

image_datasets = {}
data_loaders = {}
for split in splits:
    data_path = os.path.join(args.data_dir, split)
    transform = data_transforms[split]
    image_datasets[split] = datasets.ImageFolder(data_path, transform)
    data_loaders[split] = torch.utils.data.DataLoader(
        image_datasets[split], args.batch_size, shuffle=True, num_workers=args.num_workers
    )
dataset_sizes = {split: len(image_datasets[split]) for split in splits}
classes = image_datasets[splits[0]].classes
num_classes = len(classes)

# model
model = models.resnet50(pretrained=args.use_pretrained)
if not args.fine_tune:
    for param in model.parameters():
        param.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, num_classes)
nn.init.xavier_uniform_(model.fc.weight)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# device
if torch.cuda.is_available() and args.gpu is not None:
    torch.cuda.set_device(args.gpu)
    device = torch.device('cuda:{}'.format(args.gpu))
else: device = torch.device('cpu')
_ = model.to(device)

# dirs
dump_dir = os.path.join(args.dump_dir, args.exp_name)
os.makedirs(dump_dir, exist_ok=True)
json.dump(vars(args), open(os.path.join(dump_dir, 'conf.json'), 'w'), indent=2)
log_dir = os.path.join(args.log_dir, args.exp_name)
if os.path.exists(log_dir):
    shutil.rmtree(log_dir)
logger = SummaryWriter(log_dir)


# train
step = 1
best_model = copy.deepcopy(model.state_dict())
best_accuracy = 0.0

for epoch in range(args.num_epochs):
    start = datetime.now()
    print('Epoch {}/{}'.format(epoch+1, args.num_epochs))

    for phase in ['train', 'dev']:
        if phase == 'train':
            model.train()
        else:
            model.eval()

        epoch_loss = 0.0
        epoch_correct = 0
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        for x, y_true in data_loaders[phase]:
            x = x.to(device)
            y_true = y_true.to(device)
            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                y_h = model(x)
                _, y_pred = torch.max(y_h, 1)
                loss = criterion(y_h, y_true)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    if step % args.every_n_step == 0:
                        logger.add_scalar('train/loss', loss.item(), step)
                        print('Step {:6d} loss: {:.4f}'.format(step, loss.item()))

            correct = torch.sum(y_pred == y_true.data)
            epoch_loss += loss.item() * x.size(0)
            epoch_correct += correct
            if step % args.every_n_step == 0:
                logger.add_scalar('train/accuracy', correct.double() / data_loaders[phase].batch_size, step)
            step += 1

        epoch_loss /= dataset_sizes[phase]
        epoch_accuracy = epoch_correct.double() / dataset_sizes[phase]
        print('{} loss: {:.4f} accuracy: {:.4f}'.format(phase, epoch_loss, epoch_accuracy))

        if phase == 'dev':
            if epoch_accuracy > best_accuracy:
                best_accuracy = epoch_accuracy
                best_model = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), '{}/model-{}.bin'.format(dump_dir, epoch + 1))
            logger.add_scalar('dev/loss', epoch_loss, step)
            logger.add_scalar('dev/accuracy', epoch_accuracy, step)

    end = datetime.now()
    print('Elapsed: {}'.format(end - start))

print('Best accuracy: {:.4f}'.format(best_accuracy))
