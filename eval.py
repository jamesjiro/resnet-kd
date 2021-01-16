import copy
import os
import time

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.autograd.profiler as profiler
import torch.nn as nn
import torch.optim as optim

from PIL import Image
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, models, utils
from torchvision import transforms as T
from tqdm import tqdm


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
data_transforms = {
    'train': T.Compose([
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize
    ]),
    'val': T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize
    ])
}
image_datasets = {x: datasets.CIFAR10(".", train=(x == 'train'), download=(x == 'train'),
                                      transform=data_transforms[x]) 
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, 
                                              shuffle=(x == 'train'), num_workers=4)
               for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))
model.load_state_dict(torch.load("resnet50.pt"))
model.eval()
model.to(device)

running_corrects = 0
for i, (inputs, labels) in enumerate(tqdm(dataloaders['val'])):
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)

    running_corrects += torch.sum(preds == labels.data)

acc = running_corrects.double() / dataset_sizes['val']

print(f'Acc: {acc:.4f}')
