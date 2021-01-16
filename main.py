import argparse

import torch
import torch.autograd.profiler as profiler
import torch.nn as nn
import torch.optim as optim

from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, models
from torchvision import transforms as T

from trainer import load_teacher, loss_kd, train_model


resnets = ['resnet18', 'resnet34', 'resnet50']
students = ['resnet18', 'resnet34']
modes = ['train', 'distil', 'profile']
parser = argparse.ArgumentParser(description='Transfer Learning on CIFAR10')
parser.add_argument('-m', '--mode', default='train', choices=modes,
                    help='program mode: ' +
                        ' | '.join(resnets) +
                        ' (default: train)')
parser.add_argument('-a', '--arch', default='resnet18', choices=resnets,
                    help='model architecture: ' +
                        ' | '.join(resnets) +
                        ' (default: resnet18)')
parser.add_argument('-s', '--student', default='resnet18', choices=resnets,
                    help='student architecture: ' +
                        ' | '.join(resnets) +
                        ' (default: resnet18)')
parser.add_argument('--epochs', default=25, type=int,
                    help='number of total epochs to run')
parser.add_argument('-t', '--temp', default=1., type=float,
                    help='temperature for distillation')
parser.add_argument('--alpha', default=0.5, type=float,
                    help='weighting for hard loss during distillation')
parser.add_argument('-c', '--ckpt', default='resnet50.pt', type=string,
                    help='model checkpoint to load for inference')


def profile(arch, device):
    model = models.__dict__[arch](pretrained=False)
    model.eval()

    input_batch = torch.randn(1, 3, 224, 224)
    input_batch = input_batch.to(device)
    model.to(device)

    # Model latency

    model(input_batch) # warm-up

    with torch.no_grad():
        with profiler.profile(record_shapes=True, use_cuda=True) as prof:
            with profiler.record_function("model_inference"):
                model(input_batch)

    print(prof.key_averages(group_by_input_shape=True).table(sort_by="cuda_time_total", row_limit=10))

    with profiler.profile(profile_memory=True, record_shapes=True, use_cuda=True) as prof:
        model(input_batch)

    print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))


def train(arch, mode, temp, alpha, epochs, device):
    model = models.__dict__[arch](pretrained=True)

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
    image_datasets = {split: datasets.CIFAR10(".", train=(split == 'train'), download=(split == 'train'),
                                        transform=data_transforms[split]) 
                    for split in ['train', 'val']}
    dataloaders = {split: torch.utils.data.DataLoader(image_datasets[split], batch_size=64, 
                                                shuffle=(split == 'train'), num_workers=4)
                for split in ['train', 'val']}
    class_names = image_datasets['train'].classes

    model_ft = model
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, len(class_names))
    model_ft = model_ft.to(device)

    if mode == 'distil':
        criterion = loss_kd
        teacher = models.resnet50(pretrained=True)
        load_teacher(teacher, len(class_names))
        teacher.to(device)
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    writer = SummaryWriter(f'runs/cifar10_{mode}_{arch}_{epochs}')
    model_ft = train_model(model_ft, dataloaders, criterion, optimizer_ft, 
                           exp_lr_scheduler, writer, device, temp, alpha,
                           teacher, epochs)
    save_path = "{mode}_{arch}_{num_epochs}.pt"
    torch.save(model_ft.state_dict(), save_path)


def main():
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(f'Mode: {args.mode}')
    if args.mode == 'train':
        print(f'Training model: {args.arch}...')
        train(args.arch, args.mode, args.temp, args.alpha, args.epochs, device)
    elif args.mode == 'distil':
        print(f'Training student: {args.student}...')
        train(args.arch, args.mode, args.temp, args.alpha, args.epochs, device)
    elif args.mode == 'profile':
        profile(args.arch, device)


if __name__ == '__main__':
    main()
