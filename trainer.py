import copy
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


TEACHER = "resnet50.pt"


def load_teacher(teacher, num_outputs):
    for param in teacher.parameters():
        param.requires_grad = False

    num_ftrs = teacher.fc.in_features
    teacher.fc = nn.Linear(num_ftrs, num_outputs)
    teacher.load_state_dict(torch.load(TEACHER))
    
    
def loss_kd(outputs, teacher_outputs, labels, temp, alpha):
    beta = 1. - alpha
    q = F.log_softmax(outputs/temp, dim=1)
    p = F.softmax(teacher_outputs/temp, dim=1)
    soft_loss = nn.KLDivLoss(reduction='batchmean')(q, p)
    hard_loss = nn.CrossEntropyLoss()(outputs, labels)
    KD_loss = alpha * hard_loss + beta * soft_loss

    return KD_loss


def train_model(model, 
                dataloaders,
                criterion, 
                optimizer, 
                scheduler, 
                tensorboard_writer,
                device,
                temp,
                alpha,
                teacher=None,
                num_epochs=10):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    dataset_sizes = {split: len(dataloaders[split].dataset)
                     for split in ['train', 'val']}
    if teacher:
        print(f'Temperature: {temp}, Alpha: {alpha}')

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for batch_idx, (inputs, labels) in enumerate(tqdm(dataloaders[phase])):
                inputs, labels = inputs.to(device), labels.to(device)

                if teacher:
                    teacher_outputs = teacher(inputs)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)

                    if teacher: 
                        loss = criterion(outputs, teacher_outputs, labels, temp, 
                                         alpha)
                    else:
                        loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                if batch_idx % 100 == 99 and phase == 'train':
                    tensorboard_writer.add_scalar(
                        'training_loss',
                        running_loss / (100 * inputs.size(0)),
                        epoch * dataset_sizes[phase] + batch_idx)

            if  phase == 'train':
                scheduler.step()
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()
    
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')
    model.load_state_dict(best_model_wts)

    return model