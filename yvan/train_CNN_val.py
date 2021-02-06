# ====================================================
# imports
# ====================================================
import sys

import os
import time
import random
import copy
import argparse
from shutil import copyfile

import numpy as np
import pandas as pd
import timm

from sklearn.model_selection import StratifiedKFold, train_test_split

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau
from utils.custom_losses import LabelSmoothingLoss, FocalLoss, FocalCosineLoss, SymmetricCrossEntropy, BiTemperedLogisticLoss
from utils.dataset_utils import SingleFrameDataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

import yaml
import warnings

warnings.filterwarnings('ignore')

# ====================================================
# Utils
# ====================================================

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# ====================================================
# Transforms
# ====================================================
def get_transforms(data):
    if data == 'train':
        return A.Compose([
            A.Resize(CFG['size'], CFG['size']),
            # A.RandomResizedCrop(CFG['size'], CFG['size'], scale=(0.8, 1.0)),
            # A.Transpose(p=0.5),
            A.HorizontalFlip(p=0.5),
            # A.VerticalFlip(p=0.5),
            # A.ShiftScaleRotate(p=0.5),
            A.Normalize(
                mean=CFG['mean'],
                std=CFG['std'],
            ),
            ToTensorV2(),
        ])

    elif data == 'valid':
        return A.Compose([
            A.Resize(CFG['size'], CFG['size']),
            A.Normalize(
                mean=CFG['mean'],
                std=CFG['std'],
            ),
            ToTensorV2(),
        ])


# ====================================================
# Model
# ====================================================
def getModel(architecture_name, target_size, pretrained=False):
    net = timm.create_model(architecture_name, pretrained=pretrained)
    net_cfg = net.default_cfg
    last_layer = net_cfg['classifier']
    num_ftrs = getattr(net, last_layer).in_features
    setattr(net, last_layer, nn.Linear(num_ftrs, target_size))
    return net


# ====================================================
# Criterion
# ====================================================
def get_criterion():
    if CFG['criterion'] == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss()
    elif CFG['criterion'] == 'LabelSmoothing':
        criterion = LabelSmoothingLoss(classes=CFG['target_size'], smoothing=CFG['smoothing'])
    elif CFG['criterion'] == 'FocalLoss':
        criterion = FocalLoss()
    elif CFG['criterion'] == 'FocalCosineLoss':
        criterion = FocalCosineLoss()
    elif CFG['criterion'] == 'SymmetricCrossEntropyLoss':
        criterion = SymmetricCrossEntropy(classes=CFG['target_size'])
    elif CFG['criterion'] == 'BiTemperedLoss':
        criterion = BiTemperedLogisticLoss(t1=CFG['t1'], t2=CFG['t2'], smoothing=CFG['smoothing'])
    return criterion


# ====================================================
# Scheduler
# ====================================================
def get_scheduler(optimizer):
    if CFG['scheduler'] == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=CFG['factor'], patience=CFG['patience'],
                                      verbose=True,
                                      eps=CFG['eps'])
    elif CFG['scheduler'] == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(optimizer, T_max=CFG['T_max'], eta_min=CFG['min_lr'], last_epoch=-1)
    elif CFG['scheduler'] == 'CosineAnnealingWarmRestarts':
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=CFG['T_0'], T_mult=1, eta_min=CFG['min_lr'],
                                                last_epoch=-1)
    return scheduler


# ====================================================
# Optimizer
# ====================================================
def get_optimizer():
    if CFG['optimizer'] == 'Adam':
        optimizer = Adam(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'], amsgrad=False)
    elif CFG['optimizer'] == 'AdamW':
        optimizer = AdamW(model.parameters(), lr=CFG['lr'], amsgrad=False)
    elif CFG['optimizer'] == 'SGD':
        optimizer = SGD(model.parameters(), lr=CFG['lr'], momentum=CFG['momentum'])
    return optimizer


# ====================================================
# Main
# ====================================================
if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # ====================================================
    #  Argparser
    # ====================================================
    # '/storage/plzen1/home/grubiv/ChaLearn'
    parser = argparse.ArgumentParser()
    parser.add_argument('-core_directory', type=str, default='/storage/plzen1/home/grubiv/ChaLearn', help='Core directory')
    parser.add_argument('-config_file', type=str, default='config1.yml', help='Config file')
    args = parser.parse_args()
    core_dir = args.core_directory

    # ====================================================
    # Config file loading settings
    # ====================================================
    with open(os.path.join(core_dir, 'CFG', args.config_file), "r") as ymlfile:
        CFG = yaml.safe_load(ymlfile)

    seed_torch(seed=CFG['seed'])

    # ====================================================
    # Directory settings
    # ====================================================
    output_dir = core_dir + CFG['output_dir'] + CFG['model_name'] + '/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    copyfile(os.path.join(core_dir, 'CFG', args.config_file), os.path.join(output_dir, args.config_file))

    # ====================================================
    # Data loading
    # ====================================================
    train = pd.read_csv(core_dir + CFG['train_csv'])
    dev = pd.read_csv(core_dir + CFG['dev_csv'])

    # ====================================================
    # Train loop
    # ====================================================
    print(f"========== architecture: {CFG['model_name']} training ==========")
    # ====================================================
    # model & optimizer
    # ====================================================
    model = getModel(CFG['model_name'], CFG['target_size'], pretrained=True)
    CFG['mean'] = list(model.default_cfg['mean'])
    CFG['std'] = list(model.default_cfg['std'])
    model.to(device)
    optimizer = get_optimizer()
    scheduler = get_scheduler(optimizer)
    criterion = get_criterion()
    # ====================================================
    # loader
    # ====================================================
    train_dataset = SingleFrameDataset(train, CFG['data_path'], transform=get_transforms(data='train'))
    dev_datasset = SingleFrameDataset(dev,  CFG['data_path'], transform=get_transforms(data='valid'))

    trainloader = DataLoader(train_dataset, batch_size=CFG['batch_size'], shuffle=True,
                             num_workers=CFG['num_workers'],
                             drop_last=False)
    devloader = DataLoader(dev_datasset, batch_size=CFG['batch_size'], shuffle=False,
                           num_workers=CFG['num_workers'],
                           drop_last=False)
    # ====================================================
    # loop
    # ====================================================
    best_score = 0.
    best_loss = np.inf

    for epoch in range(CFG['epochs']):
        start_time = time.time()
        model.train()
        # model.apply(freeze_norm_stats)
        train_loss = 0.0
        running_loss = 0.0
        model.zero_grad()
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            if CFG['gradient_accumulation_steps'] > 1:
                loss = loss / CFG['gradient_accumulation_steps']
            loss.backward()
            if (i + 1) % CFG['gradient_accumulation_steps'] == 0:
                optimizer.step()
                model.zero_grad()

            running_loss += loss.item()
            train_loss += loss.item()

            # print statistics
            if (i + 1) % (CFG['print_freq'] * CFG['gradient_accumulation_steps']) == 0:
                print('[%d][%d]  Train loss: %.3f' % (
                epoch + 1, (i + 1) / CFG['gradient_accumulation_steps'], running_loss / CFG['print_freq']))
                running_loss = 0.0

        model.eval()
        print('===========================================')
        print('[%d] Average training loss: %.3f' %
              (epoch + 1, CFG['batch_size'] * CFG['gradient_accumulation_steps'] * train_loss / len(train_dataset)))

        test_loss = 0.0
        total = 0
        correct = 0
        with torch.no_grad():
            for j, data in enumerate(devloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(
            '[%d]  Average validation loss: %.3f' % (epoch + 1, CFG['batch_size'] * test_loss / len(dev_datasset)))
        epoch_acc = 100 * correct / total
        print('Validation accuracy: %.2f %%' % (epoch_acc))
        scheduler.step(test_loss)
        if epoch_acc > best_score:
            best_score = epoch_acc
            best_epoch = epoch + 1
            best_model_wts = copy.deepcopy(model.state_dict())
        if CFG['saveEachEpoch'] == True:
            torch.save(model.state_dict(),
                       output_dir + '/' + CFG['model_name'] + '_' + str(epoch + 1) + '.pth')
        end_time = time.time()
        print("Time for epoch: " + str(end_time - start_time))
        print('===========================================')
        sys.stdout.flush()

    print('Best validation accuracy: ' + str(best_score) + 'in epoch no.: ' + str(best_epoch))
    torch.save(best_model_wts, output_dir + '/' + CFG['model_name'] + '_best.pth')
    torch.save(model.state_dict(), output_dir + '/' + CFG['model_name'] + '_last.pth')
