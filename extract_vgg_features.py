import logging
import os.path as osp
from glob import glob
from os import makedirs
from sys import stdout

import numpy as np

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision.models import vgg


def setup_custom_logger(name):
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    handler = logging.FileHandler('log.txt', mode='w')
    handler.setFormatter(formatter)
    screen_handler = logging.StreamHandler(stream=stdout)
    screen_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.addHandler(screen_handler)
    return logger


logger = setup_custom_logger(__name__)


def get_network(device):
    net = vgg.vgg19(pretrained=True)

    # get features from last layer
    new_classifier = nn.Sequential(*list(net.classifier.children())[:-1])
    net.classifier = new_classifier
    return net.to(device)


def get_xforms():
    normy = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

    xforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normy,
    ])
    return xforms


def get_dataloader(im_dir, xforms, batch_size=4, num_workers=4):
    data = DataLoader(ImageFolder(im_dir, xforms),
                      batch_size=batch_size,
                      shuffle=True,
                      num_workers=num_workers)
    return data


def choose_device(use_cuda=True):
    cuda = use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    logger.info('Using: ' + str(device))
    return device


def get_features(mydata, net, device):
    features = []
    for batch_idx, data in enumerate(mydata):
        input, target = data[0].to(device), data[1].to(device)
        features.append(net(input))
    return np.vstack([f.numpy() for f in features])


def save_features(features, path):
    np.save(path, features, allow_pickle=False)


def pipeline(im_dir, out_path):
    if not osp.exists(osp.dirname(out_path)):
        makedirs(out_path)
    with torch.no_grad():
        device = choose_device()
        network = get_network(device)
        mydata = get_dataloader(im_dir, get_xforms())
        features = get_features(mydata, network, device)
        save_features(features, out_path)


if __name__ == '__main__':
    pipeline('./testdata', './outputs')
