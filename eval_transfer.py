import os, sys
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import argparse
import numpy as np
import models
import logging
#from utils.utils import Normalize, norm, set_seed
from utils import Normalize,set_seed
import torchvision.transforms as T
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torch.utils.data as data

parser = argparse.ArgumentParser()
parser.add_argument('--results_dir', type=str, default='./results')
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[
        logging.FileHandler("./results.log"),
        logging.StreamHandler(),
    ],
)
logging.info(args)

set_seed(0)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

resnet56 = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet56", pretrained=True)
resnet56 = nn.DataParallel(resnet56)
resnet56=resnet56.to(device)

resnet20 = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True)
resnet20 = nn.DataParallel(resnet20)
resnet20=resnet20.to(device)

vgg19_bn = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_vgg19_bn", pretrained=True)
vgg19_bn = nn.DataParallel(vgg19_bn)
vgg19_bn=vgg19_bn.to(device)

mobilenetv2 = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_mobilenetv2_x1_4", pretrained=True)
mobilenetv2 = nn.DataParallel(mobilenetv2)
mobilenetv2=mobilenetv2.to(device)

repvgg = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_repvgg_a2", pretrained=True)
repvgg = nn.DataParallel(repvgg)
repvgg=repvgg.to(device)

shufflenetv2 = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_shufflenetv2_x2_0", pretrained=True)
shufflenetv2 = nn.DataParallel(shufflenetv2)
shufflenetv2=shufflenetv2.to(device)



def get_success_rate(model):
    model = nn.Sequential(
    Normalize(), 
    model)

    model.to(device)
    model.eval()
    fooled = 0
    total = 0
    advfile_ls = os.listdir(args.results_dir)
    target = torch.from_numpy(np.load(args.results_dir + '/labels.npy')).long()
    for advfile_ind in range(len(advfile_ls)-1):
        adv_batch = torch.from_numpy(np.load(args.results_dir + '/batch_{}.npy'.format(advfile_ind))).float() / 255.0
        adv_batch_size = adv_batch.shape[0]
        labels = target[advfile_ind * adv_batch_size : advfile_ind * adv_batch_size + adv_batch.shape[0]]
        inputs = adv_batch.clone()
        inputs, labels = inputs.to(device), labels.to(device)
        with torch.no_grad():
            preds = torch.argmax(model(inputs), dim=1).view(1,-1)
        fooled += (labels != preds.squeeze(0)).sum().item()
        total += adv_batch_size
    print(fooled)
    print(total)
    return round(fooled / total * 100., 2)

def eval_test(model,testloader):
    model = nn.Sequential(
    Normalize(), 
    model)

    model.to(device)
    model.eval()
    correct = 0
    total = 0
    for ind, (ori_img, label) in enumerate(testloader):
        ori_img, label = ori_img.to(device), label.to(device)
        output=model(ori_img)
        correct+=(output.argmax(1) == label).sum()
        total+=len(ori_img)
    return correct/total

logging.info(('resnet56', get_success_rate(resnet56)))
logging.info(('resnet20', get_success_rate(resnet20)))
logging.info(('vgg19_bn', get_success_rate(vgg19_bn)))
logging.info(('mobilenetv2', get_success_rate(mobilenetv2)))
logging.info(('repvgg', get_success_rate(repvgg)))