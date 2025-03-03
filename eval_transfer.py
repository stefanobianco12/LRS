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


#transform_test = T.Compose([
#            T.ToTensor()
           # T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
#    ])

#testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
#testloader = data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)


#pyramidnet = models.__dict__['pyramidnet272'](num_classes=10)
#pyramidnet.load_state_dict(torch.load('models/pyramidnet272/pyramidnet272-checkpoint.pth', map_location=device)['state_dict'])

#ResNeXt_29_8_64d = models.__dict__['resnext'](
#                cardinality=8,
#                num_classes=10,
#                depth=29,
#                widen_factor=4,
#                dropRate=0,
#            )
#ResNeXt_29_8_64d = nn.DataParallel(ResNeXt_29_8_64d)
#ResNeXt_29_8_64d.load_state_dict(torch.load('models/resnext-8x64d/model_best.pth.tar', map_location=device)['state_dict'])

#DenseNet_BC_L190_k40 = models.__dict__['densenet'](
#                num_classes=10,
#                depth=190,
#                growthRate=40,
#                compressionRate=2,
#                dropRate=0,
#            )
#DenseNet_BC_L190_k40 = nn.DataParallel(DenseNet_BC_L190_k40)
#DenseNet_BC_L190_k40.load_state_dict(torch.load('models/densenet-bc-L190-k40/model_best.pth.tar', map_location=device)['state_dict'])

#DenseNet_BC_L100_k12 = models.__dict__['densenet'](
#                num_classes=10,
#                depth=100,
#                growthRate=12,
#                compressionRate=2,
#                dropRate=0,
#            )
#DenseNet_BC_L100_k12 = nn.DataParallel(DenseNet_BC_L100_k12)
#DenseNet_BC_L100_k12.load_state_dict(torch.load('models/densenet-bc-L100-k12/model_best.pth.tar', map_location=device)['state_dict'])

#WRN_28_10 = models.__dict__['wrn'](
#                num_classes=10,
#                depth=28,
#                widen_factor=10,
#                dropRate=0.3,
#            )
#WRN_28_10 = nn.DataParallel(WRN_28_10)
#WRN_28_10.load_state_dict(torch.load('models/WRN-28-10-drop/model_best.pth.tar', map_location=device)['state_dict'])

#vgg = models.__dict__['vgg19_bn'](num_classes=10)
#vgg.features = nn.DataParallel(vgg.features)
#vgg.load_state_dict(torch.load('models/vgg19_bn/model_best.pth.tar', map_location=(device))['state_dict'])

#resnet18 = models.__dict__['resnet18'](pretrained=False)
#state_dict = torch.load('models/resnet/resnet18.pt', map_location='cpu')
#resnet18.load_state_dict(state_dict)

#resnet50 = models.__dict__['resnet50'](pretrained=False)
#state_dict = torch.load('models/resnet/resnet50.pt', map_location='cpu')
#resnet50.load_state_dict(state_dict)

#inception_v3 = models.__dict__['inception_v3'](pretrained=False)
#state_dict = torch.load('models/inception_v3/inception_v3.pt', map_location='cpu')
#inception_v3.load_state_dict(state_dict)

#mobilenet_v2 = models.__dict__['mobilenet_v2'](pretrained=False)
#state_dict = torch.load('models/mobilenet_v2/mobilenet_v2.pt', map_location='cpu')
#mobilenet_v2.load_state_dict(state_dict)

#from pprint import pprint
#pprint(torch.hub.list("chenyaofo/pytorch-cifar-models", force_reload=True))

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

#vit = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_vit_l32", pretrained=True)
#vit = nn.DataParallel(vit)
#vit=vit.to(device)

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
#logging.info(('vit', get_success_rate(vit)))
#logging.info(('shufflenetv2', eval_test(shufflenetv2,testloader)))
#logging.info(('vgg19_bn', get_success_rate(vgg)))
#logging.info(('resnet18', get_success_rate(resnet18)))
#logging.info(('resnet50', get_success_rate(resnet50)))
#logging.info(('inception_v3', get_success_rate(inception_v3)))
#logging.info(('mobilenet_v2', get_success_rate(mobilenet_v2)))

#logging.info(('DenseNet_BC_L100_k12', get_success_rate(DenseNet_BC_L100_k12)))
#logging.info(('WRN_28_10', get_success_rate(WRN_28_10)))
#logging.info(('ResNeXt_29_8_64d', get_success_rate(ResNeXt_29_8_64d)))
#logging.info(('DenseNet_BC_L190_k40', get_success_rate(DenseNet_BC_L190_k40)))
#logging.info(('pyramidnet', get_success_rate(pyramidnet)))