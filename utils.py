import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np

def tulip(model, loss_fn, img, label, step_size1, lam1):

    # perturbation
    model.eval()
    img_ = img.clone().detach()
    img_.requires_grad=True
    logits = model(img_)  # Forward pass
    loss_ce = loss_fn(logits, label)
    model.zero_grad()
    loss_ce.backward()
    z=(img_+step_size1*img_.grad.data.sign()).detach()
    #z = torch.clip(z,0,1)
    model.zero_grad()

    #loss lip
    model.train()
    batch_size = img.shape[0]
    logits = model(img)  # Forward pass
    loss_ce = loss_fn(logits, label)
    loss_ori = torch.nn.CrossEntropyLoss(reduction='none')
    loss_R= (loss_ori(model(z),label) -loss_ori(model(img).detach(),label)).pow(2).mean()
    final_loss = loss_ce + (lam1) * loss_R 
    return logits,final_loss

def set_seed(seed):
    # Set Python random seed
    random.seed(seed)
    
    # Set numpy random seed
    np.random.seed(seed)
    
    # Set torch random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Set deterministic behavior (optional)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Normalize(nn.Module):
    def __init__(self):
        super(Normalize, self).__init__()
        self.mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)  # CIFAR-10 mean
        self.std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)   # CIFAR-10 std

    def forward(self, x):
        return (x - self.mean.to(x.device)) / self.std.to(x.device)