from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
import argparse
import matplotlib.pyplot as plt
from models import *
from utils import *
from torch.autograd import Variable
global best_acc

parser=argparse.ArgumentParser(description='pytorch cifar10 training')
parser.add_argument('--resume','-r',action='store_true',help='resume from checkpoint')
parser.add_argument('--deep-compress','-d',default=False)
parser.add_argument('--train','-t',action='store_true')
parser.add_argument('--prune', action='store_true')
parser.add_argument('--model', '-m', default='', help='VGG-16, ResNet-18, LeNet')
args = parser.parse_args()

model_name=args.model
if model_name!='':
    model_weights=models[model_name]
    models={model_name:model_weights}
train_loader,test_loader=get_data()

def train_models():
    epochs=[[100,0.1],[50,0.01],[50,0.001]]
    for model_name,model_weights in models.items():
        first_iter=True
        print('Training ',model_name)
        for num_epochs , learning_rate in epochs:
            for epoch in range(1,num_epochs):
                if first_iter:
                    best_acc=0
                else:
                    model_name,model_weights,best_acc=load_best(model_name,model_weights)
                optimizer=optim.SGD(model_weights.parameters(),lr=learning_rate,momentum=0.9,weight_decay=5e-4)
                train(model_weights,epoch,optimizer,train_loader)
                best_acc=test(model_name,model_weights,test_loader,best_acc)

def finetune(model_weights,best_acc,epochs,lr):
    optimizer=optim.SGD(model_weights.parameters(),lr=lr,momentum=0.9,weight_decay=5e-4)
    for epoch in range(1,epochs):
        train(model_weights,epoch,optimizer,train_loader)
        best_acc = test(model_name, model_weights, test_loader, best_acc)
    return best_acc
def deep_compression():
    for model_name,model_weights in models.items():
        base_model_name = model_name
        for s in [50.,60.,70.,80.,90.]:
            model_name,model_weights=load_best(model_name,model_weights)
            model_name=model_name+str(s)
            best_acc=0.

            model_weights=sparsify(model_weights,s)

            best_acc=finetune(model_weights,best_acc,30,0.01)
            best_acc = finetune(model_weights, best_acc, 30, 0.001)

            new_model = compress_convs(model_weights, compressed_models[base_model_name])
            # finetune again - this is just to save the model
            finetune(new_model, 0., 10, 0.001)

if args.train:
    train_models()
if args.deep_compress:
    deep_compression()
    print('compression model.....')
    writer
