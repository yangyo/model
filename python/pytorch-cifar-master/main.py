'''Train CIFAR10 with PyTorch.'''

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from load import loadCIFAR10
import os
import argparse

from models import *
from utils import progress_bar
from torch.autograd import Variable


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()



def train(model, batchSize, epoch, useCuda=True,save_point=600,modelpath=''):
    train_losses = []
    test_losses = []
    if useCuda:
        model = model.cuda()
    ceriation = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    trainLoader, testLoader = loadCIFAR10(batchSize=batchSize)
    step=0
    step_test=0
    for i in range(epoch):
        # trainning
        sum_loss = 0
        for batch_idx, (x, target) in enumerate(trainLoader):
            optimizer.zero_grad()
            if useCuda:
                x, target = x.cuda(), target.cuda()
            x, target = Variable(x), Variable(target)
            out = model(x)

            loss = ceriation(out, target)
            sum_loss += loss.data[0]
            train_losses.append([loss,step])

            loss.backward()
            optimizer.step()
            step+=1
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(trainLoader):
                print('==>>> epoch: {}, batch index: {}, train loss: {:.6f}'.format(i, batch_idx + 1, sum_loss/(batch_idx+1)))

            if (step+1)%save_point==0:
                torch.save(model.state_dict(),modelpath)
                print("save finish--------------------")
        # testing
        correct_cnt, sum_loss = 0, 0
        total_cnt = 0
        for batch_idx, (x, target) in enumerate(testLoader):
            x, target = Variable(x, volatile=True), Variable(target, volatile=True)
            if useCuda:
                x, target = x.cuda(), target.cuda()
            out = model(x)
            loss = ceriation(out, target)
            sum_loss+=loss.data[0]
            test_losses.append([loss.data[0],step_test])
            step_test+=1
            _, pred_label = torch.max(out.data, 1)
            total_cnt += x.data.size()[0]
            correct_cnt += (pred_label == target.data).sum()

            # smooth average
            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(testLoader):
                print('==>>> epoch: {}, batch index: {}, test loss: {:.6f}, acc: {:.3f}'.format(
                    i, batch_idx + 1, sum_loss/(batch_idx+1), correct_cnt * 1.0 / total_cnt))

    return test_losses,train_losses
def plot_loss(train,test):
    import matplotlib.pyplot as plt
    plt.switch_backend('agg')
    x1=[]
    y1=[]
    for loss,index in train:
        x1.append(index)
        y1.append(loss)
    plt.plot(x1,y1,marker='*',mec='r',mfc='w')
    plt.legend()
    plt.xlabel("training epoches")
    plt.ylabel("loss")
    plt.title("training loss")
    plt.show()
    plt.savefig("./save/train_loss.jpg")
    plt.close()
    x2=[]
    y2=[]
    for loss,index in test:
        x2.append(index)
        y2.append(loss)
    plt.plot(x2,y2,"b--",linewidth=1)
    plt.xlabel("test epoches")
    plt.ylabel("loss")
    plt.title("test loss")
    plt.savefig("./save/test_loss.jpg")
    plt.close()

if __name__ == '__main__':
    # Model
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.t7')
        net = checkpoint['net']
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
    else:
        print('==> Building model..')
        #net = VGG('VGG19')
        #net = DenseNet121()
        # net = PreActResNet18()
        # net = GoogLeNet()
        #net = DenseNet121()
        net=NewModel1()
        # net = ResNeXt29_2x64d()
        # net = MobileNet()
        #net = MobileNetV2()
        # net = DPN92()
        # net = ShuffleNetG2()
        # net = SENet18()
    use_cuda = torch.cuda.is_available()
        #net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        #cudnn.benchmark = True
    model_path="/home/zhangxu/python_project/test/save/new_model.pt"
    train_loss,test_loss=train(model=net, epoch=10, batchSize=128, useCuda=use_cuda,save_point=600,modelpath=model_path)
    plot_loss(train_loss,test_loss)

