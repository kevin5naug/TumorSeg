import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import h5py
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
import numpy as np
import time
import torch.nn.init as ini
import multiprocessing
from multiprocessing import Queue
import random
import pickle

class LocalPathConv(nn.Module):
    def __init__(self):
        super(LocalPathConv, self).__init__()
        self.local_conv1 = nn.Conv2d(4, 64, 7)
        self.dp1 = nn.Dropout2d(.5)
        self.local_conv2 = nn.Conv2d(64, 64, 3)
        self.dp2 = nn.Dropout2d(.5)
        self.total_conv = nn.Conv2d(64, 5, 21)

    def forward(self, x):
        x = self.local_conv1(x)
        x = self.dp1(x)
        x = F.max_pool2d(F.relu(x), 4, stride = 1)
        x = self.local_conv2(x)
        x = self.dp2(x)
        x = F.max_pool2d(F.relu(x), 2, stride = 1)
        x = self.total_conv(x)
        x = x.view(-1,5)
        return x
        
net = LocalPathConv()
net.load_state_dict(torch.load("phase1_param5.txt"))
print('loaded')
net = net.cuda(1)
print(net)
'''
x = Variable(torch.randn(1,4,33,33), requires_grad = True)
x = x.cuda(1)
y_pred = net.forward(x)
print(y_pred)
'''
#1 phase train
f = h5py.File('/home/yiqin/train.h5','r')

#2nd phase train
normal_list = open('/home/yiqin/trainval.txt','r')
train_list = []
str2 = normal_list.readlines()
for i in range(len(str2)):
    train_list.append(str2[i][0:-1].split(' '))
print(len(train_list))
normal_list.close()

def create_val_batch_phase2(batch_size):
    val_batch = []
    val_label = []
    for i in range(batch_size):
        case,x,y,z,l = train_list.pop()
        x,y,z,l = int(x), int(y), int(z), int(l)
        case1 = case[0:2]
        case2 = case[3:]
        img_patch = f[case1][case2][:, x-16:x+16+1, y-16:y+16+1, z] #sample a 33x33 patch
        val_batch.append(img_patch)
        val_label.append(l)
    val_batch = torch.from_numpy(np.array(val_batch))
    val_label = torch.from_numpy(np.array(val_label))
    return val_batch, val_label

prev_time = time.clock()
random.shuffle(train_list)
print(time.clock()-prev_time)

val_size = 1000
val_x, val_y = create_val_batch_phase2(val_size)
print(len(train_list))
print(len(val_x))


#set hyper_param
learning_rate = 5e-5
l1_reg = 5e-5
optimizer = torch.optim.SGD(net.total_conv.parameters(), lr=learning_rate, momentum = 0.9, weight_decay = 5e-5)
'''
for param in net.parameters():
    if len(param.size())==4:
        ini.uniform(param, a=-5e-3, b=5e-3)
''' 
def create_batch(size, j):
    train_batch = []
    train_label = []
    for i in range(size):
        case,x,y,z,l = train_list[j * size + i]
        x,y,z,l = int(x), int(y), int(z), int(l)
        case1 = case[0:2]
        case2 = case[3:]
        img_patch = f[case1][case2][:, x-16:x+16+1, y-16:y+16+1, z] #sample a 33x33 patch
        train_batch.append(img_patch)
        train_label.append(l)
    train_batch = torch.from_numpy(np.array(train_batch))
    train_label = torch.from_numpy(np.array(train_label))
    return train_batch, train_label

prev_time = time.clock()
num_epoch = 2
batch_size = 1024
step_size = len(train_list) // batch_size
scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
val_x = Variable(val_x).cuda(1)
val_y = val_y.cuda(1)
val_batch_size = len(val_y)
print('fuck')
print(step_size)

for i in range(num_epoch):
    random.shuffle(train_list)
    for j in range(step_size):
        x_train, y_train = create_batch(batch_size, j)
        x_train, y_train = Variable(x_train.cuda(1)), Variable(y_train.cuda(1), requires_grad=False)
        y_pred = net.forward(x_train)
        loss = F.cross_entropy(y_pred, y_train)#cross entropy loss

#        l1_crit = nn.L1Loss(size_average = False)#L1 loss
#        reg_loss = 0
#        for param in net.parameters():
#            reg_loss += l1_crit(param)
#        loss+= l1_reg * reg_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #check accuracy
        if j%10 == 0:
            print('finished:', j/num_epoch/step_size)
            print(loss)
            val_pred = net.forward(val_x)
            _, predicted = torch.max(val_pred.data, 1)
            correct = (predicted == val_y).sum()
            print('Validation accuracy:', float(correct) / val_batch_size)
            print('time used:%.3f'% (time.clock() - prev_time))
    print('train finish')
    file_name = 'phase2_param' + str(i+3) + '.txt'
    torch.save(net.state_dict(), file_name)
    print('saved')
    scheduler.step()
