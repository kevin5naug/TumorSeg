
# coding: utf-8

# In[20]:

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
from random import shuffle
import pickle

class InputCasNet(nn.Module):
    def __init__(self):
        super(InputCasNet, self).__init__()
        self.first_upper_layer1=nn.Sequential(
            nn.Conv2d(4,64,7),
            nn.ReLU(),
            nn.MaxPool2d((4,4),stride = 1)
        )
        self.first_upper_layer2=nn.Sequential(
            nn.Conv2d(64,64,3),
            nn.ReLU(),
            nn.MaxPool2d((2,2),stride = 1)
        )
        self.first_under_layer1=nn.Sequential(
            nn.Conv2d(4,160,13),
            nn.ReLU()
        )
        
        self.first_final_layer=nn.Conv2d(224,5,21)
        
        self.second_upper_layer1=nn.Sequential(
            nn.Conv2d(9,64,7),
            nn.ReLU(),
            nn.MaxPool2d((4,4),stride = 1)
        )
        self.second_upper_layer2=nn.Sequential(
            nn.Conv2d(64,64,3),
            nn.ReLU(),
            nn.MaxPool2d((2,2),stride = 1)
        )
        self.second_under_layer1=self.under_layer1 = nn.Sequential(
            nn.Conv2d(9,160,13),
            nn.ReLU()
        )
        self.second_final_layer = nn.Conv2d(224,5,21)
    
    def forward(self, x1):
        upper_x=self.first_upper_layer2(self.first_upper_layer1(x1))
        under_x=self.first_under_layer1(x1)
        x=torch.cat((upper_x, under_x), 1)
        x=self.first_final_layer(x)
        x2=x1[:, :, 16:48+1, 16:48+1]*1.0
        x2=torch.cat((x, x2), 1)
        upper_x2=self.second_upper_layer2(self.second_upper_layer1(x2))
        under_x2=self.second_under_layer1(x2)
        x3=torch.cat((upper_x2, under_x2), 1)
        x3=self.second_final_layer(x3)
        return x3

cas_net=InputCasNet()
x1 = Variable(torch.randn(1,4,65,65), requires_grad = True)
y_pred = cas_net.forward(x1)
print(y_pred)


# In[2]:

f = h5py.File('/home/yiqin/train.h5','r')

#get the training set for phase 1
f_in=open("/home/yiqin/training-65x65-balanced.txt", "r")
content=f_in.readlines()
data_train_phase1=[]
data_val=[]
for line in content:
    no_n_line=line[0:len(line)-1]
    item=no_n_line.split(" ")
    if item[0]!="HG/0001":
        data_train_phase1.append([item[0], item[1], item[2], item[3], item[4]])
    else:
        data_val.append([item[0], item[1], item[2], item[3], item[4]])
f_in.close()
print ("phase 1 data preparation process completed.")

#get the training set for phase 2 and the validation set
f_in=open("/home/yiqin/training-65x65-unbalanced.txt", "r")
content=f_in.readlines()
data_train_phase2=[]
for line in content:
    no_n_line=line[0:len(line)-1]
    item=no_n_line.split(" ")
    if item[0]!="HG/0001":
        data_train_phase2.append([item[0], item[1], item[2], item[3], item[4]])
    else:
        data_val.append([item[0], item[1], item[2], item[3], item[4]])
f_in.close()
print ("phase 2 data preparation process completed.")

print(len(data_train_phase1))
print(len(data_train_phase2))
print(len(data_val))


# In[3]:

def create_val_batch_phase1(step = 6000, key = 299):
    val_x1 = []
    val_label = []
    batch_size = len(data_val) // step
    for i in range(batch_size):
        case,x,y,z,l = data_val.pop(i * step + key - i)
        x,y,z,l = int(x), int(y), int(z), int(l)
        case1 = case[0:2]
        case2 = case[3:]
        x1 = f[case1][case2][:, x-32:x+32+1, y-32:y+32+1, z]
        val_x1.append(x1)
        val_label.append(l)
    val_x1 = torch.from_numpy(np.array(val_x1))
    val_label = torch.from_numpy(np.array(val_label))
    return val_x1, val_label


# In[4]:

#create valid_set
val_len = len(data_val)

val_x1, val_y = create_val_batch_phase1()
val_y = val_y.view(-1)
val_x1=Variable(val_x1.cuda(2))
print(len(data_val))
print(len(val_y))


# In[21]:

#set hyper_param
learning_rate = 5e-4
l1_reg = 5e-5
optimizer = torch.optim.SGD(cas_net.parameters(), lr=learning_rate, momentum = 0.9, weight_decay = 5e-7)


# In[22]:

for param in cas_net.parameters():
    if len(param.size())==4:
        ini.uniform(param, a=-5e-4, b=5e-4)


# In[23]:

def create_batch(size, j):
    train_x1 = []
    train_label = []
    for i in range(size):
        case,x,y,z,l = data_train_phase1[j * size + i]
        x,y,z,l = int(x), int(y), int(z), int(l)
        case1 = case[0:2]
        case2 = case[3:]
        x1 = f[case1][case2][:, x-32:x+32+1, y-32:y+32+1, z]
        train_x1.append(x1)
        train_label.append(l)
    train_x1 = torch.from_numpy(np.array(train_x1))
    train_label = torch.from_numpy(np.array(train_label))
    return train_x1, train_label


# In[24]:

cas_net.cuda(2)
prev_time = time.clock()
num_epoch = 5
batch_size = 128
step_size = len(data_train_phase1) // batch_size
num_times=num_epoch*step_size
print(num_times)
scheduler = StepLR(optimizer, step_size=1, gamma=0.1)


# In[25]:

for i in range(num_epoch):
    random.shuffle(data_train_phase1)
    for j in range(step_size):
        training_x1, training_label = create_batch(batch_size, j)
        x1_train, y_train = Variable(training_x1.cuda(2)), Variable(training_label.cuda(2), requires_grad=False)
        y_pred = cas_net.forward(x1_train)
        y_pred = y_pred.view(-1,5)
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
        index=i*step_size+j
        if index % 2000 == 0:
            print ("")
            print (str(index)+' time used %.3f' % (time.clock()-prev_time))
            print ('phase 1: '+str(float(index)/num_times*100)+"% completed")
            print (loss)
            y_val_pred=cas_net.forward(val_x1)
            y_val_pred=y_val_pred.view(-1,5)
            useless, predicted=torch.max(y_val_pred.data, 1)
            correct = (predicted == val_y.cuda(2)).sum()
            print('Validation accuracy:', float(correct)/len(val_y))
    scheduler.step()
    print ("phase1 epoch: " + str(i)+" successfully trained!")
    torch.save(cas_net.state_dict(), "/home/yiqin/phase1_input_cas_net.txt")
    print ("phase1 epoch: " + str(i)+" successfully saved!")


# In[26]:

def create_batch2(size, j):
    train_x1 = []
    train_label = []
    for i in range(size):
        case,x,y,z,l = data_train_phase2[j * size + i]
        x,y,z,l = int(x), int(y), int(z), int(l)
        case1 = case[0:2]
        case2 = case[3:]
        x1 = f[case1][case2][:, x-32:x+32+1, y-32:y+32+1, z]
        train_x1.append(x1)
        train_label.append(l)
    train_x1 = torch.from_numpy(np.array(train_x1))
    train_label = torch.from_numpy(np.array(train_label))
    return train_x1, train_label


# In[27]:

cas_net=InputCasNet()
cas_net.load_state_dict(torch.load("/home/yiqin/phase1_input_cas_net.txt"))
cas_net.cuda(2)
prev_time = time.clock()
num_epoch = 2
batch_size = 128
step_size = len(data_train_phase2) // batch_size
num_times=num_epoch*step_size
print(num_times)
learning_rate = 5e-5
l1_reg = 5e-5
optimizer = torch.optim.SGD(cas_net.second_final_layer.parameters(), lr=learning_rate, momentum = 0.9, weight_decay = 5e-7)
scheduler = StepLR(optimizer, step_size=1, gamma=0.1)


# In[ ]:

for i in range(num_epoch):
    random.shuffle(data_train_phase2)
    for j in range(step_size):
        training_x1, training_label = create_batch2(batch_size, j)
        x1_train, y_train = Variable(training_x1.cuda(2)), Variable(training_label.cuda(2), requires_grad=False)
        y_pred = cas_net.forward(x1_train)
        y_pred = y_pred.view(-1,5)
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
        index=i*step_size+j
        if index % 2000 == 0:
            print ("")
            print (str(index)+' time used %.3f' % (time.clock()-prev_time))
            print ('phase 2: '+str(float(index)/num_times*100)+"% completed")
            print (loss)
            y_val_pred=cas_net.forward(val_x1)
            y_val_pred=y_val_pred.view(-1,5)
            useless, predicted=torch.max(y_val_pred.data, 1)
            correct = (predicted == val_y.cuda(2)).sum()
            print('Validation accuracy:', float(correct)/len(val_y))
    scheduler.step()
    print ("phase2 epoch: " + str(i)+" successfully trained!")
    torch.save(cas_net.state_dict(), "/home/yiqin/phase2_input_cas_net.txt")
    print ("phase2 epoch: " + str(i)+" successfully saved!")


# In[ ]:



