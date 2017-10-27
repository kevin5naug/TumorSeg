
# coding: utf-8

# In[14]:

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

class TwoPathConv(nn.Module):
    def __init__(self):
        super(TwoPathConv, self).__init__()
        self.local_conv1 = nn.Conv2d(4, 64, 7)
        #self.dp1 = nn.Dropout2d(.3)
        self.local_conv2 = nn.Conv2d(64, 64, 3)
        #self.dp2 = nn.Dropout2d(.3)
        self.local_conv3 = nn.Conv2d(4, 160, 13)
        #self.dp3 = nn.Dropout2d(.3)
        self.total_conv = nn.Conv2d(224, 5, 21)

    def forward(self, x):
        under_x = F.relu(self.local_conv3(x))
        x = self.local_conv1(x)
        #x = self.dp1(x)
        x = F.max_pool2d(F.relu(x), 4, stride = 1)
        x = self.local_conv2(x)
        #x = self.dp2(x)
        x = F.max_pool2d(F.relu(x), 2, stride = 1)
        x = torch.cat((x, under_x), 1)
        x = self.total_conv(x)
        x = x.view(-1,5)
        return x
    
import h5py
#challenge_f = h5py.File('Challenge.h5', 'r') #load challenge data
'''
train_f = h5py.File('train.h5', 'r')
SAMPLE = [ "LG/0001", "LG/0002", "LG/0004", "LG/0006", "LG/0008", "LG/0011",
          "LG/0012", "LG/0013", "LG/0014", "LG/0015", "HG/0001", "HG/0002",
          "HG/0003", "HG/0004", "HG/0005", "HG/0006", "HG/0007", "HG/0008",
          "HG/0009", "HG/0010", "HG/0011", "HG/0012", "HG/0013", "HG/0014",
          "HG/0015", "HG/0022", "HG/0024", "HG/0025", "HG/0026", "HG/0027",]
'''

train_f = h5py.File('Challenge.h5', 'r')
SAMPLE = [ "HG/0301", "HG/0302",
          "HG/0303", "HG/0304", "HG/0305", "HG/0306", "HG/0307", "HG/0308",
          "HG/0309", "HG/0310", ]

def create_test_batch(img = 0, x = 16, z= 0):
    case = SAMPLE[img]
    case1 = case[:2]
    case2 = case[3:]
    batch = []
    _, X, Y, Z = train_f[case1][case2].shape
    img1 = train_f[case1][case2][:,:,:,z]
    img1 = np.pad(img1, pad_width = ((0,0), (16,16), (16,16)), mode = 'constant')
    for y in range(16, Y + 16):
        batch.append(img1[:, x-16:x+17, y-16:y+17])
    batch = torch.from_numpy(np.array(batch))
    return batch

import time
import numpy as np
from torch.autograd import Variable
net = TwoPathConv()
#net = LocalPathConv()
#net = InputCasNet()
net.load_state_dict(torch.load('two_path_cnn/p2_TPWconv_net1.txt'))
#net.load_state_dict(torch.load('local_path_cnn/phase2_param4.txt'))
#net.load_state_dict(torch.load('phase1_input_cas_net.txt'))
net.cuda(1)

prev_time = time.clock()
#print(matrix_pred)
s = 0
#matrix_pred = {}
pred = {}
matrix_pred ={}

for img in range(10):
    case = SAMPLE[img]
    case1 = case[:2]
    case2 = case[3:]
    #_, X, Y, Z = challenge_f[case1][case2].shape
    _, X, Y, Z = train_f[case1][case2].shape
    print(X, Y, Z)
    matrix_pred[img] = []
    for x in range(16, X + 16):
    #for x in range(32, X + 32):
        pred[(img,x)] = []
        for z in range(Z):
            s += 1
            X_batch = create_test_batch(img = img, x = x, z = z)
            #X_batch = create_test_batch_cas(img = img, x = x, z = z)
            X_batch = Variable(X_batch.cuda(1))
            y_pred = net.forward(X_batch)
            y_pred = y_pred.data.cpu().numpy()
            if (s%100 == 0):
                print ('Ongoing ...' ,(img, x, z))
                print ('time used %.3f' % (time.clock()-prev_time))
            pred[(img,x)].append(y_pred.argmax(axis = 1)) 
        matrix_pred[img].append(pred[(img,x)])
    matrix_pred[img] = np.array(matrix_pred[img], dtype = 'int16').transpose(1,2,0)
    print(matrix_pred[img].shape)
#array_img = np.array(matrix_pred[10]).transpose(1,2,0)
#print(array_img.shape)
import pickle as pkl
output1 = open('Challenge.pkl', 'wb')
pkl.dump(matrix_pred, output1, protocol = 2)
