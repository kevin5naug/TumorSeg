
# coding: utf-8

# In[14]:

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle as pkl

class TwoPathConv(nn.Module):
    def __init__(self):
        super(TwoPathConv, self).__init__()
        self.upper_layer1 = nn.Sequential(
            nn.Conv2d(4,64,7, padding = 16, ),
            nn.ReLU(),
            nn.MaxPool2d((4,4),stride = 1)
        )
        self.upper_layer2 = nn.Sequential(
            nn.Conv2d(64,64,3),
            nn.ReLU(),
            nn.MaxPool2d((2,2),stride = 1)
        )
        self.under_layer1 = nn.Sequential(
            nn.Conv2d(4,160,13, padding = 16),
            nn.ReLU()
        )
        self.final_layer = nn.Conv2d(224,5,21)
        
    def forward(self, x):
        upper_x = self.upper_layer2(self.upper_layer1(x))
        under_x = self.under_layer1(x)
        #upper_x = F.max_pool2d(F.relu(self.upper_conv1(x)), (4, 4),stride = 1)
        #upper_x = F.max_pool2d(F.relu(self.upper_conv2(upper_x)), (2, 2), stride = 1)
        #under_x = F.relu(self.under_conv1(x))
        final_x = torch.cat((under_x, upper_x), 1)
        out = self.final_layer(final_x)
        #out = out.view(-1,5)
        return out
        
net = TwoPathConv()
print(net)

#x = Variable(torch.randn(1,4,33,33), requires_grad = True)
#y_pred = net.forward(x)
#print(type(y_pred))


import h5py
#challenge_f = h5py.File('Challenge.h5', 'r') #load challenge data
train_f = h5py.File('training.h5', 'r')
SAMPLE = [ "LG/0001", "LG/0002", "LG/0004", "LG/0006", "LG/0008", "LG/0011",
          "LG/0012", "LG/0013", "LG/0014", "LG/0015", "HG/0001", "HG/0002",
          "HG/0003", "HG/0004", "HG/0005", "HG/0006", "HG/0007", "HG/0008",
          "HG/0009", "HG/0010", "HG/0011", "HG/0012", "HG/0013", "HG/0014",
          "HG/0015", "HG/0022", "HG/0024", "HG/0025", "HG/0026", "HG/0027",]

#for i in enumerate(SAMPLE):
#    index, case = i
#    case1 = case[:2]
#    case2 = case[3:]
#    print(challenge_f[case1][case2].shape)
    
def create_test_batch(img = 0, x = 16, z= 0):
    case = SAMPLE[img]
    case1 = case[:2]
    case2 = case[3:]
    batch = []
    _, X, Y, Z = train_f[case1][case2].shape
    for y in range(16, Y - 17):
        batch.append(train_f[case1][case2][:, x-16:x+17, y-16:y+17, z])
    batch = torch.from_numpy(np.array(batch))
    return batch


# In[ ]:

import time
import numpy as np
from torch.autograd import Variable
net = TwoPathConv()
net.cuda(3)
net.load_state_dict(torch.load('premature_net_phase2_without_biasreg.txt'))

prev_time = time.clock()
#print(matrix_pred)
s = 0
#matrix_pred = {}
pred = {}
for img in range(30):
    case = SAMPLE[img]
    case1 = case[:2]
    case2 = case[3:]
    #_, X, Y, Z = challenge_f[case1][case2].shape
    _, X, Y, Z = train_f[case1][case2].shape
    print(X, Y, Z)
    pred[img] = []
    for z in range(Z):
        X_batch = torch.from_numpy(train_f[case1][case2][:,:,:,z].reshape(1,4,X,Y))
        print(X_batch.size())
        X_batch = Variable(X_batch.cuda(3))
        y_pred = net.forward(X_batch)
        y_pred = y_pred.data.cpu().numpy()
        #print(y_pred.shape)
        if (s%10 == 0):
            print ('Ongoing ...', (img, z))
            print ('time used %.3f' % (time.clock()-prev_time))
        pred[img].append(np.reshape(y_pred.argmax(axis = 1), (X,Y)))
    print(np.array(pred[img]).shape)
        
output1 = open('train_slide_pred.pkl', 'wb')
pkl.dump(pred, output1, protocol = 2)