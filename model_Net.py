#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import torch.nn as nn


# In[ ]:


import numpy as np


# In[ ]:


from torchvision.transforms import functional as func


# In[16]:


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 7, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(7, 10, 3)
        self.fc1 = nn.Linear(38440, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# In[ ]:


class CustomDatasetFromImages(Dataset):
    def __init__(self, img_data, cl_data):
        self.image_arr = img_data
        self.cl_arr = cl_data - 1
        self.data_len = len(self.image_arr)
    def __getitem__(self, index):
        img = np.asarray(self.image_arr[index])
        img = torch.as_tensor(img)/255
        img = img.unsqueeze(0)
        
        img = func.resize(img, size=[256, 256])
        
        cl = np.asarray(self.cl_arr[index])
        cl = torch.as_tensor(cl)
        
        return (img.float(), cl.type(torch.LongTensor))

    def __len__(self):
        return self.data_len


# In[ ]:





# In[ ]:


LEARNING_RATE = 0.001
CRITERION = nn.CrossEntropyLoss()
CLASS_NAMES = {1: 'palm', 2: 'l', 3: 'fist', 4: 'fist_moved', 5: 'thumb',
               6: 'index', 7: 'ok', 8: 'palm_moved', 9: 'c', 10: 'down'}

