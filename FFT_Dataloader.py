import random
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score, confusion_matrix
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import os
from os import listdir

def new_list(old_list, classes = 'ALL'):
    labels = []
    new_names = []
    for i in old_list:
        s = i.split("_")
        if s[1] == "BD1" or s[1] == "BD2" or s[1] == "BDSP":
            dataclass = 2       
        elif s[1] == "MDD":
            dataclass = 1
        elif s[1] == "NC":
            dataclass = 0
        if classes != 'ALL':
            if dataclass not in classes:
                continue
  
        labels.append(dataclass)
        new_names.append(i)
    return np.array(new_names), np.array(labels)

def check_exist( datalist, datapath = '/home/bml_dl/CCHuang/Data/MEG_All_2s_500Hz/',length = 3, sparse = 0.5, classes = 'ALL',time_start = None, time_stop = None):
    new_list = []
    used = int(length * sparse)
    for j,i in enumerate(datalist):
       
        s = i.split("_")
        num = int(s[8].rstrip('.npy'))
        if num % used != 0:
            continue
        s[8] = str(num + length -1) + '.npy'
        new_name = '_'.join(s)
        print(new_name)
        new_path = datapath + new_name
        if time_start != None:
            if  num < time_start:
                continue
        if time_stop != None:
            if (num+length+1) > time_stop:
                continue
        if classes != 'ALL':
            if s[1] == "BD1" or s[1] == "BD2" or s[1] == "BDSP":
                    dataclass = 2       
            elif s[1] == "MDD":
                dataclass = 1
            elif s[1] == "NC":
                dataclass = 0
            if dataclass not in classes:
                continue
        if os.path.isfile(new_path):
            new_list.append(i)
    return np.array(new_list)

def check_exist2(datapath, datalist,length = 3, sparse = 0.5, classes = 'ALL',time_start = None, time_stop = None):
    new_list = []
    used = int(length * sparse)
    for j,i in enumerate(datalist):
        
        s = i.split("_")
        num = int(s[8])
        if time_start != None:
            if  num < time_start:
                continue
        if time_stop != None:
            if (num+length+1) > time_stop:
                continue
        count_num = num-time_start
        if count_num % used != 0:
            continue
        s[8] = str(num + length -1)
        new_name = '_'.join(s)
        new_path = datapath + new_name
        
        if classes != 'ALL':
            
            if s[1] == "BD1" or s[1] == "BD2" or s[1] == "BDSP":
                    dataclass = 2       
            elif s[1] == "MDD":
                dataclass = 1
            elif s[1] == "NC":
                dataclass = 0
            if dataclass not in classes:
                continue
        
        if os.path.isfile(new_path):
            new_list.append(i)
        
    return np.array(new_list)

def check_exist3(datapath, subject_list,length = 10, sparse = 0.5, classes = 'ALL',time_start = None, time_stop = None, datatype = 'Raw'):
    new_list = []
    used = int(length * sparse)
    for j,i in enumerate(subject_list):
        for num in range(time_start, time_stop-length):
            s = i.split("_")
            if datatype == 'Raw':
                s.append(str(num)+'.npy')
            elif datatype == 'FFT':
                s.append(str(num)+'_FFT.npy')
            count_num = num - time_start
            if count_num % used != 0:
                continue
            new_name = '_'.join(s)
            new_path = datapath + new_name

            if classes != 'ALL':

                if s[1] == "BD1" or s[1] == "BD2" or s[1] == "BDSP":
                        dataclass = 2       
                elif s[1] == "MDD":
                    dataclass = 1
                elif s[1] == "NC":
                    dataclass = 0
                if dataclass not in classes:
                    continue

            if os.path.isfile(new_path):
                new_list.append(new_name)
    return np.array(new_list)
    
class Subject(Dataset):
    def __init__(self, path, subject_list, batch=32, sparse = 0.5 , length = 1, device = 'cuda:0', classes = 'ALL', datatype = 'FFT',
                 time_start = None, time_stop = None):
        self.path = path
        self.batch_size = batch
        self.length = length
        self.datatype = datatype
        self.time_start = time_start
        self.time_stop = time_stop
        self.sparse = sparse

        self.datalist = check_exist3(datapath = path,  subject_list =  subject_list, length = length, sparse = sparse, classes = classes, time_start = self.time_start, time_stop = self.time_stop, datatype = datatype)
        np.random.shuffle(self.datalist)
        print(self.datalist.shape)
        self.device = device
    def __len__(self):
        return (len(self.datalist)-1)//self.batch_size +1
    def __getitem__(self, idx):
        start = True
#         full_X = np.zeros([self.batch_size,self.length,3, 102, 512])
#         full_num = np.zeros([self.batch_size])
#         full_IDs = np.zeros([self.batch_size])
        for i in range(idx*self.batch_size, (idx+1)*self.batch_size):
            if i >= len(self.datalist):
                break
            
            iter_num = i - idx*self.batch_size
            dataname = self.datalist[i]#data_list[i]
            s = dataname.split('_')
            name = '_'.join(s[0:8])
        
#             domain = self.domain_IDs[np.where(self.domain_list == "_".join(s[0:-2]))]
            
            #print(s[4])
            if s[1] == "BD1" or s[1] == "BD2" or s[1] == "BDSP":
                dataclass = 2       
            elif s[1] == "MDD":
                dataclass = 1
            elif s[1] == "NC":
                dataclass = 0
            datapath = self.path + dataname
            full_data = torch.FloatTensor(np.load(datapath))
#             full_data = (full_data - full_data.mean(axis = 1).unsqueeze(1) ) / torch.max(torch.abs(full_data), dim = 1, keepdim = True)[0] #[-1, 1]
#             full_data = (full_data - full_data.mean(axis = 1).unsqueeze(1) ) / full_data.std(axis = 1).unsqueeze(1)
#             full_data = (full_data - full_data.mean(axis = 1).unsqueeze(1) ) / full_data.std(axis = 1).unsqueeze(1) #z
#             full_data = (full_data - torch.min(full_data))/ (torch.max(full_data)- torch.min(full_data)) #min max full signal
            for coil in range(len(full_data)):
                full_data[coil] = (full_data[coil] - torch.min(full_data[coil]))/ (torch.max(full_data[coil])- torch.min(full_data[coil])) #min max along coil 
#             print(torch.max(full_data),torch.min(full_data))
            X = torch.unsqueeze(full_data, 0)
            if self.datatype == 'FFT':
                num = int(s[8])
            elif self.datatype == 'Raw':
                num = int(s[8][0:-4])
            if self.length != 1:
                for j in range(1,self.length):
                    if self.datatype == 'FFT':
                        s[8] = str(num+j)
                    elif self.datatype == 'Raw':
                        s[8] = str(num) + '.npy'
                    new_name = '_'.join(s)
                    datapath = self.path + new_name
                    if not os.path.isfile(datapath):
                        print('error', j, datapath)
                        continue
                    full_data = torch.FloatTensor(np.load(datapath))
#                     full_data = (full_data - full_data.mean(axis = 1).unsqueeze(1) ) / torch.max(torch.abs(full_data), dim = 1, keepdim = True)[0]
#                     full_data = (full_data - full_data.mean(axis = 1).unsqueeze(1) ) / full_data.std(axis = 1).unsqueeze(1) #z
                    for coil in range(len(full_data)):
                        full_data[coil] = (full_data[coil] - torch.min(full_data[coil]))/ (torch.max(full_data[coil])- torch.min(full_data[coil])) #min max along coil 
#                     print(torch.max(full_data), torch.min(full_data))
                    full_data = torch.unsqueeze(full_data, 0)
                    X = torch.cat((X, full_data))
            if torch.isnan(X).any():
                continue
            Y = torch.LongTensor(np.array([dataclass]))
#             full_num[iter_num] = int((num - self.time_start)//int(self.sparse * self.length))
#             full_IDs[iter_num] = domain
            if start:
                full_X = torch.unsqueeze(X, 0)
                full_Y = Y
                full_num = np.array([int((num - self.time_start)//int(self.sparse * self.length))])
                full_Names = np.array([name])
                start = False
            else:
                full_X = torch.cat((full_X, torch.unsqueeze(X, 0)),0)
                full_Y = torch.cat((full_Y,Y),0)
                full_num = np.concatenate((full_num,[int((num - self.time_start)//int(self.sparse * self.length))]))
                full_Names = np.concatenate((full_Names, [name]))
#             if int((num - self.time_start)//int(self.sparse * self.length)) > int((self.time_stop - self.time_start - self.length)//(self.length*self.sparse)):
#                 print(int((num - self.time_start)//int(self.sparse * self.length)), dataname)
        return full_X.to(self.device), full_Y.to(self.device), full_num, full_Names#, full_vector, dataname
