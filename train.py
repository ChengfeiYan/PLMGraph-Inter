#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 21:44:20 2020

@author: yunda_si
"""

from ResNetB import resnet18
import torch
import torch.optim as optim
import random
import pickle
import os
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn as nn

def top_statistics_ppi(pred_map,contact_map,Topk_list):

    count = 0
    single_statictics = np.ones((len(Topk_list)))
    len_x = len(torch.where(torch.sum(contact_map,dim=0)!=-contact_map.shape[0])[0])
    len_y = len(torch.where(torch.sum(contact_map,dim=1)!=-contact_map.shape[1])[0])
    L = min(len_x,len_y)

    Label = contact_map.reshape(-1)
    pred_map[contact_map==-1] = -1
    
    for Topk in Topk_list:
        if isinstance(Topk,str):
            Topk = int(L/int(Topk[Topk.index('/')+1:]))
            if Topk<1:
                Topk = 1
                
        SortedP = torch.topk(pred_map.reshape(-1),Topk,largest=True, sorted=True)[1]
        TTL = torch.sum(Label[SortedP]).item()
        
        single_statictics[count] = TTL/Topk
        count = count+1

    return single_statictics


class ppi_loss(nn.Module):


    def __init__(self, alpha=None,  inter=24, clamp=False, reduction='sum'):
        super(ppi_loss, self).__init__()
        self.alpha = alpha
        self.inter = inter
        self.clamp = clamp        
        self.reduction = reduction
        if isinstance(alpha,(float,int)):
            self.alpha = torch.tensor(alpha,dtype=torch.float32,requires_grad=True)


    def forward(self, Input, Label, mask):

        device = Label.device
        Constant = torch.tensor(1.0,dtype=torch.float32,requires_grad=True).to(device)

        if self.alpha is not None:
            self.alpha = self.alpha.to(device)#
            weight1 = self.alpha*(2-Input)**2
            weight2 = (Constant-self.alpha)*(1+Input)**3
            loss = -Label*torch.log(Input)*weight1 - (Constant-Label)*torch.log(Constant-Input)*weight2
        else:
            loss = -Label*torch.log(Input) - (Constant-Label)*torch.log(Constant-Input)

        loss = mask*loss

        if self.reduction == 'sum':
            loss = torch.sum(loss)
        elif self.reduction == 'mean':
            loss = torch.mean(loss)

        return loss
    

random.seed(42)
###################              load dataset               ###################
homo_path = '/mnt/data/yunda_si/self/data/Homodataset/train_paired'
hetero_path = '/mnt/data/yunda_si/self/data/Heterodataset/train_paired/'
train_all_path = '/mnt/data/yunda_si/self/data/train_ppi/'

hetero_lists = sorted(os.listdir(hetero_path))
homo_lists = sorted(os.listdir(homo_path))
all_lists = sorted(os.listdir(train_all_path))

for i in range(10):
    random.shuffle(hetero_lists)
    random.shuffle(homo_lists)    
    random.shuffle(all_lists)


valid_list = all_lists[:1050]
train_list = all_lists[1050:]

trainset = Dataset(train_all_path, train_list)
train_loader = DataLoader(trainset, shuffle=True, num_workers=6, prefetch_factor=3, 
                          batch_size=1, persistent_workers=True)

validset = Dataset(train_all_path, valid_list)
valid_loader = DataLoader(validset, shuffle=True, num_workers=6, prefetch_factor=3,
                          batch_size=1, persistent_workers=True)

max_aa = 400

###################               import net                ###################
device = torch.device("cuda:1")
print(device)
model = resnet18().to(device)

criterion_ppi = ppi_loss(alpha=None, reduction='sum')
optimizer = optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999),
                        weight_decay=0.1)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                        eps=1e-6, patience=3, factor=0.1, verbose=True)

epoch_num = 100


###################             top statistics              ###################
topk_ppi = ['L/5','L/10','L/20',50,20,10,5,1]
dict_statics = {'min_loss':np.inf,'valid_loss':[]}

for key in topk_ppi:
    dict_statics[key] = {'highest':0,'save':'','train_acc':[],'valid_acc':[]}


###################               save model                ###################
if os.path.exists('final_model'):
    pass
else:
    os.mkdir('final_model')
epoch_target = {}

savepth = './final_model/GCN1_5'

for key in topk_ppi:
    dict_statics[key]['save'] = '{0}_{1}.pth'.format(savepth, str(key).replace('/','_'))
loss_save = f'{savepth}_minloss.pth'


###################                training                 ###################
for epoch in range(epoch_num):

    for phase in ['train', 'valid']:
        print('\n')
        if phase == 'train':
            model.train()
            dataloader = train_loader
        else:
            model.eval()
            dataloader = valid_loader
            
        acc_all = np.zeros((0,len(topk_ppi)))
        acc_batch = np.zeros((0,len(topk_ppi)))
        running_loss = 0.0
        optimizer.zero_grad()

        for d, (target, proteinA, proteinB, p2d, mask_map, contact_map) in enumerate(dataloader):
            
            proteinA = {key:item.to(device) for key,item in proteinA.items() if type(item)==torch.Tensor}
            proteinB = {key:item.to(device) for key,item in proteinB.items() if type(item)==torch.Tensor}

            nodeA = (proteinA['nodes_scat'][0,:], proteinA['nodes_vec'][0,:])
            edgeA = (proteinA['edge_scat'][0,:], proteinA['edge_vec'][0,:])
            edge_indexA = proteinA['edge_index'][0,:]

            nodeB = (proteinB['nodes_scat'][0,:], proteinB['nodes_vec'][0,:])
            edgeB = (proteinB['edge_scat'][0,:], proteinB['edge_vec'][0,:])
            edge_indexB = proteinB['edge_index'][0,:]

            p2d = torch.cat([i.to(device) for i in p2d],axis=1).squeeze().float()
            mask_map = mask_map.squeeze().to(device).float()
            contact_map = contact_map.squeeze().to(device).float()
            

            la,lb = contact_map.shape
            starta = 0 if la<=max_aa else np.random.randint(0,la-max_aa+1)
            startb = 0 if lb<=max_aa else np.random.randint(0,lb-max_aa+1)   

            mask_map = mask_map[starta:(starta+max_aa), startb:(startb+max_aa)]
            contact_map = contact_map[starta:(starta+max_aa), startb:(startb+max_aa)]

            
            with torch.set_grad_enabled(phase == 'train'):
                preds = model(nodeA, edgeA, edge_indexA,
                              nodeB, edgeB, edge_indexB, 
                              p2d, starta, startb, max_aa)
                loss = criterion_ppi(preds, contact_map, mask_map)
                
                if phase == 'train':

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                          
            running_loss += loss.item()
            
            ##################          statistics           ##################
            accuracy = top_statistics_ppi(preds,contact_map,topk_ppi)
            acc_all = np.vstack([acc_all,accuracy])
            acc_batch = np.vstack([acc_batch,accuracy])
            
            
            if (d+1)%100==0:
                mean_acc = np.mean(acc_all[-100:], 0)*100
                print(f'[{epoch:3d}, {d+1:4d}]  loss:{running_loss:11.2f} {"  ".join([f"{i.item():7.3f}" for i in mean_acc])}')
                batch_loss = 0
                acc_batch = np.zeros((0,len(topk_ppi)))
            if (d+1)==len(dataloader):
                mean_acc = np.mean(acc_all,0)*100
                print(f'[{epoch:3d}, {d+1:4d}]  loss:{running_loss:11.2f} {"  ".join([f"{i.item():7.3f}" for i in mean_acc])}')


        if phase == 'valid':
            scheduler.step(running_loss)
            dict_statics['valid_loss'].append(running_loss)
            for index,key in enumerate(topk_ppi):
                dict_statics[key]['valid_acc'].append(mean_acc[index])
        else:
            for index,key in enumerate(topk_ppi):
                dict_statics[key]['train_acc'].append(mean_acc[index])            
            
    ##################                 save                  ##################
    for key in topk_ppi:
        acc = dict_statics[key]['valid_acc'][-1]
        highest = dict_statics[key]['highest']
        if acc>highest:
            print(f'save_{str(key):5s}:{acc:6.3f}  highest: {highest:6.3f}  delta:{acc-highest:6.3f}')
            dict_statics[key]['highest'] = acc

            if os.path.exists(dict_statics[key]['save']):
                os.remove(dict_statics[key]['save'])
            torch.save(model.state_dict(), dict_statics[key]['save'])

    if running_loss<dict_statics['min_loss']:
        print('save_minloss:%11.2f    %11.2f'%(running_loss,dict_statics['min_loss']))
        dict_statics['min_loss'] = running_loss
        torch.save(model.state_dict(), loss_save)
    
    torch.save(model.state_dict(), savepth+f'_{epoch}.pkl')












