#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 19:32:45 2022

@author: yunda_si
"""


import os
import numpy as np
import torch
import pickle


def concat(A_f1d, B_f1d, p2d):
    
    def rep_new_axis(mat, rep_num, axis):
        return torch.repeat_interleave(torch.unsqueeze(mat,axis=axis),rep_num,axis=axis)
    
    len_channel,lenA = A_f1d.shape
    len_channel,lenB = B_f1d.shape        
    
    row_repeat = rep_new_axis(A_f1d, lenB, 2)
    col_repeat = rep_new_axis(B_f1d, lenA, 1)        

    return  torch.unsqueeze(torch.cat((row_repeat, col_repeat, p2d),axis=0),0)

    

def read_alnstats(stats_file):
    temp_pair = np.loadtxt(stats_file,dtype=np.float32)
    length = int(temp_pair.max())
    alnstats = np.zeros((3,length,length))
    for ii in range(len(temp_pair)):
        alnstats[:,int(temp_pair[ii][0])-1,int(temp_pair[ii][1])-1] = temp_pair[ii][2:]
        alnstats[:,int(temp_pair[ii][1])-1,int(temp_pair[ii][0])-1] = temp_pair[ii][2:]
    return alnstats



def graph_feature(result_path):
    
    chain_feature = []
    for chain in ['A', 'B']:
        pssm_file = os.path.join(result_path, f'{chain}_hhm.pkl')
        esm1b_repr_file = os.path.join(result_path, f'{chain}_esm1b.repr.npy')
        msa1b_repr_file = os.path.join(result_path, f'{chain}_msa1b.repr.npy')
        esmif_repr_file = os.path.join(result_path, f'{chain}_esmif.repr.npy')
        graph_file = os.path.join(result_path, f'graph{chain}.pkl')
        
        graph = pickle.load(open(graph_file,'rb'))
        PSSM = pickle.load(open(pssm_file,'rb'))['PSSM']
        esm1b_repr = np.load(esm1b_repr_file)
        msa1b_repr = np.load(msa1b_repr_file)
        esmif_repr = np.load(esmif_repr_file)
        
        feature_1d = np.hstack((graph['nodes_sact'], PSSM, esm1b_repr, msa1b_repr, esmif_repr))
        graph['nodes_scat'] = torch.from_numpy(feature_1d)
        chain_feature.append(graph)

    return chain_feature


def paired_feature(result_path):

    alnstats_file = os.path.join(result_path, 'paired.pairout')
    ccmpred_file = os.path.join(result_path, 'paired.ccmpred')
    rtattn_msa1b = os.path.join(result_path, 'msa1b_rt.attn.npy')
    swattn_msa1b = os.path.join(result_path, 'msa1b_sw.attn.npy')
    

    rtattn_msa1b = np.load(rtattn_msa1b) 
    l,h,la,lb = rtattn_msa1b.shape
    rtattn_msa1b = np.reshape(rtattn_msa1b, (l*h,la,lb))        

    ccmpred = np.loadtxt(ccmpred_file, dtype=np.float32)[:la,la:]
    ccmpred = np.expand_dims(ccmpred, 0)
    alnstats = read_alnstats(alnstats_file)[:,:la,la:]
    
    swattn_msa1b = np.load(swattn_msa1b) 
    l,h,la,lb = swattn_msa1b.shape
    swattn_msa1b = np.reshape(swattn_msa1b, (l*h,la,lb))  
    
    
    rt_feature_2d = [ccmpred, alnstats, rtattn_msa1b]
    rt_feature_2d = [torch.from_numpy(i) for i in rt_feature_2d] 
    rt_feature_2d = torch.cat(rt_feature_2d, axis=0)
    
    sw_feature_2d = [ccmpred.swapaxes(-2,-1), 
                     alnstats.swapaxes(-2,-1), swattn_msa1b]
    sw_feature_2d = [torch.from_numpy(i) for i in sw_feature_2d] 
    sw_feature_2d = torch.cat(sw_feature_2d, axis=0)
    
    return rt_feature_2d, sw_feature_2d

    
    

























    