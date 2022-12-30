#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 15:42:08 2022

@author: yunda_si
"""

import os

hhfilter = '/home/Common_softwares/hh-suite/build/bin/hhfilter'
targets_path = '/mnt/data/yunda_si/self/data/baker_dataset/'

count = 0
for index_target,target in enumerate(os.listdir(targets_path)):
    name,chainA,_,chainB = target.split('_')
    target_path = os.path.join(targets_path,target,'paired_xu')
    print(f'{index_target:4d} {target}')
    
    msa_file = os.path.join(target_path,'paired.a3m')
    hhfilter_msa_file = os.path.join(target_path,'paired_hhfilter_256.a3m')

    if not os.path.exists(hhfilter_msa_file):
        os.system(f'{hhfilter} -i {msa_file} -o {hhfilter_msa_file} -diff 256 -maxseq 10000000')