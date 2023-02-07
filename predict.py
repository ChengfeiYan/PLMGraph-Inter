#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 15:37:51 2022

@author: yunda_si
"""

import os
import sys
import paired.pair_msa as pair_msa
import plm.msa1b_attn as msa1b_attn
import plm.esm1b_repr as esm1b_repr
import plm.msa1b_repr as msa1b_repr
import plm.esmif_repr as esmif_repr
import load_feature
import torch
from model import resnet18
import numpy as np
import pdb_graph


#### path

CCMPred = '/home/yunda_si/self/software_p/CCMpred_pad/bin/ccmpred'
reformat = '/home/Common_softwares/DeepMSA/bin/fasta2aln'
alnstats = '/home/yunda_si/self/software_p/metapsicov-2.0.3/bin/alnstats'
hhmake = '/home/yunda_si/self/software_p/hh-suite/build/bin/hhmake'
hhfilter = '/home/Common_softwares/hh-suite/build/bin/hhfilter'
LoadHHM = '/mnt/data/yunda_si/self/PythonProjects/PPI_contact/github/plm/LoadHHM.py'

# Location of model weights 
esm1b_location = "/mnt/data/yunda_si/self/software_p/esm/model/esm1b_t33_650M_UR50S.pt"
esm_msa1b_location = "/mnt/data/yunda_si/self/software_p/esm/model/esm_msa1b_t12_100M_UR50S.pt"
esm_if1_location = '/home/yunda_si/yunda_si/self/software_p/esm/model/esm_if1_gvp4_t16_142M_UR50.pt'


################################     main      ################################

fasA, a3mA, pdbA, fasB, a3mB, pdbB, result_path, device = sys.argv[1:]
device = torch.device(device)

if not os.path.exists(result_path):
    os.mkdir(result_path)


#### prepare feature

## prepare paired feature

# 1. paired MSA
paired_a3m = os.path.join(result_path, 'paired.a3m')
file_dict = {'fastaA':fasA,
              'fastaB':fasB,
              'msaA':a3mA,
              'msaB':a3mB,
              'outpath':result_path}

pair_msa.main(file_dict, 0.5, 100000)



# 2. reformat MSA
filter_paired_a3m = os.path.join(result_path, 'filtered_paired.a3m')
os.system(f'{hhfilter} -i {paired_a3m} -o {filter_paired_a3m} -diff 256')

paired_aln = os.path.join(result_path, 'paired.aln')
os.system(f'{reformat} {paired_a3m} {paired_aln}')  

filter_a3mA = os.path.join(result_path, 'filteredA.a3m')
os.system(f'{hhfilter} -i {a3mA} -o {filter_a3mA} -diff 256')

filter_a3mB = os.path.join(result_path, 'filteredB.a3m')
os.system(f'{hhfilter} -i {a3mB} -o {filter_a3mB} -diff 256')



# 3. paired seq
paired_seq = os.path.join(result_path, 'paired.fasta')
seqA = open(fasA).readlines()[-1].strip()
seqB = open(fasB).readlines()[-1].strip()

with open(paired_seq, 'w') as f:
    f.write('>paired\n')
    f.write(seqA+seqB)



# 4. cal ccmpred & alnstats
paired_ccmpred = os.path.join(result_path, 'paired.ccmpred')
os.system(f'{CCMPred} -R {paired_aln} {paired_ccmpred}') 

alnstats_sing = os.path.join(result_path, 'paired.singout')
alnstats_pair = os.path.join(result_path, 'paired.pairout')
os.system(f'{alnstats} {paired_aln} {alnstats_sing} {alnstats_pair}')



# 6. cal msa-1b attention
rtattn_msa1b = os.path.join(result_path, 'msa1b_rt.attn')
swattn_msa1b = os.path.join(result_path, 'msa1b_sw.attn')
msa1b_attn.main(esm_msa1b_location, filter_paired_a3m, 
                fasA, rtattn_msa1b, swattn_msa1b, device)



## prepare 1D feature

# 7. cal PSSM
hhmA = os.path.join(result_path, 'A.hhm')
pssmA = os.path.join(result_path, 'A_hhm.pkl')
hhmB = os.path.join(result_path, 'B.hhm')
pssmB = os.path.join(result_path, 'B_hhm.pkl')

os.system(f'{hhmake} -i {a3mA} -o {hhmA}')
os.system(f'python {LoadHHM} {hhmA}')
os.system(f'{hhmake} -i {a3mB} -o {hhmB}')
os.system(f'python {LoadHHM} {hhmB}')


# 8. esm-1b repr
reprA_esm1b = os.path.join(result_path, 'A_esm1b.repr')
reprB_esm1b = os.path.join(result_path, 'B_esm1b.repr')

esm1b_repr.main(esm1b_location, fasA, reprA_esm1b, device)
esm1b_repr.main(esm1b_location, fasB, reprB_esm1b, device)



# 9. msa-1b repr
reprA_msa1b = os.path.join(result_path, 'A_msa1b.repr')
reprB_msa1b = os.path.join(result_path, 'B_msa1b.repr')

msa1b_repr.main(esm_msa1b_location, filter_a3mA, reprA_msa1b, device)
msa1b_repr.main(esm_msa1b_location, filter_a3mB, reprB_msa1b, device)



# 10. esm-if1 repr
reprA_esmif = os.path.join(result_path, 'A_esmif.repr')
reprB_esmif = os.path.join(result_path, 'B_esmif.repr')

esmif_repr.main(esm_if1_location, pdbA, reprA_esmif, device)
esmif_repr.main(esm_if1_location, pdbB, reprB_esmif, device)



## prepare graph feature

# 11. cal graph
graphA = os.path.join(result_path, 'graphA.pkl')
graphB = os.path.join(result_path, 'graphB.pkl')

pdb_graph.main(pdbA, graphA)
pdb_graph.main(pdbB, graphB)



#### load feature
featureA, featureB = load_feature.graph_feature(result_path)
rt_p2d, sw_p2d = load_feature.paired_feature(result_path)

nodeA = (featureA['nodes_scat'].to(device), featureA['nodes_vec'].to(device))
edgeA = (featureA['edge_scat'].to(device), featureA['edge_vec'].to(device))
edge_indexA = featureA['edge_index'].to(device)

nodeB = (featureB['nodes_scat'].to(device), featureB['nodes_vec'].to(device))
edgeB = (featureB['edge_scat'].to(device), featureB['edge_vec'].to(device))
edge_indexB = featureB['edge_index'].to(device)
    
rt_p2d = rt_p2d.to(device).float()
sw_p2d = sw_p2d.to(device).float()


#### model
weight_list = [f'model/{i}' for i in range(1,8)]
model = resnet18() 
torch.set_grad_enabled(False)

_,lenx,leny = rt_p2d.shape
all_preds = torch.zeros(lenx,leny)

for weight_file in weight_list:
    model.load_state_dict(torch.load(weight_file, map_location=device))
    model.to(device)
    model.eval()
    
    pred = model(nodeA, edgeA, edge_indexA, 
                 nodeB, edgeB, edge_indexB, 
                 rt_p2d).detach().cpu()
 
    all_preds += pred

    pred = model(nodeB, edgeB, edge_indexB, 
                  nodeA, edgeA, edge_indexA, 
                  sw_p2d).detach().cpu()
 
    all_preds += pred.T

all_preds = all_preds.numpy()
np.savetxt(os.path.join(result_path,'pred.txt'), all_preds)
