#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 14:34:23 2021

@author: yunda_si
"""


import os
import paired.cluster_species as cs
import paired.rw_a2m as rw_a2m


def final_pair(TaxID_dict,topn):
    
    paired = []
    
    for TaxID in TaxID_dict:
        faA_list,faB_list = TaxID_dict[TaxID]

        faA_list = faA_list[:topn]
        faB_list = faB_list[:topn]
        
        for faA,faB in zip(faA_list,faB_list):             

            header = faA[-1][0].strip()+'||'+faB[-1][0].strip()
            seq = faA[-1][1].strip()+faB[-1][1].strip()
            paired.append([header,seq])  
        
    print(f'len paired: {len(paired)}')
    return paired[:100000]


def main(file_dict,cov,topn):


    refA = rw_a2m.read_refseq(file_dict['fastaA'])
    refB = rw_a2m.read_refseq(file_dict['fastaB'])
    
    ######################     1. read msa      ###############################
    msaA_data = rw_a2m.read_a2m(file_dict['msaA'],len(refA[-1][-1]),cov)
    print(f'1.1A sequence count: {len(msaA_data)}')

    msaA_data.pop(0)
    msaA_data = rw_a2m.parse_msa(msaA_data)
    print(f'1.1A init filter, sequence count: {len(msaA_data)}')

                   ############################################################
    msaB_data = rw_a2m.read_a2m(file_dict['msaB'],len(refB[-1][-1]),cov)
    print(f'1.1B sequence count: {len(msaB_data)}')

    msaB_data.pop(0)
    msaB_data = rw_a2m.parse_msa(msaB_data)
    print(f'1.1B init filter, sequence count: {len(msaB_data)}')


    ######################     2. common Tax     ##############################
    common_species = cs.common_Tax(msaA_data,msaB_data)
    TaxID_dict = cs.Tax_groupmsa(common_species, msaA_data, msaB_data)
    print(f'\n1.2 common tax count: {len(TaxID_dict)}')
    unpaired_num = cs.unpairedseq(TaxID_dict)
    print(f'1.2 unpaired count: {unpaired_num}')
    TaxID_dict = cs.sorted_sim(TaxID_dict,refA,refB)


    ##################        3. final paired        ##########################     
    paired = final_pair(TaxID_dict,topn)
    paired_file = os.path.join(file_dict['outpath'],'paired.a3m')
    
    with open(paired_file,'w') as f:
        header = refA[-1][0].strip()+'&'+refB[-1][0].strip()
        seq = refA[-1][1].strip()+refB[-1][1].strip()
        f.write(header)
        f.write('\n')
        f.write(seq)
        f.write('\n')
            
        for header,seq in paired:
            f.write(header)
            f.write('\n')
            f.write(seq)
            f.write('\n')

















