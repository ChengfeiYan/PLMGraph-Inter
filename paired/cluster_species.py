#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 12:06:13 2021

@author: yunda_si
"""

import paired.rw_a2m as rw_a2m
import numpy as np


def cal_similarity(square_mat,np_fasta):

    for i in range(len(square_mat)):
        idendity = np.sum(np_fasta[i]==np_fasta[-1])
        square_mat[i] = idendity
        
    return square_mat


def Tax_groupmsa(common_species, parsed_msaA, parsed_msaB):

    TaxID_dict = {}
    for TaxID in common_species:
        TaxID_dict[TaxID] = [[],[]]

    for fasta in parsed_msaA:
        TaxID = fasta[0][4]
        if TaxID in common_species:
            TaxID_dict[TaxID][0].append(fasta)

    for fasta in parsed_msaB:

        TaxID = fasta[0][4]
        if TaxID in common_species:
            TaxID_dict[TaxID][1].append(fasta)

    return TaxID_dict



def common_Tax(parsed_msaA,parsed_msaB):

    TaxID_msaA = set()
    for fasta in parsed_msaA:
        TaxID = fasta[0][4]
        TaxID_msaA.add(TaxID)
    TaxID_msaA.discard(-1)

    TaxID_msaB = set()
    for fasta in parsed_msaB:
        TaxID = fasta[0][4]
        TaxID_msaB.add(TaxID)
    TaxID_msaB.discard(-1)

    common_species = TaxID_msaA & TaxID_msaB

    return common_species



def unpairedseq(TaxID_dict):

    count = 0
    for TaxID in TaxID_dict:
        faA_list,faB_list = TaxID_dict[TaxID]
        faA_num = len(faA_list)
        faB_num = len(faB_list)
        
        if faA_num+faB_num>2:
            count += min(faA_num,faB_num)
    
    return count


def sorted_sim(TaxID_dict,seqA,seqB):

    for TaxID in TaxID_dict:
        faA_list,faB_list = TaxID_dict[TaxID]

        faA_num = len(faA_list)
        faB_num = len(faB_list)

        if faA_num>1:
            faA_list.append(seqA)
            np_fastaA = rw_a2m.encode_a2m(faA_list)
            square_mat = np.zeros((faA_num),dtype=np.float64)
            square_mat = cal_similarity(square_mat,np_fastaA)
            filtered_index = square_mat.argsort()[::-1]

            sorted_faA_list = []
            for index in filtered_index:
                sorted_faA_list.append(faA_list[index])
            TaxID_dict[TaxID][0] = sorted_faA_list

        if faB_num>1:
            faB_list.append(seqB)
            np_fastaB = rw_a2m.encode_a2m(faB_list)
            square_mat = np.zeros((faB_num),dtype=np.float64)
            square_mat = cal_similarity(square_mat,np_fastaB)
            filtered_index = square_mat.argsort()[::-1]

            sorted_faB_list = []
            for index in filtered_index:
                sorted_faB_list.append(faB_list[index])
            TaxID_dict[TaxID][1] = sorted_faB_list

    return TaxID_dict

