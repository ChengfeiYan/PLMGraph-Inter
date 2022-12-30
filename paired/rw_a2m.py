#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 14:19:49 2021

@author: yunda_si
"""

import numpy as np
from collections import deque

def read_a2m(a2m_file,lenseq,min_cov=0):
    msa_data = []
    seq = 'TEMP'
    gap_cov = 1-min_cov
    header = deque(['',''],maxlen=2)

    with open(a2m_file,'rb',buffering=8192) as f:
        for row in f:
            row = bytes.decode(row).strip()
            if row.startswith('>'):
                header.append(row)
                if seq.count('-')<lenseq*gap_cov:
                    msa_data.append((header.popleft(),seq))
                seq = ''
            else:
                seq += row
        lenseq = len(seq)
        if seq.count('-')<lenseq*gap_cov:
            msa_data.append((header.popleft(),seq))

    f.close()
    return msa_data[1:]


def encode_a2m(msa_data):

    residue_constant = {'-':0, 'A':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6, 'H':7,
                  'I':8, 'K':9, 'L':10, 'M':11, 'N':12, 'P':13, 'Q':14,
                  'R':15, 'S':16, 'T':17, 'V':18, 'W':19, 'Y':20}

    alignment_height = len(msa_data)
    alignment_width = len(msa_data[0][-1][-1])

    encoded_alignment = np.zeros((alignment_height,alignment_width),dtype=np.uint8)

    for ii in range(alignment_height):
        sequence = msa_data[ii][-1][-1]
        encoded_alignment[ii,:] = [residue_constant.get(amino,0) for amino in sequence]

    return encoded_alignment



def read_refseq(fasta_file):

    header,sequence = open(fasta_file).readlines()
    return (['','','','','',''],[header.strip(),sequence.strip()])

    
def parse_msa(msa_data):
    parsed_msa = []

    standard_amino = set('ARNDCQEGHILKMFPSTWYV-')

    
    for header,sequence in msa_data:

        try:
            UqID,temp = header[1:].split(maxsplit=1)
                    
            n_index = temp.index('n=')
            Molecule,[Members,temp] = temp[:n_index].strip(),temp[n_index:].split(maxsplit=1)
            
            TaxID_index = temp.index('TaxID=')
            Tax,[TaxID,RepID] = temp[:TaxID_index].strip(),temp[TaxID_index:].split()

            Tax = Tax.split('=')[1]
            TaxID = int(TaxID.split('=')[1])
            Members = int(Members.split('=')[1])
            RepID = RepID.split('=')[1]
                
        except:
            continue

        seq_amino = set(sequence)
        Molecule = Molecule.upper()
        if seq_amino.issubset(standard_amino):
            temp = ([UqID,Molecule,Members,Tax,TaxID,RepID],[header,sequence])
            parsed_msa.append(temp)

    return parsed_msa









