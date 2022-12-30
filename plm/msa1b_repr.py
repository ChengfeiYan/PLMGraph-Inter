#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 14:25:09 2022

@author: yunda_si
"""

import pickle
import torch
from typing import List, Tuple
import numpy as np
from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained
from Bio import SeqIO
import string
import itertools
import os


deletekeys = dict.fromkeys(string.ascii_lowercase)
deletekeys["."] = None
deletekeys["*"] = None
translation = str.maketrans(deletekeys)

def read_sequence(filename: str) -> Tuple[str, str]:
    """ Reads the first (reference) sequences from a fasta or MSA file."""
    record = next(SeqIO.parse(filename, "fasta"))
    return record.description, str(record.seq)

def remove_insertions(sequence: str) -> str:
    """ Removes any insertions into the sequence. Needed to load aligned sequences in an MSA. """
    return sequence.translate(translation)

def read_msa(filename: str, nseq: int) -> List[Tuple[str, str]]:
    """ Reads the first nseq sequences from an MSA file, automatically removes insertions."""
    return [(record.description, remove_insertions(str(record.seq)))
            for record in itertools.islice(SeqIO.parse(filename, "fasta"), nseq)]



def main(esm_msa1b_location, msa_file, repr_file, device):
        
    repr_layers = [12]
    max_msa = 256
    
    model, alphabet = pretrained.load_model_and_alphabet_local(esm_msa1b_location)
    model = model.eval().to(device)
    msa_batch_converter = alphabet.get_batch_converter()
    
    msa_data = read_msa(msa_file, max_msa)
    msa_labels, msa_strs, msa_tokens = msa_batch_converter(msa_data)
    
    with torch.no_grad():
        msa_tokens = msa_tokens.to(device=device, non_blocking=True)
        out = model(msa_tokens, repr_layers=repr_layers, return_contacts=True)
        representations = out['representations'][12].cpu().numpy()[0,0,1:,:]
        
    np.save(repr_file,representations)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    