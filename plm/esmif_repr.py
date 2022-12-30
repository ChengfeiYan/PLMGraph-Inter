#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 15:45:05 2022

@author: yunda_si
"""

import esm
import numpy as np


def main(esm_if1_location, pdb_file, repr_file, device):

    model, alphabet = esm.pretrained.load_model_and_alphabet_local(esm_if1_location)
    model = model.to(device)

    structure = esm.inverse_folding.util.load_structure(pdb_file)
    coords, pdb_seq = esm.inverse_folding.util.extract_coords_from_structure(structure)

    batch_converter = esm.inverse_folding.util.CoordBatchConverter(alphabet)
    batch = [(coords, None, None)]
    coords, confidence, _, _, padding_mask = batch_converter(batch)


    encoder_out = model.encoder.forward(coords.to(device), padding_mask.to(device), 
                                        confidence.to(device), return_all_hiddens=False)
    
    representations = encoder_out['encoder_out'][0][1:-1, 0].detach().cpu().numpy()
    
    np.save(repr_file, representations)

