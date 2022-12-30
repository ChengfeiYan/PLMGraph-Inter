#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 16:37:08 2022

@author: yunda_si
"""

import numpy as np
import torch
import torch_geometric
import pickle
import torch.nn.functional as F
from Bio.PDB.PDBParser import PDBParser


def nan_to_num(ts, val=0.0):
    """
    Replaces nans in tensor with a fixed value.    
    """
    val = torch.tensor(val, dtype=ts.dtype, device=ts.device)
    return torch.where(~torch.isfinite(ts), val, ts)

def norm(tensor, dim, eps=1e-8, keepdim=False):
    """
    Returns L2 norm along a dimension.
    """
    return torch.sqrt(
            torch.sum(torch.square(tensor), dim=dim, keepdim=keepdim))


def normalize(tensor, dim=-1):
    """
    Normalizes a tensor along a dimension after removing nans.
    """
    return nan_to_num(
        torch.div(tensor, norm(tensor, dim=dim, keepdim=True))
    )


def get_rotation_frames(coords):
    # From https://github.com/facebookresearch/esm/tree/982c3f69478f141de60a6fa35fff2880793141ce/esm/inverse_folding
    """
    Returns a local rotation frame defined by N, CA, C positions.
    Args:
        coords: coordinates, tensor of shape (length x 5 x 3)
        where the second dimension is in order of N, CA, C, O, Cb
    Returns:
        Local relative rotation frames in shape (length x 5 x 3)
    """    
    v1 = coords[:, 1] - coords[:, 2]
    v2 = coords[:, 1] - coords[:, 0]
    e1 = normalize(v1, dim=-1)
    u2 = v2 - e1 * torch.sum(e1 * v2, dim=-1, keepdim=True)
    e2 = normalize(u2, dim=-1)
    e3 = torch.cross(e1, e2, dim=-1)
    R = torch.stack([e1, e2, e3], dim=-2)     
        
    return R


def get_local_orientations(local_coords, L, device):
    
    forward_coords = torch.cat((torch.zeros(L,1,5,3).to(device),
                                local_coords[:,:-1,:,:]),1)
    backward_coords = torch.cat((local_coords[:,1:,:,:],
                                 torch.zeros(L,1,5,3).to(device)),1)
    
    nodes_vec = torch.zeros((L,50,3)).to(device)
    
    for b in range(L):
        forward = forward_coords[b,b]
        ori = local_coords[b,b]
        backward = backward_coords[b,b]
        
        forward = torch.repeat_interleave(forward,5,0)
        backward = torch.repeat_interleave(backward,5,0)
        ori = ori.repeat(5,1)
        
        forward_vec = ori - forward
        backward_vec = ori - backward
        
        nodes_vec[b] = torch.cat((forward_vec, backward_vec), 0)
    
    nodes_vec = normalize(nodes_vec,-1)
        
    return nodes_vec


def positional_embeddings(edge_index, num_embeddings=16):
    # From https://github.com/jingraham/neurips19-graph-protein-design

    d = edge_index[0] - edge_index[1]
 
    frequency = torch.exp(
        torch.arange(0, num_embeddings, 1, dtype=torch.float32)
        * -(np.log(10000.0) / num_embeddings)
    )
    angles = d.unsqueeze(-1) * frequency
    E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
    return E
    

def cal_dismap(tensor, L):
    '''
    Calculate pairwise distances
    '''
    dis_map = tensor.unsqueeze(1).repeat(1,L,1) - \
              tensor.unsqueeze(0).repeat(L,1,1)
    dis_map = torch.sqrt(torch.sum(dis_map**2,-1)) 
    
    return dis_map
    

def cal_edge_index(coords, contact_cutoff, L):
    Ca_coords = coords[:,1]
    dis_map = cal_dismap(Ca_coords, L)
    
    dis_map[dis_map>contact_cutoff] = 0
    dis_map[dis_map!=0] = 1
    edge_index = torch_geometric.utils.dense_to_sparse(dis_map)[0]
    
    return edge_index
        
    
def rbf(tensor, D_min=2, D_max=12, D_count=16):
    # From https://github.com/jingraham/neurips19-graph-protein-design
    
    D_mu = torch.linspace(D_min, D_max, D_count)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    RBF = torch.exp(-((tensor.unsqueeze(-1) - D_mu) / D_sigma) ** 2)
    
    return RBF
    

def edge_distance(coords, edge_index, contact_cutoff, rbf_count, len_edge):
    
    left_edge = coords[edge_index[0]]
    right_edge = coords[edge_index[1]]
    edge_dis = torch.cdist(left_edge,right_edge).reshape(len_edge, 5*5)
    edge_rbf = rbf(edge_dis, D_max=contact_cutoff, D_count=rbf_count)
    edge_rbf = edge_rbf.transpose(-2,-1).reshape(len_edge,25*rbf_count)
    
    return edge_rbf
        

def edge_vector(len_edge, local_coords, edge_index):
    
    edge_vec = torch.zeros(len_edge,25,3)
    for index,(left,right) in enumerate(edge_index.T.tolist()):
        left_coords = local_coords[left,left]
        right_coords = local_coords[left,right]
        edge_vec[index] = right_coords.repeat(5,1) - torch.repeat_interleave(left_coords,5,0)
    edge_vec = normalize(edge_vec,-1)
    
    return edge_vec
        

def virtualCb(coords):
    N, Ca, C = coords[:, 0], coords[:, 1], coords[:, 2] 
    b = Ca - N
    c = C - Ca
    a = torch.cross(b, c)
    
    return -0.58273431*a + 0.56802827*b - 0.54067466*c + Ca


def dihedrals(X, eps=1e-7):
    # From https://github.com/jingraham/neurips19-graph-protein-design
    X = X[:,:-1,:]
    X = torch.reshape(X[:, :3], [3*X.shape[0], 3])
    dX = X[1:] - X[:-1]
    U = normalize(dX, dim=-1)
    u_2 = U[:-2]
    u_1 = U[1:-1]
    u_0 = U[2:]

    # Backbone normals
    n_2 = normalize(torch.cross(u_2, u_1), dim=-1)
    n_1 = normalize(torch.cross(u_1, u_0), dim=-1)

    # Angle between normals
    cosD = torch.sum(n_2 * n_1, -1)
    cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
    D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)

    # This scheme will remove phi[0], psi[-1], omega[-1]
    D = F.pad(D, [1, 2]) 
    D = torch.reshape(D, [-1, 3])
    # Lift angle representations to the circle
    D_features = torch.cat([torch.cos(D), torch.sin(D)], 1)
    return D_features
    


def main(pdb_file, feature_file):
    
    contact_cutoff = 18
    rbf_count = 16
    device = 'cpu'
    torch.set_num_threads(1)


    # get coords
    
    parser = PDBParser()
    structure = parser.get_structure('fas_seq',pdb_file)
    model = structure[0]
    chain = model['A']
    coords = np.stack([[aa[atom].coord for atom in ['N', 'CA', 'C', 'O']] for aa in chain])
    # coords = np.load(coords_file)
    coords = torch.from_numpy(coords)
    Cb = virtualCb(coords)
    coords = torch.cat((coords,Cb.unsqueeze(1)),1).numpy()
    
    coords = torch.from_numpy(coords).to(device)
    L = coords.shape[0]
    R = get_rotation_frames(coords)
        
    
    #  Rotate and translate all coordinates with Ca of each residue 
    #  as the center to construct the local coordinate system
    trans_coords = coords.unsqueeze(0).repeat(L,1,1,1) - \
        coords[:,1].unsqueeze(1).unsqueeze(1).repeat(1,L,5,1)
    
    local_coords = torch.einsum('blij,bjr->blir',[trans_coords, R.transpose(-2,-1)])
    
    
    
    # orientations
    nodes_vec = get_local_orientations(local_coords, L, device)


    # node scatter
    nodes_sact = dihedrals(coords, eps=1e-7)
    
    
    # edge index
    edge_index = cal_edge_index(coords, contact_cutoff, L)
    len_edge = edge_index.shape[-1]
    
    
    # position embedding
    edge_position = positional_embeddings(edge_index)
    
    
    # edge distance
    edge_rbf = edge_distance(coords, edge_index, contact_cutoff, rbf_count, len_edge)
    
    
    # edge vector
    edge_vec = edge_vector(len_edge, local_coords, edge_index)

    
    feature_dict = {'nodes_vec':nodes_vec,
                    'nodes_sact':nodes_sact,
                    'edge_scat':torch.cat((edge_rbf,edge_position),1),
                    'edge_index':edge_index,
                    'edge_vec':edge_vec}
    
    
    pickle.dump(feature_dict, open(feature_file,'wb'))
        
        