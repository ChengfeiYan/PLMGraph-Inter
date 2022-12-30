# -*- coding: utf-8 -*-
"""
Created on Wed May 20 16:28:07 2020

@author: yunda_si
"""

import torch.nn as nn
import torch
from collections import OrderedDict
import gvp
import torch.nn.functional as F

def concat(A_f1d, B_f1d, p2d):
    
    def rep_new_axis(mat, rep_num, axis):
        return torch.repeat_interleave(torch.unsqueeze(mat,axis=axis),rep_num,axis=axis)
    
    len_channel,lenA = A_f1d.shape
    len_channel,lenB = B_f1d.shape        
    
    row_repeat = rep_new_axis(A_f1d, lenB, 2)
    col_repeat = rep_new_axis(B_f1d, lenA, 1)        


    return  torch.unsqueeze(torch.cat((row_repeat, col_repeat, p2d),axis=0),0)


def make_conv_layer(in_channels,
                    out_channels,
                    kernel_size,
                    padding_size,
                    non_linearity=True,
                    instance_norm=False,
                    dilated_rate=1):
    layers = []

    layers.append(
        ('conv', nn.Conv2d(in_channels, out_channels, kernel_size,
                           padding=padding_size, dilation=dilated_rate, bias=False)))
    # layers.append(
    #     ('dropout', nn.Dropout2d(p=0.3, inplace=True)))
    if instance_norm:
        layers.append(('in', nn.InstanceNorm2d(num_features=out_channels,momentum=0.1,
                                        affine=True,track_running_stats=False)))
    if non_linearity:
        layers.append(('leaky', nn.LeakyReLU(negative_slope=0.01,inplace=False)))

    layers.append(
        ('conv2', nn.Conv2d(out_channels, out_channels, kernel_size,
                           padding=padding_size, dilation=dilated_rate, bias=False)))
    if instance_norm:
        layers.append(('in2', nn.InstanceNorm2d(num_features=out_channels,momentum=0.1,
                                        affine=True,track_running_stats=False)))

    return nn.Sequential(OrderedDict(layers))


def make_1x1_layer(in_channels,
                    out_channels,
                    kernel_size,
                    padding_size,
                    non_linearity=True,
                    instance_norm=False,
                    dilated_rate=1):
    layers = []

    layers.append(
        ('conv', nn.Conv2d(in_channels, out_channels, kernel_size=1,
                           padding=0, dilation=1, bias=False)))
    if instance_norm:
        layers.append(('in', nn.InstanceNorm2d(num_features=out_channels,momentum=0.1,
                                        affine=True,track_running_stats=False)))
    if non_linearity:
        layers.append(('leaky', nn.LeakyReLU(negative_slope=0.01,inplace=False)))

    return nn.Sequential(OrderedDict(layers))

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels,
                       out_channels,
                       dilated_rate):
        super(BasicBlock, self).__init__()

        self.leakyrelu = nn.LeakyReLU(negative_slope=0.01,inplace=False)
        self.dilated_rate = dilated_rate
        self.concatenate = False
        self.threshold = [1,20,40]
        self.Bool_in = True
        self.Bool_nl = True

        self.conv_3x3 = make_conv_layer(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=(3,3),
                                        padding_size=(1,1),
                                        non_linearity=self.Bool_nl,
                                        instance_norm=self.Bool_in,
                                        dilated_rate=(dilated_rate,dilated_rate))

        if dilated_rate in self.threshold:
            self.conv_1xn = make_conv_layer(in_channels=in_channels,
                                            out_channels=out_channels,
                                            kernel_size=(1,15),
                                            padding_size=(0,7*dilated_rate),
                                            non_linearity=self.Bool_nl,
                                            instance_norm=self.Bool_in,
                                            dilated_rate=(1,dilated_rate))

            self.conv_nx1 = make_conv_layer(in_channels=in_channels,
                                            out_channels=out_channels,
                                            kernel_size=(15,1),
                                            padding_size=(7*dilated_rate,0),
                                            non_linearity=self.Bool_nl,
                                            instance_norm=self.Bool_in,
                                            dilated_rate=(dilated_rate,1))

            if self.concatenate:
                self.conv_1x1 = make_1x1_layer(in_channels=in_channels*3,
                                               out_channels=out_channels,
                                               kernel_size=(1,1),
                                               padding_size=(0,0),
                                               non_linearity=self.Bool_nl,
                                               instance_norm=self.Bool_in,
                                               dilated_rate=(1,1))

    def forward(self, x):

        out = x

        identity1 = self.conv_3x3(x)

        if self.dilated_rate in self.threshold:
            identity2 = self.conv_1xn(x)
            identity3 = self.conv_nx1(x)

            if self.concatenate:
                identity = torch.cat((identity1,identity2,identity3),1)
                identity = self.conv_1x1(identity)
            else:
                identity = identity1+identity2+identity3

        else:
            identity = identity1

        out = out+identity

        if self.Bool_nl:
            out = self.leakyrelu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, blocks_num, gvp_num):
        super(ResNet, self).__init__()
        self.in_channel = 96
        node_input_dim = (2586, 50)
        edge_input_dim = (432, 25)
        
        node_hidden_dim = (256, 64)
        edge_hidden_dim = (432, 25)
        
        input_channel = 0*(node_input_dim[0] + 0*node_input_dim[1])  \
                        + 2*(node_hidden_dim[0] + 3*node_hidden_dim[1])  \
                        + 1*(144 + 4)
                        
        self.first_layer = make_1x1_layer(in_channels=input_channel, #4176,#3902,#7998,#4198,#5422
                                          out_channels=self.in_channel,
                                          kernel_size=(1,1),
                                          padding_size=(0,0),
                                          non_linearity=True,
                                          instance_norm=True,
                                          dilated_rate=(1,1))
        
        self.hidden_layer = self._make_layer(in_channel=self.in_channel, out_channel=self.in_channel,
                                             block_num=blocks_num,dilated_rate=1)

        self.output_layer = make_1x1_layer(in_channels=self.in_channel,
                                          out_channels=1,
                                          kernel_size=(1,1),
                                          padding_size=(0,0),
                                          non_linearity=False,
                                          instance_norm=False,
                                          dilated_rate=(1,1))
        
        
        self.Sig = nn.Sigmoid()

        self.embed_node = nn.Sequential(
            gvp.GVP(node_input_dim, node_hidden_dim, 
                    activations=(None, None), vector_gate=True),
            gvp.LayerNorm(node_hidden_dim))    

        self.gvp_layers = self._make_gvpconv_layer(node_hidden_dim, edge_hidden_dim, gvp_num)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.01, mode='fan_in',nonlinearity='leaky_relu')

    def _make_layer(self, in_channel, out_channel, block_num, dilated_rate):

        layers = []

        for index in range(block_num):
            layers.append(('block'+str(index),BasicBlock(in_channel, out_channel, dilated_rate)))

        return nn.Sequential(OrderedDict(layers))


    def _make_gvpconv_layer(self, node_dim, edge_dim, layer_num):
        
        layers = []
        
        for index in range(layer_num):
            layers.append(gvp.GVPConvLayer(node_dim, edge_dim,  
                                           n_message=3, n_feedforward=2,
                                           drop_rate=0.1, vector_gate=True))
            
        return nn.ModuleList(layers)
        
    
    def forward(self, nodeA, edgeA, edge_indexA, 
                      nodeB, edgeB, edge_indexB, 
                      p2d):
        
        strucA = self.embed_node(nodeA)
        strucB = self.embed_node(nodeB)
        
        for layer in self.gvp_layers:
            strucA = layer(strucA, edge_indexA, edgeA)
            strucB = layer(strucB, edge_indexB, edgeB)            

        strucsA, strucvA = strucA
        strucsB, strucvB = strucB
 
        nodesA = torch.hstack((strucsA,  strucvA.flatten(-2,-1)))
        nodesB = torch.hstack((strucsB,  strucvB.flatten(-2,-1)))
        
        Input = concat(nodesA.T, nodesB.T, p2d)

        x = self.first_layer(Input)

        x = self.hidden_layer(x)

        x = self.output_layer(x)

        x = torch.squeeze(x)

        x = self.Sig(x)

        return x



def resnet18():
    return ResNet(9, 3)

