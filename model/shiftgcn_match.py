import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from model.utils.tgcn import ConvTemporalGraphical
from model.utils.graph import Graph
import math
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
from sklearn import manifold
import numpy as np
from scipy.special import binom



import sys
sys.path.append("./model/Temporal_shift/")

from cuda.shift import Shift




def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

def conv_init(conv):
    nn.init.kaiming_normal(conv.weight, mode='fan_out')
    nn.init.constant(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant(bn.weight, scale)
    nn.init.constant(bn.bias, 0)

def ln_init(layer):
    nn.init.ones_(layer.weight)
    nn.init.zeros_(layer.bias)


class tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class Shift_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(Shift_tcn, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.bn = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(in_channels)
        bn_init(self.bn2, 1)
        self.relu = nn.ReLU(inplace=True)
        self.shift_in = Shift(channel=in_channels, stride=1, init_scale=1)
        self.shift_out = Shift(channel=out_channels, stride=stride, init_scale=1)

        self.temporal_linear = nn.Conv2d(in_channels, out_channels, 1)
        nn.init.kaiming_normal(self.temporal_linear.weight, mode='fan_out')

    def forward(self, x):
        x = self.bn(x)
        # shift1
        x = self.shift_in(x)
        x = self.temporal_linear(x)
        x = self.relu(x)
        # shift2
        x = self.shift_out(x)
        x = self.bn2(x)
        return x


class Shift_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, num_subset=3):
        super(Shift_gcn, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.Linear_weight = nn.Parameter(torch.zeros(in_channels, out_channels, requires_grad=True, device='cuda'), requires_grad=True)
        nn.init.normal_(self.Linear_weight, 0,math.sqrt(1.0/out_channels))

        self.Linear_bias = nn.Parameter(torch.zeros(1,1,out_channels,requires_grad=True,device='cuda'),requires_grad=True)
        nn.init.constant(self.Linear_bias, 0)

        self.Feature_Mask = nn.Parameter(torch.ones(1,25,in_channels, requires_grad=True,device='cuda'),requires_grad=True)
        nn.init.constant(self.Feature_Mask, 0)

        self.bn = nn.BatchNorm1d(25*out_channels)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

        index_array = np.empty(25*in_channels).astype(int)
        # index_array = np.empty(25*in_channels, dtype=int)
        for i in range(25):
            for j in range(in_channels):
                index_array[i*in_channels + j] = (i*in_channels + j + j*in_channels)%(in_channels*25)
        self.shift_in = nn.Parameter(torch.from_numpy(index_array),requires_grad=False)

        index_array = np.empty(25*out_channels).astype(int)
        for i in range(25):
            for j in range(out_channels):
                index_array[i*out_channels + j] = (i*out_channels + j - j*out_channels)%(out_channels*25)
        self.shift_out = nn.Parameter(torch.from_numpy(index_array),requires_grad=False)
        

    def forward(self, x0):
        n, c, t, v = x0.size()
        x = x0.permute(0,2,3,1).contiguous()

        # shift1
        x = x.view(n*t,v*c)
        x = torch.index_select(x, 1, self.shift_in)
        x = x.view(n*t,v,c)
        x = x * (torch.tanh(self.Feature_Mask)+1)

        x = torch.einsum('nwc,cd->nwd', (x, self.Linear_weight)).contiguous() # nt,v,c
        x = x + self.Linear_bias

        # shift2
        x = x.view(n*t,-1) 
        x = torch.index_select(x, 1, self.shift_out)
        x = self.bn(x)
        x = x.view(n,t,v,self.out_channels).permute(0,3,1,2) # n,c,t,v

        x = x + self.down(x0)
        x = self.relu(x)
        return x


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = Shift_gcn(in_channels, out_channels, A)
        self.tcn1 = Shift_tcn(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU()

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.tcn1(self.gcn1(x)) + self.residual(x)
        return self.relu(x)



class SHIFTGCNModel(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3):
        super(SHIFTGCNModel, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        self.l1 = TCN_GCN_unit(3, 64, A, residual=False)
        self.l2 = TCN_GCN_unit(64, 64, A)
        self.l3 = TCN_GCN_unit(64, 64, A)
        self.l4 = TCN_GCN_unit(64, 64, A)
        self.l5 = TCN_GCN_unit(64, 128, A, stride=2)
        self.l6 = TCN_GCN_unit(128, 128, A)
        self.l7 = TCN_GCN_unit(128, 128, A)
        self.l8 = TCN_GCN_unit(128, 256, A, stride=2)
        self.l9 = TCN_GCN_unit(256, 256, A)
        self.l10 = TCN_GCN_unit(256, 256, A)

        self.fc = nn.Linear(256, num_class)
        nn.init.normal(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)

    def forward(self, x):
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)

        # N*M,C,T,V
        c_new = x.size(1)

        x_match_feature = x.view(N, c_new, T//4, V, M)
        x_match_feature = x_match_feature.mean(4)
        
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        # x = self.fc(x)
        # return x, self.fc(x)
        return x_match_feature, x





class STGCNModel(nn.Module):
    r"""Spatial temporal graph convolutional networks.

    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_args (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    def __init__(self, in_channels,  num_class, graph_args,
                 edge_importance_weighting, head=['ViT-B/32'], num_point=25, num_person=2, graph=None, **kwargs):
        super().__init__()

        # load graph
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels, 64, kernel_size, 1, residual=False, **kwargs0),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 128, kernel_size, 2, **kwargs),
            st_gcn(128, 128, kernel_size, 1, **kwargs),
            st_gcn(128, 128, kernel_size, 1, **kwargs),
            st_gcn(128, 256, kernel_size, 2, **kwargs),
            st_gcn(256, 256, kernel_size, 1, **kwargs),
            st_gcn(256, 256, kernel_size, 1, **kwargs),
        ))

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        # fcn for prediction
        self.fcn = nn.Conv2d(256, num_class, kernel_size=1)

        # contrastive learning
        self.linear_head_text = nn.ModuleDict()
        # self.linear_head_image = nn.ModuleDict()

        self.logit_scale_text = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        # self.logit_scale_image = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # self.text_part_list = nn.ModuleList()

        # for i in range(6):
        #     self.text_part_list.append(nn.Linear(256,512))
        self.head = head
        if 'ViT-B/32' in self.head:
            self.linear_head_text['ViT-B/32'] = nn.Linear(256,512)
            conv_init(self.linear_head_text['ViT-B/32'])
            # self.linear_head_image['ViT-B/32'] = nn.Linear(256,512)
            # conv_init(self.linear_head_image['ViT-B/32'])
        if 'ViT-B/16' in self.head:
            self.linear_head_text['ViT-B/16'] = nn.Linear(256,512)
            conv_init(self.linear_head_text['ViT-B/16'])
            # self.linear_head_image['ViT-B/16'] = nn.Linear(256,512)
            # conv_init(self.linear_head_image['ViT-B/16'])
        if 'ViT-L/14' in self.head:
            self.linear_head_text['ViT-L/14'] = nn.Linear(256,768)
            conv_init(self.linear_head_text['ViT-L/14'])
            # self.linear_head_image['ViT-L/14'] = nn.Linear(256,768)
            # conv_init(self.linear_head_image['ViT-L/14'])
        if 'ViT-L/14@336px' in self.head:
            self.linear_head_text['ViT-L/14@336px'] = nn.Linear(256,768)
            conv_init(self.linear_head_text['ViT-L/14@336px'])
            # self.linear_head_image['ViT-L/14@336px'] = nn.Linear(256,768)
            # conv_init(self.linear_head_image['ViT-L/14@336px'])
        if 'RN50x64' in self.head:
            self.linear_head_text['RN50x64'] = nn.Linear(256,1024)
            conv_init(self.linear_head_text['RN50x64'])
            # self.linear_head_image['RN50x64'] = nn.Linear(256,1024)
            # conv_init(self.linear_head_image['RN50x64'])
        if 'RN50x16' in self.head:
            self.linear_head_text['RN50x16'] = nn.Linear(256,768)
            conv_init(self.linear_head_text['RN50x16'])
            # self.linear_head_image['RN50x16'] = nn.Linear(256,768)
            # conv_init(self.linear_head_image['RN50x16'])

    def forward(self, x):

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forwad
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)


        # N*M,C,T,V
        c_new = x.size(1)

        

        x_match_feature = x.view(N, c_new, T//4, V, M)
        x_match_feature = x_match_feature.mean(4)

        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)

        text_feature_dict = dict()
        # image_feature_dict = dict()

        for name in self.head:
            text_feature_dict[name] = self.linear_head_text[name](x)   # global text feature
            # image_feature_dict[name] = self.linear_head_image[name](x)   # global image feature



        return x_match_feature, x, text_feature_dict, self.logit_scale_text
    

    def extract_feature(self, x):

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forwad
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        _, c, t, v = x.size()
        feature = x.view(N, M, c, t, v).permute(0, 2, 3, 4, 1)

        # prediction
        x = self.fcn(x)
        output = x.view(N, M, -1, t, v).permute(0, 2, 3, 4, 1)

        return output, feature










class ModelMatch(nn.Module):
    def __init__(self,num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3,
                 drop_out=0, adaptive=True, body_part=6, motion=90, unseen=55, project_dim=256):
        super(ModelMatch, self).__init__()
        # pretrain model
        self.pretraining_model = SHIFTGCNModel(num_class=num_class,num_point=num_point, num_person=num_person, graph=graph, graph_args=graph_args, in_channels=in_channels)
        for p in self.parameters():
            p.requires_grad = False
        self.body_part = body_part
        self.motion = motion
        self.body_part_index_list = nn.ParameterList()
        # head hand arm hip leg foot 
        self.body_part_index_list.append(nn.Parameter(torch.Tensor([2,3,20]).long(), requires_grad=False))  # head
        self.body_part_index_list.append(nn.Parameter(torch.Tensor([7,11,21,22,23,24]).long(), requires_grad=False)) # hand
        self.body_part_index_list.append(nn.Parameter(torch.Tensor([4,5,6,8,9,10]).long(), requires_grad=False))  # arm
        self.body_part_index_list.append(nn.Parameter(torch.Tensor([0,1]).long(), requires_grad=False))  # hip
        self.body_part_index_list.append(nn.Parameter(torch.Tensor([12,13,16,17]).long(), requires_grad=False))  # leg
        self.body_part_index_list.append(nn.Parameter(torch.Tensor([14,15,18,19]).long(), requires_grad=False))  # foot
        self.body_part_st_attention_networks = nn.ModuleList([nn.MultiheadAttention(embed_dim=project_dim, kdim=256, vdim=256, num_heads=4, batch_first=True) for _ in range(self.body_part)])
        self.semantic_attention_networks = nn.ModuleList([nn.MultiheadAttention(embed_dim=768, kdim=768, vdim=768, num_heads=4, batch_first=True) for _ in range(self.body_part)])
        # self.semantic_attention_networks = nn.ModuleList([ScaledMultiheadAttention(embed_dim=768, kdim=768, vdim=768, num_heads=4, batch_first=True) for _ in range(self.body_part)])
        

        self.ske_ffn = nn.ModuleList([nn.Sequential(nn.Linear(project_dim, project_dim), nn.ReLU(), nn.Linear(project_dim, project_dim)) for _ in range(self.body_part)])
        for param in self.ske_ffn:
            for layer in param:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal(layer.weight)
                    nn.init.constant(layer.bias, 0)
        self.sem_ffn = nn.ModuleList([nn.Sequential(nn.Linear(768, 768), nn.ReLU(), nn.Linear(768, 768)) for _ in range(self.body_part)])
        for param in self.sem_ffn:
            for layer in param:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal(layer.weight)
                    nn.init.constant(layer.bias, 0)

        self.body_part_prompts = nn.ParameterList([nn.Parameter(nn.init.normal_(torch.empty(motion, project_dim)), requires_grad=True) for _ in range(self.body_part)])
        
        self.norm1 = nn.ModuleList([nn.LayerNorm(project_dim) for _ in range(self.body_part)])
        for layer in self.norm1:
            ln_init(layer)
        self.norm2 = nn.ModuleList([nn.LayerNorm(project_dim) for _ in range(self.body_part)])
        for layer in self.norm2:
            ln_init(layer)
        self.semantic_norm1 = nn.ModuleList([nn.LayerNorm(768) for _ in range(self.body_part)])
        for layer in self.semantic_norm1:
            ln_init(layer)
        self.semantic_norm2 = nn.ModuleList([nn.LayerNorm(768) for _ in range(self.body_part)])
        for layer in self.semantic_norm2:
            ln_init(layer)
  
        self.ske_projector = nn.ModuleList([nn.Linear(project_dim, project_dim) for _ in range(self.body_part)])
        for layer in self.ske_projector:
            nn.init.kaiming_normal(layer.weight)
            nn.init.constant(layer.bias, 0)
        
        self.text_projector = nn.ModuleList([nn.Linear(768, project_dim) for _ in range(self.body_part)])  # 768 512 512 256 (48_12) 
        for layer in self.text_projector:
            nn.init.kaiming_normal(layer.weight)
            nn.init.constant(layer.bias, 0)
      
        self.ske_dummy = nn.Parameter(nn.init.normal_(torch.empty(1, 1, project_dim)), requires_grad=True)
        self.text_dummy = nn.Parameter(nn.init.normal_(torch.empty(1, 1, project_dim)), requires_grad=True)
        self.ske_adj_A = nn.Parameter(nn.init.normal(torch.randn(7, 256)), requires_grad=True)
        self.text_adj_A = nn.Parameter(nn.init.normal(torch.randn(7, 768)), requires_grad=True)
 
        self.condition_project = nn.Linear(project_dim, project_dim)
        nn.init.kaiming_normal(self.condition_project.weight)
        nn.init.constant(self.condition_project.bias, 0)

    def construct_skeleton_adjacent_matrix(self, ske):
        n, _, _ = ske.size()
        mask = torch.tensor([[1,1,1,1,1,1,1],
                          [0,1,1,1,1,1,1],
                          [0,1,1,1,1,1,1],
                          [0,1,1,1,1,1,1],
                          [0,1,1,1,1,1,1],
                          [0,1,1,1,1,1,1],
                          [0,1,1,1,1,1,1]], device=ske.device).float().unsqueeze(0).expand(n, -1, -1)
        adj_A = torch.einsum('npd,kd->npk', ske, self.ske_adj_A)
        mask = mask.clone()
        mask[:,0,:] += torch.sigmoid(adj_A[:,0,:])
        mask[:,1:,1:] += torch.sigmoid(adj_A[:,1:,1:])
        adjacency_matrices = mask
        return adjacency_matrices
    

    def construct_text_adjacent_matrix(self, text):
        c, _, _ = text.size()
        mask = torch.tensor([[1,1,1,1,1,1,1],
                            [0,1,1,1,1,1,1],
                            [0,1,1,1,1,1,1],
                            [0,1,1,1,1,1,1],
                            [0,1,1,1,1,1,1],
                            [0,1,1,1,1,1,1],
                            [0,1,1,1,1,1,1]], device=text.device).float().unsqueeze(0).expand(c, -1, -1)
        adj_A = torch.einsum('npd,kd->npk', text, self.text_adj_A)
        mask = mask.clone()
        mask[:,0,:] += torch.sigmoid(adj_A[:,0,:])
        mask[:,1:,1:] += torch.sigmoid(adj_A[:,1:,1:])
        adjacency_matrices = mask
        return adjacency_matrices
    

    def forward(self, x):
        gcn_x, _ = self.pretraining_model(x)
        n,c,t,v = gcn_x.size()
        emb_fea_list = []
        gcn_fea_list = []
        gcn_fea_list.append(gcn_x.mean(3).mean(2).unsqueeze(1))
        for i, part_name in enumerate(["head", "hand", "arm", "hip", "leg", "foot"]):
            # skeleton stream
            ske_fea = gcn_x[:,:,:,self.body_part_index_list[i]].view(n,c,-1).permute(0,2,1)
            gcn_fea_list.append(ske_fea.mean(1).unsqueeze(1))
            att_motion, _ = self.body_part_st_attention_networks[i](self.body_part_prompts[i].unsqueeze(0).expand(n, -1, -1), ske_fea, ske_fea)  # n 100 256
            att_motion = self.norm1[i](att_motion)
            aug_fea = self.ske_ffn[i](att_motion)
            aug_fea = self.norm2[i](aug_fea)   # n 100 256
            emb_fea = self.ske_projector[i](aug_fea).mean(1)
            emb_fea_list.append(emb_fea.unsqueeze(1))

        ske_part = torch.cat(emb_fea_list, dim=1)  # n 7 256
        emb_part = ske_part
        gcn_fea_list = torch.cat(gcn_fea_list, dim=1)
        A = self.construct_skeleton_adjacent_matrix(gcn_fea_list)
        ske_dummy = torch.cat((self.ske_dummy.expand(n,-1,-1), ske_part), dim=1)
        ske_emb = torch.matmul(A, ske_dummy)
        ske_emb = torch.matmul(A, ske_emb)
        emb_global = ske_emb[:,0,:] 
        
        return emb_part, emb_global

    def text_interaction(self, text, pool_feature):
        semantic_fea = []
        attention_value = []
        for i, part_name in enumerate(["head", "hand", "arm", "hip", "leg", "foot"]):
            q = pool_feature[i].cuda(text.device)  # token, 768
            kv = text[:,i,:,:] # category 77 256
            c, _, _ = kv.size()
            att_semantic, att_value = self.semantic_attention_networks[i](q.unsqueeze(0).expand(c, -1, -1), kv, kv)
            att_semantic = self.semantic_norm1[i](att_semantic)
            aug_fea = self.sem_ffn[i](att_semantic)
            aug_fea = self.semantic_norm2[i](aug_fea)   # category token 256
            emb_fea = self.text_projector[i](aug_fea).mean(1)
            semantic_fea.append(emb_fea.unsqueeze(1))
            attention_value.append(att_value)
        semantic_fea = torch.cat(semantic_fea, dim=1) 
        return semantic_fea, attention_value
        


    def loss_cal(self, emb_part, emb_global, an, part_language, label, pool_feature, temperature_rate):
        n, _, _ = emb_part.size()
        # construct text adjacent matrix
        text = part_language  # 48 6 77 768
        text_con = torch.cat((an.mean(1).unsqueeze(1), text.mean(2)), dim=1)
        A = self.construct_text_adjacent_matrix(text_con)  # 48 6 768
        text_part, text_att_weights = self.text_interaction(text, pool_feature)
        # text GCN
        c, _, _ = text_part.size()
        text_dummy = torch.cat((self.text_dummy.expand(c, -1, -1), text_part), dim=1)
        text_emb = torch.matmul(A, text_dummy)
        text_emb = torch.matmul(A, text_emb)
        text_global = text_emb[:,0,:]
        # multi-part cross-entropy loss:
        part_sim = torch.sum(F.normalize(emb_part, dim=2, p=2).unsqueeze(2) * F.normalize(text_part, dim=2, p=2).permute(1, 0, 2).unsqueeze(0), dim=-1)# n 6 48
        loss_part_ce = []
        for i in range(self.body_part):
            loss_part = self.nce_loss(part_sim[:,i,:], label.float(), temperature_rate)
            loss_part_ce.append(loss_part)
        loss_part_ce = sum(loss_part_ce)/6
        # global cross-entropy loss:
        col_idx = torch.arange(self.body_part).unsqueeze(0).expand(n, self.body_part) 
        part_softmax_idx = torch.max(torch.softmax(part_sim, dim=-1), dim=2)[1]
        condition = self.condition_project(text_part[part_softmax_idx, col_idx, :].mean(1))
        global_sim = torch.einsum('nk,ck->nc', F.normalize(emb_global+condition, dim=1, p=2), F.normalize(text_global, dim=1, p=2)) # n 48
        loss_global_ce = self.nce_loss(global_sim, label.float(), temperature_rate)

        return loss_global_ce, loss_part_ce

    def get_zsl_acc(self, emb_global, emb_part, an, label, part_language, pool_feature):
        n, _, _ = emb_part.size()
        # construct text adjacent matrix
        text = part_language
        text_con = torch.cat((an.mean(1).unsqueeze(1), text.mean(2)), dim=1)
        A = self.construct_text_adjacent_matrix(text_con)  # 48 6 768
        text_part, text_att_weights = self.text_interaction(text, pool_feature)
        # text GCN
        c, _, _ = text_part.size()
        text_dummy = torch.cat((self.text_dummy.expand(c, -1, -1), text_part), dim=1)
        text_emb = torch.matmul(A, text_dummy)
        text_emb = torch.matmul(A, text_emb)
        text_global = text_emb[:,0,:]
        # condition
        col_idx = torch.arange(self.body_part).unsqueeze(0).expand(n, self.body_part)  
        part_sim = torch.sum(F.normalize(emb_part, dim=2, p=2).unsqueeze(2) * F.normalize(text_part, dim=2, p=2).permute(1, 0, 2).unsqueeze(0), dim=-1)
        part_softmax_idx = torch.max(part_sim, dim=2)[1]
        condition = self.condition_project(text_part[part_softmax_idx, col_idx, :].mean(1))
        # global
        global_sim = torch.einsum('nk,ck->nc', F.normalize(emb_global+condition, dim=1, p=2), F.normalize(text_global, dim=1, p=2))
        global_sim = F.softmax(global_sim, dim=1)
        global_sim_idx = torch.max(global_sim, dim=1)[1].data.cpu().numpy()
        label = torch.max(label, dim=1)[1].data.cpu().numpy()
        return global_sim_idx, label, text_att_weights 
    
    def get_gzsl_acc(self, emb_global, emb_part, an, label, part_language, pool_feature, calibration_factor, num_classes, unseen_classes, temperature_rate):
        calibration_factor_part, calibration_factor_global = calibration_factor
        seen_classes = list(set(range(num_classes))-set(unseen_classes))
        n, _, _ = emb_part.size()
        # construct text adjacent matrix
        text = part_language
        text_con = torch.cat((an.mean(1).unsqueeze(1), text.mean(2)), dim=1)
        A = self.construct_text_adjacent_matrix(text_con)  # 60 6 768
        text_part, text_att_weights = self.text_interaction(text, pool_feature)
        # text GCN
        c, _, _ = text_part.size()
        text_dummy = torch.cat((self.text_dummy.expand(c, -1, -1), text_part), dim=1)
        text_emb = torch.matmul(A, text_dummy)
        text_emb = torch.matmul(A, text_emb)
        text_global = text_emb[:,0,:]
        # condition
        col_idx = torch.arange(self.body_part).unsqueeze(0).expand(n, self.body_part) 
        part_sim = torch.sum(F.normalize(emb_part, dim=2, p=2).unsqueeze(2) * F.normalize(text_part, dim=2, p=2).permute(1, 0, 2).unsqueeze(0), dim=-1)/temperature_rate # n 6 60
        calibration_factor_part_matrix = torch.zeros_like(part_sim)
        calibration_factor_part_matrix[:, :, seen_classes] = calibration_factor_part

        part_sim = torch.softmax(part_sim, dim=2) - calibration_factor_part_matrix
        part_softmax_idx = torch.max(part_sim, dim=2)[1]
        condition = self.condition_project(text_part[part_softmax_idx, col_idx, :].mean(1))
        
        # global
        global_sim = torch.einsum('nk,ck->nc', F.normalize(emb_global+condition, dim=1, p=2), F.normalize(text_global, dim=1, p=2))/temperature_rate
        calibration_factor_global_matrix = torch.zeros_like(global_sim)
        calibration_factor_global_matrix[:, seen_classes] = calibration_factor_global
        global_sim = F.softmax(global_sim, dim=1) - calibration_factor_global_matrix
        global_sim_idx = torch.max(global_sim, dim=1)[1].data.cpu().numpy()
        label = torch.max(label, dim=1)[1].data.cpu().numpy()
        return global_sim_idx, label

    def nce_loss(self, similarity_matrix, labels_one_hot, temperature):
        positive_similarities = torch.sum(similarity_matrix * labels_one_hot, dim=1)  # Shape: (n,)
        exp_sim = torch.exp(similarity_matrix / temperature)
        exp_pos_sim = torch.exp(positive_similarities /temperature)  # Shape: (n,)
        sum_exp_sim = exp_sim.sum(dim=1)  
        loss = -torch.log(exp_pos_sim / sum_exp_sim)
        loss = loss.mean()
        return loss

    





        





  
        
       
        





        