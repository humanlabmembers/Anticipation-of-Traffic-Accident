from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys




class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)



class DilatedConv2D(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=3, dilation_rates=[1, 2, 4],device='cuda'):
        super(DilatedConv2D, self).__init__()
        self.device='cuda'
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(
                in_channels=input_dim if i == 0 else output_dim,
                out_channels=output_dim,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=(kernel_size - 1) * dilation // 2
            ) for i, dilation in enumerate(dilation_rates)
        ])

        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x):

        identity = x  
        x = x.permute(0, 2, 1)  

        for conv in self.conv_layers:
            x = F.relu(conv(x))  

        x = x.permute(0, 2, 1)  
        x = self.norm(x + identity)
        return x
    
class AdaptiveAdjacencyMatrix(nn.Module):
    def __init__(self, num_nodes, num_timesteps, batch_size, supports=None, addaptadj=True, aptinit=None, device='cuda'):
        super(AdaptiveAdjacencyMatrix, self).__init__()

        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)
            self.supports = supports
        else:
            self.supports = []
        
        self.num_timesteps = num_timesteps
        self.batch_size = batch_size

        if addaptadj:
            if aptinit is None:
                self.nodevec1 = nn.Parameter(torch.randn(batch_size, num_nodes, 10).to(device)* 0.1, requires_grad=True)
                self.nodevec2 = nn.Parameter(torch.randn(batch_size, 10, num_nodes).to(device)* 0.1, requires_grad=True)
                self.supports_len += 1
            else:
                m, p, n = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1.unsqueeze(0).repeat(batch_size, 1, 1), requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(initemb2.unsqueeze(0).repeat(batch_size, 1, 1), requires_grad=True).to(device)
                self.supports_len += 1
    
    def forward(self, x):
        batch_size, num_timesteps, num_nodes, feature_dim = x.shape

        adaptive_adj = torch.bmm(self.nodevec1, self.nodevec2)  # [batch_size, num_nodes, num_nodes]
        adaptive_adj = adaptive_adj.unsqueeze(1).repeat(1, num_timesteps, 1, 1)  # [batch_size, num_timesteps, num_nodes, num_nodes]
        
        if len(self.supports) > 0:
            adj_list = [adaptive_adj] + self.supports
            adj = torch.cat(adj_list, dim=0)  # [support_len + 1, batch_size, num_timesteps, num_nodes, num_nodes]
        else:
            adj = adaptive_adj

        return adj












def normalize_depth(depth):
    return depth / 255.0

def compute_centers(detection):
    return (detection[..., :2] + detection[..., 2:4]) / 2

def compute_depth_values(depth_normalized, centers, batch_size, num_frames, num_objects):
    x_coords = (centers[:, :, :, 0] / 10).astype(int)  
    y_coords = (centers[:, :, :, 1] / 10).astype(int)  

    depth_values = depth_normalized[np.arange(batch_size)[:, None, None], 
                                    np.arange(num_frames)[None, :, None], 
                                    y_coords, x_coords]
    
    return depth_values

def compute_distances(centers, depth_values, batch_size, num_frames, num_objects):
    if isinstance(centers, np.ndarray):
        centers = torch.tensor(centers)

    if isinstance(depth_values, np.ndarray):
        depth_values = torch.tensor(depth_values)

    centers = centers.to('cpu')
    depth_values = depth_values.to('cpu')

    distances = torch.zeros((batch_size, num_frames, num_objects, num_objects), device='cpu')

    centers_diff = centers[:, :, :, None, :] - centers[:, :, None, :, :]
    euclidean_dist = torch.norm(centers_diff, dim=-1) / 1450

    depth_diff = torch.abs(depth_values[:, :, :, None] - depth_values[:, :, None, :])

    euclidean_dist = torch.clamp(euclidean_dist, min=0)
    depth_diff = torch.clamp(depth_diff, min=0)
    distances = torch.sqrt(euclidean_dist ** 2 + depth_diff ** 2)

    return distances



def compute_velocities(distances, batch_size, num_frames, num_objects):
    velocities = distances[:, 1:, :, :] - distances[:, :-1, :, :]
    
    velocities = np.pad(velocities, ((0, 0), (1, 0), (0, 0), (0, 0)), mode='constant', constant_values=0)
    
    return velocities


def normalize(arr):
    min_val = torch.min(arr)
    max_val = torch.max(arr)

    if max_val - min_val == 0:
        return torch.zeros_like(arr)  
    normalized = (arr - min_val) / (max_val - min_val)
    
    return normalized


def vel_norm(vel):
    vel_norm = np.zeros_like(vel)

    neg_mask = vel < 0
    if np.any(neg_mask):
        vel_norm[neg_mask] = 0

    pos_mask = vel > 0
    if np.any(pos_mask):
        vel_pos = vel[pos_mask]
        min_pos, max_pos = np.min(vel_pos), np.max(vel_pos)
        if min_pos != max_pos:
            vel_norm[pos_mask] = 0 + (vel_pos - min_pos) * (1 - 0) / (max_pos - min_pos)
        else:
            vel_norm[pos_mask] = 1  

    zero_mask = vel == 0
    vel_norm[zero_mask] = 0

    return vel_norm

def compute_weights(distances, velocities):
    normalized_distances = normalize(distances)
    normalized_velocities = vel_norm(velocities)
    weights = 0.8 * (np.exp(-normalized_distances)) + 0.2 * normalized_velocities
    weights=normalize(weights)
    return weights


def clear_and_rearrange(detection):
    num_batches, num_frames, num_detections, num_features = detection.shape
    
    new_detection = np.zeros((num_batches,num_frames, num_detections, num_features))
    
    for batch in range(num_batches):
        for frame in range(num_frames):
            batch_data = detection[batch,frame]
            
            mask = batch_data[:, 4] != 2
            
            valid_data = batch_data[mask]
            
            num_valid = valid_data.shape[0]
            
            new_detection[batch,frame, :num_valid, :] = valid_data.cpu().numpy()
    
    return new_detection

class weights(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,detection,depth):
        batch_size,num_frames, num_objects, _ = detection.shape
        detection=clear_and_rearrange(detection) 
        depth_normalized = normalize_depth(depth) 
        centers = compute_centers(detection) 

        depth_values = compute_depth_values(depth_normalized, centers,batch_size, num_frames, num_objects)
        distances = compute_distances(centers, depth_values, batch_size, num_frames, num_objects)
        velocities = compute_velocities(distances,batch_size, num_frames, num_objects)
        graphweights = compute_weights(distances, velocities)
        
        return graphweights








class WeightedGCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(WeightedGCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False).to('cuda')

    def forward(self, x, adj, weights):
   
        adj_normalized = self.normalize_adjacency_matrix(adj).to('cuda')
        x_transformed = self.linear(x).float()  
        weights=weights.to('cuda')

        weighted_adj = adj_normalized * weights
        weighted_adj=weighted_adj.float()
        out = torch.matmul(weighted_adj, x_transformed)  
        return out

    def normalize_adjacency_matrix(self, adj):
       
        I = torch.eye(adj.size(-1)).to(adj.device)  
        adj = adj + I 
        degree_matrix = adj.sum(dim=-1, keepdim=True)  
        degree_matrix_inv_sqrt = degree_matrix.pow(-0.5)  # D^(-1/2)
        degree_matrix_inv_sqrt[torch.isinf(degree_matrix_inv_sqrt)] = 0  
        adj_normalized = degree_matrix_inv_sqrt * adj * degree_matrix_inv_sqrt.transpose(-1, -2)
        return adj_normalized

class DynamicWeightedGCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_layers):
        super(DynamicWeightedGCN, self).__init__()
        self.num_layers = num_layers

        self.gcn_layers = nn.ModuleList([WeightedGCNLayer(in_features if i == 0 else hidden_features, hidden_features) for i in range(num_layers)])

        self.lstm = nn.LSTM(input_size=hidden_features, hidden_size=hidden_features, batch_first=True)

        self.fc = nn.Linear(hidden_features, out_features)

    def forward(self, x, adj, weights):
        batch_size, num_timesteps, num_nodes, in_features = x.size()

        gcn_outputs = []

        for t in range(num_timesteps):
            xt = x[:, t, :, :]  
            adjt = adj[:, t, :, :]  
            weightt = weights[:, t, :, :]  
            for layer in self.gcn_layers:
                xt = F.relu(layer(xt, adjt, weightt))  
            gcn_outputs.append(xt)

        gcn_output = torch.stack(gcn_outputs, dim=1)  # [batch_size, num_timesteps, num_nodes, hidden_features]

        gcn_output = gcn_output.mean(dim=2)  # [batch_size, num_timesteps, hidden_features]

        lstm_output, _ = self.lstm(gcn_output)  #  [batch_size, num_timesteps, hidden_features]

        out = self.fc(lstm_output)  # [batch_size, num_timesteps, out_features]

        return out

class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=2, n_layers=1, dropout=None):
        super(GRUNet, self).__init__()
        if dropout is None:
            dropout = [0.5, 0.2]
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True)
        self.dropout = dropout
        self.dense1 = torch.nn.Linear(hidden_dim, 64)
        self.dense2 = torch.nn.Linear(64, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        out, h = self.gru(x)
        out = F.dropout(out, self.dropout[0])
        out = self.relu(self.dense1(out))
        out = F.dropout(out, self.dropout[1])
        out = self.dense2(out)
        return out
    
class MYModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.device=torch.device('cuda')

        class_weights = torch.tensor([1,5], dtype=torch.float32)
        self.ce_loss = torch.nn.CrossEntropyLoss(weight=class_weights)
        self.dilatedconv=DilatedConv2D(1024,1024)
        self.getWeights=weights()
        self.adj=AdaptiveAdjacencyMatrix(19,100,10).to(self.device)
        self.GCN=DynamicWeightedGCN(512, 256, 512, 2).to(self.device) #DynamicWeightedGCN(in_features, hidden_features, out_features, num_layers)
        self.GRU_net = GRUNet(input_dim=1024, hidden_dim=256).to(self.device)
        
        
    def forward(self, features,label,toa,detection,depth):
        if torch.isnan(features).any() or torch.isinf(features).any():
            raise ValueError("NaN or Inf found in features!!.")
        visual_features = features.to('cuda')        
        visual_features = visual_features.float()
        visual_features =LayerNorm(512).to(self.device)(visual_features)

        obj_features=visual_features[:,:,1:,:]
        frame_features=visual_features[:,:,1,:]
        adj=self.adj(obj_features)
        adj=torch.where(torch.isnan(adj), torch.tensor(1e-10, dtype=adj.dtype), adj)

        graphweights=self.getWeights(detection,depth)
       

        feature2=self.GCN(obj_features,adj,graphweights)
        feature2=torch.where(torch.isnan(feature2), torch.tensor(1e-10, dtype=feature2.dtype), feature2)
        feature2= torch.cat((feature2,frame_features),dim=-1)
        feature2=self.dilatedconv(feature2)
        out = self.GRU_net(feature2)
        out=torch.where(torch.isnan(out), torch.tensor(1e-10, dtype=feature2.dtype), out)


        losses = {'cross_entropy': 0,
                  'L4':0,
                  'L5':0,
                  'total_loss': 0}
        
        L4=torch.tensor(float(0))
        L3=0
        L5=torch.tensor(float(0))

        

        for t in range(visual_features.size(1)):
            L3 += self._exp_loss(out[:, t, :], label, t, toa=toa)
            
        
        losses['cross_entropy'] += L3
        losses['L4'] += L4
        losses['L5'] += L5
            
            
            

        return losses,out
    
    def compute_mil_loss(self,out, label, device):
        batch_size, num_instances, num_classes = out.shape
        
        labels = label / torch.sum(label, dim=1, keepdim=True)
        labels = labels.to(device)
        
        instance_logits = torch.zeros(0).to(device)
        
        for i in range(batch_size):
            tmp, _ = torch.topk(out[i], k=20, largest=True, dim=0)
            instance_logits = torch.cat([instance_logits, torch.mean(tmp, dim=0, keepdim=True)], dim=0)
        
        milloss = -torch.mean(torch.sum(labels * F.log_softmax(instance_logits, dim=1), dim=1), dim=0)
        return milloss





    def _exp_loss(self, pred, target, time, toa, fps=10):
        '''
        :param pred:
        :param target: onehot codings for binary classification
        :param time:
        :param toa:
        :param fps:
        :return:
        '''
        target_cls = target[:, 1]
        target_cls = target_cls.to(torch.long)
        penalty = -torch.max(torch.zeros_like(toa).to(toa.device, pred.dtype), (toa.to(pred.dtype) - time - 1) / fps).to(self.device)
        pos_loss = -torch.mul(torch.exp(penalty), -self.ce_loss(pred, target_cls))
        neg_loss = self.ce_loss(pred, target_cls)
        
        loss = torch.mean(torch.add(torch.mul(pos_loss, target[:, 1]), torch.mul(neg_loss, target[:, 0])))
        return loss



def load_checkpoint(model, optimizer=None, filename='checkpoint.pth.tar', isTraining=True):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model'])
        if isTraining:
            optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer, start_epoch
