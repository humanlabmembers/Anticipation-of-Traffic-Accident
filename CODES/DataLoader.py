import itertools
import os
import time
import networkx
import numpy as np
import torch.nn as nn
import pickle
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch_geometric
from torch_geometric.nn import GCNConv,GATConv
from torch_scatter import scatter
from torch_geometric.typing import Adj, OptTensor, Size
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch import Tensor
from torch.nn import Parameter
import warnings


root_path=your_path

frames=100
n_obj=19

class DADDataset(Dataset):
    def __init__(self, root_path=root_path, feature='dad', phase='training', toTensor=True, device=torch.device('cuda')):
        self.feature_path = os.path.join("/home/jvzi/jv/VLM_based", 'data', 'vgg16_features_512' ,feature,phase)
        self.feature_files = self.get_filelist(self.feature_path)
        self.toTensor=toTensor
        self.device=device
        self.det_path=os.path.join("/home/jvzi/jv/VLM_based", 'data', 'deta_obj_features' ,feature,phase)
        self.depth_path=os.path.join("/home/jvzi/jv/VLM_based", 'data', 'reduced_depth' ,feature,phase)
    def get_filelist(self, featurefilepath):
        assert os.path.exists(featurefilepath), "Directory does not exist: %s"%(featurefilepath)
        file_list = []
        for filename in sorted(os.listdir(featurefilepath)):
            file_list.append(filename)
        return file_list
    
    def __len__(self):
        data_len = len(self.feature_files)
        return data_len
    
    def __getitem__(self, index):
        
        data_file = os.path.join(self.feature_path,self.feature_files[index])
        
        try:
            # Load feature data file
            data = np.load(data_file,allow_pickle=True) 
            labels = torch.tensor(int(data['label']))
            labels = labels.long()
            labels = torch.nn.functional.one_hot(labels, num_classes=2)
            ID=data_file.split('_')[-1].split('.')[0]
           
        except:
            raise IOError('Load data error! File: %s' % (data_file))

        # Load det file
        if labels[1] > 0:
            toa = [90]
            dir = 'positive'
        else:
            toa = [101]
            dir = 'negative'

        IDs=str(ID)+'.npz'

        det_file = os.path.join(self.det_path,dir,IDs)
        detection_data=np.load(det_file)
        detection=detection_data['det']
        detection= np.array(detection)
        detection = torch.Tensor(detection).to(self.device)

        depth_file = os.path.join(self.depth_path,dir,IDs)
        depth_data=np.load(depth_file)
        depth=depth_data['depth_data']
        depth = np.array(depth)
        depth = torch.Tensor(depth).to(self.device)

        frame_features=data['ffeat']
        obj_features=data['features']
        frame_features= torch.tensor(frame_features).to(self.device)
        frame_features=frame_features.unsqueeze(1)
        obj_features= torch.tensor(obj_features).to(self.device)
        
        features=torch.concat((frame_features,obj_features), dim=1)

        if self.toTensor:
            
            labels = np.array(labels)
            toa = np.array(toa)  
            # Convert to tensors            
            labels = torch.tensor(labels).to(self.device)
            toa = torch.Tensor(toa).to(self.device)
                       

        return features, labels, toa,detection,depth
    

