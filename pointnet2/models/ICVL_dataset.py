from torch.utils.data import Dataset
import pickle
import os
from open3d import io as o3dio
import numpy as np
import torch
osp = os.path


class ICVLHand(Dataset):
    def __init__(self,task):
        super(ICVLHand, self).__init__()
        datadir = '/home/liyuan/HandEstimation/dataset/ICVL_data_1024'
        if task == 'train':
            self.pcddir = osp.join(datadir,'train')
        if task == 'test':
            self.pcddir = osp.join(datadir,'test')

        self.labeldir = osp.join(self.pcddir,'label.pkl')
        with open(self.labeldir,'rb') as fl:
            self.label_ = pickle.load(fl)
        self.len = len(self.label_)
    
    def __len__(self):
        return self.len

    def __getitem__(self,index):
        
        if index == 7065: #wrongly labled frame
            index = 7066

        pcd_name = '%i.pcd'%(index+1)
        pcd_label = self.label_[str(index+1)]
        pcd = o3dio.read_point_cloud(osp.join(self.pcddir,pcd_name))
        points = np.asarray(pcd.points)
        current_points = torch.from_numpy(points.copy()).float()
        current_labels = torch.flatten(torch.from_numpy(pcd_label.copy()).float())
        return current_points,current_labels

class ICVLHand_16Ver(Dataset):
    def __init__(self,task):
        super(ICVLHand_16Ver, self).__init__()
        datadir = '/home/liyuan/HandEstimation/dataset/ICVL_16Ver_1024'
        if task == 'train':
            self.pcddir = osp.join(datadir,'train')
        if task == 'test':
            self.pcddir = osp.join(datadir,'test')

        self.labeldir = osp.join(self.pcddir,'label.pkl')
        with open(self.labeldir,'rb') as fl:
            self.label_ = pickle.load(fl)
        self.len = len(self.label_)
    
    def __len__(self):
        return self.len

    def __getitem__(self,index):
        

        pcd_name = '%i.pcd'%(index)
        pcd_label = self.label_[index]
        pcd = o3dio.read_point_cloud(osp.join(self.pcddir,pcd_name))
        points = np.asarray(pcd.points)
        current_points = torch.from_numpy(points.copy()).float()
        current_labels = torch.flatten(torch.from_numpy(pcd_label.copy()).float())
        return current_points,current_labels

class ICVL_n(Dataset):
    def __init__(self,task):
        super(ICVL_n, self).__init__()
        if task == 'train':
            dir = '/home/liyuan/HandEstimation/dataset/ICVL_1024_nor/train'
        if task == 'test':
            dir = '/home/liyuan/HandEstimation/dataset/ICVL_1024_nor/test'
        
        datadir = osp.join(dir,'data.pkl')
        labeldir = osp.join(dir,'label.pkl')
        with open(datadir,'rb') as fd:
            self.data = pickle.load(fd)
        with open(labeldir,'rb') as fl:
            self.label = pickle.load(fl)
        self.len = len(self.data)
    def __len__(self):
        return self.len
    
    def __getitem__(self,index):
        return self.data[index],self.label[index]

class ICVL_16Ver_n(Dataset):
    def __init__(self,task):
        super(ICVL_16Ver_n, self).__init__()
        if task == 'train':
            dir = '/home/liyuan/HandEstimation/dataset/ICVL_16Ver_nor/train'
        if task == 'test':
            dir = '/home/liyuan/HandEstimation/dataset/ICVL_16Ver_nor/test'
        
        datadir = osp.join(dir,'data.pkl')
        labeldir = osp.join(dir,'label.pkl')
        with open(datadir,'rb') as fd:
            self.data = pickle.load(fd)
        with open(labeldir,'rb') as fl:
            self.label = pickle.load(fl)
        self.len = len(self.data)
    def __len__(self):
        return self.len
    
    def __getitem__(self,index):
        return self.data[index],self.label[index]
if __name__ == '__main__':
    from torch.utils.data import DataLoader
    data = ICVLHand_16Ver('train')
    dloader = DataLoader(data, batch_size = 128, shuffle=True)
    inputs, labels = next(iter(dloader))
    # torch.save(inputs,'inputs.pt')
    # torch.save(labels,'labels.pt')
    print(inputs[0],labels[0])