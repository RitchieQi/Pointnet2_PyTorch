from torch.utils.data import Dataset
import pickle
import os
from open3d import io as o3dio
from open3d import geometry as o3dg
from open3d import utility as o3du
import numpy as np
import torch
from datapreprocess import data_process_wPhase2
from tqdm import tqdm
osp = os.path


class ICVLHand():
    def __init__(self,task):
        super(ICVLHand, self).__init__()
        datadir = '/home/liyuan/HandEstimation/dataset/ICVL_16Ver_1024'
        if task == 'train':
            self.pcddir = osp.join(datadir,'train')
        if task == 'test':
            self.pcddir = osp.join(datadir,'test')

        self.labeldir = osp.join(self.pcddir,'label.pkl')
        with open(self.labeldir,'rb') as fl:
            self.label_ = pickle.load(fl)
        self.len = len(self.label_)
    
    def totallen(self):
        return self.len

    def getitem(self,index):
        pcd_name = '%i.pcd'%index
        pcd_label = self.label_[index]
        pcd = o3dio.read_point_cloud(osp.join(self.pcddir,pcd_name))
        points = np.asarray(pcd.points)
        # current_points = torch.from_numpy(points.copy()).float()/1000
        # current_labels = torch.flatten(torch.from_numpy(pcd_label.copy()).float())
        labels = o3dg.PointCloud()
        labels.points = o3du.Vector3dVector(pcd_label)
        points,target= data_process_wPhase2(pcd,labels)
        

        return points,target

if __name__ == '__main__':
    ICVL = ICVLHand('train')
    points = []
    label = []
    newdir = '/home/liyuan/HandEstimation/dataset/ICVL_16Ver_nor/train'
    for _,idx in tqdm(enumerate(range(ICVL.totallen())),total =ICVL.totallen(),smoothing=0.9):
        pc,lb = ICVL.getitem(idx)
        points.append(pc)
        label.append(lb)
    with open(osp.join(newdir,'data.pkl'),'wb') as fp:
        pickle.dump(points,fp)
    with open(osp.join(newdir,'label.pkl'),'wb') as fp:
        pickle.dump(label,fp)


    ICVL = ICVLHand('test')
    points = []
    label = []
    newdir = '/home/liyuan/HandEstimation/dataset/ICVL_16Ver_nor/test'
    for _,idx in tqdm(enumerate(range(ICVL.totallen())),total =ICVL.totallen(),smoothing=0.9):
        pc,lb = ICVL.getitem(idx)
        points.append(pc)
        label.append(lb)
    with open(osp.join(newdir,'data.pkl'),'wb') as fp:
        pickle.dump(points,fp)
    with open(osp.join(newdir,'label.pkl'),'wb') as fp:
        pickle.dump(label,fp)