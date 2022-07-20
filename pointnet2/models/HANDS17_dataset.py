# author: LiyuanQi

from torch.utils.data import Dataset
import pickle
import os
import numpy as np
from natsort import natsorted, ns
# from open3d import io as o3dio
# from datapreprocess import data_process_wPhase2
# from open3d import io as o3dio
# from open3d import geometry as o3dg
# from open3d import utility as o3du
# from open3d import t as o3dt
osp = os.path
import torch
from tqdm import tqdm


class HANDS17():
    def __init__(self,n_sample,task):
        super(HANDS17,self).__init__()
        if n_sample == 1024:
            if task == 'train':
                self.datadir = '/home/liyuan/HandEstimation/dataset/HANDS17_data_1024/train'
            if task == 'test':
                self.datadir = '/home/liyuan/HandEstimation/dataset/HANDS17_data_1024/test'
        if n_sample == 2048:
            if task == 'train':
                self.datadir = '/home/liyuan/HandEstimation/dataset/HANDS17_data_2048/train'
            if task == 'test':
                self.datadir = '/home/liyuan/HandEstimation/dataset/HANDS17_data_2048/test'
        
        self.len = len(os.listdir(self.datadir))-1
        print(self.len)
        ''' get label '''
        self.labeldir = osp.join(self.datadir,'label.pkl')
        with open(self.labeldir, 'rb') as fl:
            self.label = pickle.load(fl)
        ''' get file list '''
        self.filelist = os.listdir(self.datadir)
        self.filelist.remove('label.pkl')
        self.filelist = natsorted(self.filelist, key=lambda y: y.lower())
        self.hands2msra = np.array([0,2,9,10,11,3,12,13,14,4,15,16,17,5,18,19,20,1,6,7,8])

    def totallen(self):
        return self.len

    def getitem(self,index):
        point_name = self.filelist[index]
        idx = point_name.split('.')
        label_key = idx[0]+'.png'
        pcd = o3dio.read_point_cloud(osp.join(self.datadir,point_name))
        label_hands = np.asarray(self.label[label_key])
        labels = label_hands[self.hands2msra]#/1000
        # points = np.asarray(pcd.points)/1000
        # current_points = torch.from_numpy(points.copy()).float()
        # current_labels = torch.flatten(torch.from_numpy(labels.copy()).float())
        points,target= data_process_wPhase2(pcd,labels)
        return points,target

class Hands17data(Dataset):
    def __init__(self,task):
        super(Hands17data,self).__init__()
        if task == 'train':
            self.dir = '/home/liyuan/HandEstimation/model/P2Preg/HANDS_data/train'
        if task == 'test':
            self.dir = '/home/liyuan/HandEstimation/model/P2Preg/HANDS_data/test'
        labeldir = osp.join(self.dir,'label.pkl')
        with open(labeldir,'rb') as fl:
            self.label = pickle.load(fl)
        self.len = len(self.label)

    def __len__(self):
        return self.len
    
    def __getitem__(self,index):
        label = self.label[index]
        pointname = str(index)+'.pkl'
        pointdir = osp.join(self.dir,pointname)
        with open(pointdir,'rb') as datafile:
            points = pickle.load(datafile)
        return points,label





# if __name__ == '__main__':
#     newdir = '/home/liyuan/HandEstimation/model/P2P-reg/HANDS_data/test'
#     hands = HANDS17(n_sample=1024,task = 'test')
#     label = []
#     for _,idx in tqdm(enumerate(range(hands.totallen())),total =hands.totallen(),smoothing=0.9):
#         pc,lb = hands.getitem(idx)
#         label.append(lb)
#         name = str(idx)+'.pkl'
#         with open(osp.join(newdir,name),'wb') as fp:
#             pickle.dump(pc,fp)
    
#     with open(osp.join(newdir,'label.pkl'),'wb') as fp:
#         pickle.dump(label,fp)
