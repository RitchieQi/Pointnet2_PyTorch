
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_ops.pointnet2_modules import PointnetSAModule, PointnetSAModuleMSG
from pointnet2.models.pointnet2_ssg_cls import PointNet2ClassificationSSG
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_sched
from torch.utils.data import Dataset,random_split
import numpy as np


def get_model():
    return PointNet2ClassificationSSG()

class PointNet2ClassificationSSG(nn.Module):
    def __init__(self):
        super().__init__()

        self.skeleton = torch.from_numpy(np.array([
                                     [0, -0.15, -0.15, -0.15, -0.15,  0,     0,     0,     0,    0.175, 0.175, 0.175, 0.175, 0.3,  0.3,  0.3,  0.3,  -0.12,  -0.12,  -0.12,  -0.12,  ],
                                     [0,  0.26,  0.4,   0.5,   0.6,   0.33,  0.49,  0.59,  0.69, 0.3,   0.45,  0.55,  0.65,  0.175,0.275, 0.35, 0.425,  0.07,  0.2,  0.3,  0.4]], dtype=np.float32)).cuda()
        self._build_model()

    def _build_model(self):
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=512,
                radius=0.1,
                nsample=64,
                mlp=[3, 64, 64, 128],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=128,
                radius=0.2,
                nsample=64,
                mlp=[128, 128, 128, 256],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                mlp=[256, 256, 512, 1024], use_xyz=True
            )
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(1024, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, 63),
        )
        
        self.netFolding1_1 = nn.Sequential(
            # B*1024
            nn.Conv2d(2+1024, 256, kernel_size=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # B*1024
            nn.Conv2d(256, 256, kernel_size=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # B*512
            nn.Conv2d(256, 128, kernel_size=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # nn.Linear(nstates_plus_3[4], self.num_outputs),
            # B*num_outputs
        )

        self.netFolding1_2 = nn.Sequential(
            nn.Conv2d(128, 3, kernel_size=(1, 1)),
        )
        self.netFolding2_1 = nn.Sequential(
            # B*131*sample_num_level2*knn_K
            nn.Conv2d(3+256*3+3+64, 256, kernel_size=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # B*128*sample_num_level2*knn_K
            nn.Conv2d(256, 256, kernel_size=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # B*128*sample_num_level2*knn_K
            nn.Conv2d(256, 256, kernel_size=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # B*256*sample_num_level2*knn_K
            nn.MaxPool2d((64,1),stride=1)
        )


        self.netFolding2_2 = nn.Sequential(
            # B*1024
            nn.Conv2d(256, 256, kernel_size=(1, 1)),
            nn.BatchNorm2d( 256),
            # nn.Linear(nstates_plus_3[2], nstates_plus_3[3]),
            # nn.BatchNorm1d(nstates_plus_3[3]),
            nn.ReLU(inplace=True),
            # B*1024
            nn.Conv2d(256,  256, kernel_size=(1, 1)),
            nn.BatchNorm2d(256),
            # nn.Linear(nstates_plus_3[3], nstates_plus_3[4]),
            # nn.BatchNorm1d(nstates_plus_3[4]),
            nn.ReLU(inplace=True),
            # B*512
            nn.Conv2d(256, 3, kernel_size=(1, 1)),
            # nn.Linear(nstates_plus_3[4], self.num_outputs),
            # B*num_outputs

            # B*256*sample_num_level2*1
        )
        self.netFolding3_1 = nn.Sequential(
            # B*131*sample_num_level2*knn_K
            nn.Conv2d(3+256*3+3+128, 256, kernel_size=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # B*128*sample_num_level2*knn_K
            nn.Conv2d(256, 256, kernel_size=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # B*128*sample_num_level2*knn_K
            nn.Conv2d(256,256, kernel_size=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # B*256*sample_num_level2*knn_K
            nn.MaxPool2d((64,1),stride=1)

        )
        self.netFolding3_2 = nn.Sequential(
            # B*1024
            nn.Conv2d(256, 256, kernel_size=(1, 1)),
            nn.BatchNorm2d( 256),
            # nn.Linear(nstates_plus_3[2], nstates_plus_3[3]),
            # nn.BatchNorm1d(nstates_plus_3[3]),
            nn.ReLU(inplace=True),
            # B*1024
            nn.Conv2d(256,  256, kernel_size=(1, 1)),
            nn.BatchNorm2d(256),
            # nn.Linear(nstates_plus_3[3], nstates_plus_3[4]),
            # nn.BatchNorm1d(nstates_plus_3[4]),
            nn.ReLU(inplace=True),
            # B*512
            nn.Conv2d(256, 3, kernel_size=(1, 1)),
            # nn.Linear(nstates_plus_3[4], self.num_outputs),
            # B*num_outputs
            # B*256*sample_num_level2*1
        )
    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, pointcloud):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz, features = self._break_up_pc(pointcloud)

        for module in self.SA_modules:
            xyz, features = module(xyz, features)
        features = features.view(features.size(0),1024,1,1)
        skeleton = self.skeleton.unsqueeze(0).unsqueeze(-2).expand(features.size(0),2, 1, 21)
        code = features.expand(features.size(0),1024, 1, 21)
        x = torch.cat((skeleton, code),1)
        fold1_1 = self.netFolding1_1(x)
        fold1 = self.netFolding1_2(fold1_1)
        
        #print(fold1.size())
        return fold1.squeeze()

  


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target):
        B,_,_ = pred.size()
        batch_output = []
        for i in range(B):
            
            joints = pred[i].permute(1,0)
            label = target[i].view(21,3)
            current_hand = []
            for j in range(21):
                current_hand.append(torch.cdist(joints[j].view(1,3),label[j].view(1,3)))
            current_hand_avg = torch.mean(torch.stack(current_hand))
            batch_output.append(current_hand_avg)
        batch_output_avg = torch.mean(torch.stack(batch_output))
        #total_loss = F.mse_loss(pred, target)

        return batch_output_avg
