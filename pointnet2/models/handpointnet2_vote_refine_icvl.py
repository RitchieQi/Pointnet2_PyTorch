import pytorch_lightning as pl
import torch
import torch.nn as nn
from pointnet2_ops.pointnet2_modules import PointnetFPModule, PointnetSAModule,jointSAModule,jointFPModule
from pointnet2_ops import pointnet2_utils

from pointnet2.models.pointnet2_ssg_cls import PointNet2ClassificationSSG
from pointnet2_ops import pointnet2_utils
from handpointnet2_icvl import getPretrainedHandglobal
import torch.nn.functional as F
import numpy as np

def get_model():
    return PointNet2_refine()


class PointNet2_refine(nn.Module):
    def __init__(self):
        super().__init__()
        self.refinesample = 128
        
        self._build_model()
    def _build_model(self):
        self.handGloballayer = getPretrainedHandglobal()        

        self.jointSA = jointSAModule(radius=0.2,nsamples=128,mlp=[6,128,256,512])
        self.jointFP = jointFPModule([512+9,256,128,128])
        
        self.joint_fc_layer = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 64, kernel_size=1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv2d(64, 3, kernel_size=1)
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
        with torch.no_grad():
            estimate,l_xyz = self.handGloballayer(pointcloud)

            #B*63
            est_centroids = estimate.unsqueeze(-1).view(estimate.size(0),16,3).contiguous()
        ''' refine part '''
        
        refinePC,refineFeature = self.jointSA(pointcloud,est_centroids)
        newFeature = self.jointFP(refinePC,est_centroids,refinePC,refineFeature)
        
        offsetMap = self.joint_fc_layer(newFeature)        
        
        return refinePC, offsetMap.permute(0,2,3,1),est_centroids,l_xyz