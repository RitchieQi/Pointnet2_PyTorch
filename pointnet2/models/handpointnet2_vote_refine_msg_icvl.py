import pytorch_lightning as pl
import torch
import torch.nn as nn
from pointnet2_ops.pointnet2_modules import PointnetFPModule, PointnetSAModule,jointSAModuleMSG,jointFPModule
from pointnet2_ops import pointnet2_utils

from pointnet2.models.pointnet2_ssg_cls import PointNet2ClassificationSSG
from pointnet2_ops import pointnet2_utils
from handpointnet2_icvl import getPretrainedHandglobal
import torch.nn.functional as F
import numpy as np

def getPretrainedHandglobal_refine():
    net = PointNet2_refine()
    net = net.cuda()
    checkpoint = torch.load('checkpt/jointSAMSG.pth')
    net.load_state_dict(checkpoint['model_state_dict'])
    return net

def get_model():
    return PointNet2_refine()


class PointNet2_refine(nn.Module):
    def __init__(self):
        super().__init__()
        self.refinesample = 128
        
        self._build_model()
    def _build_model(self):
        self.handGloballayer = getPretrainedHandglobal()        

        self.jointSA = jointSAModuleMSG(radius=[0.1,0.2,0.4],nsamples=[32,128,256],mlps=[[6,32,32,64],[6,64,64,128],[6,128,256,512]])
        self.jointFP = jointFPModule([128+64+512+9,512,256,128])
        
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
        
            est_centroids = estimate.unsqueeze(-1).view(-1,16,3).contiguous()
        ''' refine part '''
        
        refinePC,refineFeature = self.jointSA(pointcloud,est_centroids)
        newFeature = self.jointFP(refinePC[-2],est_centroids,refinePC[-2],refineFeature)
        
        offsetMap = self.joint_fc_layer(newFeature)        
        
        return refinePC[-2], offsetMap.permute(0,2,3,1),est_centroids,l_xyz