import pytorch_lightning as pl
import torch
import torch.nn as nn
from pointnet2_ops.pointnet2_modules import PointnetFPModule, PointnetSAModule
from pointnet2_ops.pointnet2_utils import QuaryandGroupJoints
from pointnet2_ops import pointnet2_utils
from pointnet2.models.pointnet2_ssg_cls import PointNet2ClassificationSSG
import torch.nn.functional as F

def get_model():
    return PointNet2SemSegSSG()

class PointNet2SemSegSSG(nn.Module):
    def __init__(self):
        super().__init__()
        self.refinesample = 64
        self.patchup_idx = torch.Tensor([17,2,3,4,4,6,7,8,8,10,11,12,12,14,15,16,16,18,19,20,20]).type(torch.int64).cuda()
        self.patchdown_idx = torch.Tensor([0,1,1,2,3,5,5,6,7,9,9,10,11,13,13,14,15,0,17,18,19]).type(torch.int64).cuda()
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

        self.quaryJoint = QuaryandGroupJoints()

        self.refineSA = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(True),        
            nn.Conv2d(64, 128, kernel_size=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(True),  
            nn.Conv2d(128, 256, kernel_size=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(True),  
        )
        # MAXPOOL
        # (B,256+3,21,1)
        self.refineFC = nn.Sequential(
            nn.Conv2d(256+3, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),        
            nn.Conv2d(256, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            #nn.Dropout(0.4),    
            nn.Conv2d(128, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Dropout(0.3),  
            nn.Conv2d(128, 3, kernel_size=1),
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
        B,_,_ = pointcloud.size()
        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]
        with torch.no_grad():
            for i in range(len(self.SA_modules)):
                li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
                l_xyz.append(li_xyz)
                l_features.append(li_features)
        
        #B*N*1
            estimate = self.fc_layer(l_features[-1].squeeze(-1))

        #B*63
        
        est_centroids = estimate.unsqueeze(-1).view(estimate.size(0),21,3).contiguous()
        ''' refine part '''
        # #B*21*3
        #with torch.no_grad():
        idx = pointnet2_utils.ball_query(0.1,self.refinesample,xyz,est_centroids).type(torch.int64)

        new_pointcloud = torch.gather(pointcloud.unsqueeze(1).expand(pointcloud.size(0),21,1024,6), 2, idx.unsqueeze(-1).expand(B,21,self.refinesample,6))
        patch_u = torch.gather(new_pointcloud,1,self.patchup_idx.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(B,21,self.refinesample,6))
        patch_d = torch.gather(new_pointcloud,1,self.patchdown_idx.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(B,21,self.refinesample,6))

        new_pointcloud = torch.cat([patch_u,new_pointcloud,patch_d],dim =2)
        n_feat = new_pointcloud.permute(0,3,1,2)

        n_embedding = self.refineSA(n_feat)
        n_embedding =  F.max_pool2d(n_embedding, kernel_size=[1, n_embedding.size(3)])

        new_feat = torch.cat([est_centroids.unsqueeze(-1).permute(0,2,1,3),n_embedding], dim = 1)
        refine =  self.refineFC(new_feat)
        

        
        return est_centroids,refine.squeeze(-1).permute(0,2,1)

class get_dis_loss(nn.Module):
    def __init__(self):
        super(get_dis_loss, self).__init__()

    def forward(self, pred, target):
        #print(pred.size())
        B,_,_ = pred.size()
        batch_output = []
        for i in range(B):
            
            joints = pred[i]
            label = target[i].view(21,3)
            current_hand = []
            for j in range(21):
                current_hand.append(torch.cdist(joints[j].view(1,3),label[j].view(1,3)))
            current_hand_avg = torch.mean(torch.stack(current_hand))
            batch_output.append(current_hand_avg)
        batch_output_avg = torch.mean(torch.stack(batch_output))
        #total_loss = F.mse_loss(pred, target)

        return batch_output_avg

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target):
        total_loss = F.mse_loss(pred.float(), target.unsqueeze(-1).view(target.size(0),21,3).float())
        #total_loss = F.mse_loss(pred, target)

        return torch.sqrt(total_loss)


if __name__ == '__main__':
    # from MSRAhand_dataset import MSRAhand_n
    # print('load')
    # train_dataset = MSRAhand_n(task = 'train')
    # trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0, drop_last=True)
    # inputs,label = next(iter(trainDataLoader))
    net = get_model()
    net = net.cuda()
    net = net.train()
    params = net.state_dict()
    keys = list(params.keys())
    print(keys[1])
    # inputs = inputs.cuda()
    # #print(inputs.size())
    # pred = net(inputs)
    # print(pred.size())
    