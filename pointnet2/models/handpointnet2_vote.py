import pytorch_lightning as pl
import torch
import torch.nn as nn
from pointnet2_ops.pointnet2_modules import PointnetFPModule, PointnetSAModule

from pointnet2.models.pointnet2_ssg_cls import PointNet2ClassificationSSG

import faiss
def get_model():
    return PointNet2SemSegSSG()

class PointNet2SemSegSSG(nn.Module):
    def __init__(self):
        super().__init__()
        self.res = faiss.StandardGpuResources()
        self.index_flat = faiss.IndexFlatL2(3)
        self._build_model()
    def _build_model(self):
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=512,
                radius=0.1,
                nsample=64,
                mlp=[3, 32, 32, 64],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=256,
                radius=0.2,
                nsample=64,
                mlp=[64, 64, 64, 128],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=64,
                radius=0.4,
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
        
        self.refineSA = nn.ModuleList()
        self.refineMLP = nn.ModuleList()
        self.refine_fc_layer = nn.Sequential(
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(64, 3),
        )
        for _ in range(21):
            self.refineSA.append(
                PointnetSAModule(
                mlp=[3, 64, 128, 512], use_xyz=True
                )
            )
            self.refineMLP.append(self.refine_fc_layer)
            

#         self.FP_modules = nn.ModuleList()
#         self.FP_modules.append(PointnetFPModule(mlp=[128 + 6, 128, 128, 128]))
#         self.FP_modules.append(PointnetFPModule(mlp=[256 + 64, 256, 128]))
#         self.FP_modules.append(PointnetFPModule(mlp=[256 + 128, 256, 256]))
#         self.FP_modules.append(PointnetFPModule(mlp=[1024 + 256, 256, 256]))

#         self.fc_lyaer = nn.Sequential(
#             nn.Conv1d(256, 256, kernel_size=1, bias=False),
#             nn.BatchNorm1d(256),
#             nn.ReLU(True),
#             nn.Dropout(0.5),
#             nn.Conv1d(256, 4, kernel_size=1),
#         )
        
        self.fc_layer = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(128, 63),
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
        for i in range(len(self.SA_modules)):

            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)
        
        #4* SA module 
        #B*N*1
        
        estimate = self.fc_layer(l_features[-1].squeeze(-1))
        #B*63
        
        est_centroids = estimate.unsqueeze(-1).view(estimate.size(0),21,3).contiguous()
        #B*21*3
        index= [] # B*21*32

        for i,centroids in enumerate(est_centroids):
            gpu_index_flat = faiss.index_cpu_to_gpu(self.res, 0, self.index_flat)
            gpu_index_flat.add(pointcloud[i][:,:3].cpu().numpy()) # pointcloud(1024,3)
            D, I = gpu_index_flat.search(centroids.contiguous().cpu().detach().numpy(), 32) #centroids(n,3) I(n,64)
            index.append(torch.tensor(I))
            del gpu_index_flat
        
        index_t = torch.stack(index).unsqueeze(-1).expand(B,21,32,6).cuda()
        new_pc = torch.gather(pointcloud.unsqueeze(1).expand(B,21,1024,6),2,index_t)
        new_xyz = new_pc[..., 0:3].contiguous()
        new_feature = new_pc[..., 3:].transpose(2, 3).contiguous()
        l_nfeat =[]
        for i in range(len(self.refineSA)):

            li_nxyz,li_nfeat = self.refineSA[i](new_xyz[:,i,:,:],new_feature[:,i,:,:])
            
            li_nfeat = self.refineMLP[i](li_nfeat.squeeze())
            
            l_nfeat.append(li_nfeat)
        
            
        
        return estimate.unsqueeze(-1).view(B,21,3),torch.stack(l_nfeat).permute(1,0,2) #self.fc_lyaer(l_features[0])

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

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
   
if __name__ == '__main__':
    from MSRAhand_dataset import MSRAhand_n
    print('load')
    train_dataset = MSRAhand_n(task = 'train')
    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0, drop_last=True)
    inputs,label = next(iter(trainDataLoader))
    net = get_model()
    net = net.cuda()
    net = net.train()
    inputs = inputs.cuda()
    #print(inputs.size())
    pred = net(inputs)
    print(pred.size())
    