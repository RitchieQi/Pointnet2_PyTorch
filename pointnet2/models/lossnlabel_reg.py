import torch
import torch.nn as nn

class get_offsetmap(nn.Module):
    # target B*21*3
    # pointcloud B*1024*6
    def __init__(self):    
        super(get_offsetmap,self).__init__() 

    def torch_knn(self,xq, xb, k): 

        norms_xq = (xq ** 2).sum(axis=2)
        
        norms_xb = (xb ** 2).sum(axis=2)
        #print(xq.size(),xb.T.size(),norms_xq.size(),norms_xb.size())
        temp = norms_xq.view(norms_xq.size(0),-1, 1).permute(1,0,2) + norms_xb
        #print(temp.size())
        distances = temp.permute(1,0,2) -2 * xq @ xb.permute(0,2,1)
        return torch.topk(distances, k, largest=False)
    def forward(self,pointcloud,target):
        pointcloud = pointcloud[...,:3]
        
        #B,_,_ = pointcloud.size(0)
        D, I = self.torch_knn(target,pointcloud,64)
        subPointcloud = torch.gather(pointcloud.unsqueeze(1).expand(-1,21,1024,3),2,I.unsqueeze(-1).expand(-1,21,64,3))
        pointcloud = pointcloud.unsqueeze(1).expand(-1,21,1024,3)
        
        allZeroPC = torch.zeros_like(pointcloud)
        #print(pointcloud.size(),I.size())
        mask = allZeroPC.scatter_(2,I.unsqueeze(-1).expand(-1,21,64,3),1)
        
        mask_label = mask * pointcloud
        print(mask_label)
        return mask_label#subPointcloud# 

        









            








class ofstL1Loss(nn.Module):
    def __init__(self, margin=0.1):
        super(ofstL1Loss, self).__init__()
    
    def forward(self, target,nPointcloud,pred):
        r'''
        parameter:
        target: B,21,3
        nPointcloud: B,21,N,6
        pred: B,21,N,3
        '''
        
        ofstMap = target.view(-1,21,3).unsqueeze(2).expand(-1,-1,nPointcloud.size(2),-1)
        ofstTarget = nPointcloud[...,:3] - ofstMap

        l1Loss = pred - ofstTarget
        absL1Loss = torch.abs(l1Loss)
        #B,21,N,3
        sumLoss = torch.sum(absL1Loss.view(-1,21*absL1Loss.size(2),3),2)
        mean = torch.mean(sumLoss)
        return mean



if __name__ == '__main__':
    pass
