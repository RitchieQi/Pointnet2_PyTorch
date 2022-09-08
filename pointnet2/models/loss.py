import torch
import torch.nn as nn


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
        
        ofstMap = target.view(-1,16,3).unsqueeze(2).expand(-1,-1,nPointcloud.size(2),-1)
        ofstTarget = nPointcloud[...,:3] - ofstMap

        l1Loss = pred - ofstTarget
        absL1Loss = torch.abs(l1Loss)
        #B,21,N,3
        sumLoss = torch.sum(absL1Loss.view(-1,16*absL1Loss.size(2),3),2)
        mean = torch.mean(sumLoss)
        return mean



        