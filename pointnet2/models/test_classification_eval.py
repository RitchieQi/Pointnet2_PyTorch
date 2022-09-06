"""
Author: Benny
Date: Nov 2019
"""
import sys
sys.path.append('/home/liyuan/HandEstimation/model/P2Preg/')

from dataset.MSRAhand_dataset import MSRAhand
import argparse
import numpy as np
import os
import torch
import logging
from tqdm import tqdm
from ICVL_dataset import ICVLHand

import importlib
from datapreprocess import data_process
from datapreprocess import data_postprocess
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Testing')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in training')
    parser.add_argument('--num_category', default=63, type=int,  help='training on ModelNet10/40')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--log_dir', type=str, default='..',help='Experiment root')
    parser.add_argument('--use_normals', action='store_true', default=True, help='use normals')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    # parser.add_argument('--num_votes', type=int, default=3, help='Aggregate classification scores with voting')
    return parser.parse_args()


def test(model, loader):
    
    classifier = model.eval()
    l2_list = []
    l2_pm = []
    l2_ThR = []
    l2_ThT = []
    l2_idR = []
    l2_idT = []
    l2_mdR = []
    l2_mdT = []
    l2_rnR = []
    l2_rnT = []
    l2_pkR = []
    l2_pkT = []
    for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):


        #points = points.transpose(2, 1)
        nor_points = []
        oBBLs = []
        nor_centers = []
        fst_rs = []
        sec_rs = []
        r_cens = []
        for i in range(32):
            nor_point,nor_label,oBBL,nor_center,fst_r,sec_r,r_cen = data_process(points[i],target[i])
            nor_points.append(nor_point)
            oBBLs.append(oBBL)
            nor_centers.append(nor_center)
            fst_rs.append(fst_r)
            sec_rs.append(sec_r)
            r_cens.append(r_cen)
        nor_points = torch.stack(nor_points)
        #oBBLs = torch.stack(oBBL)
        #nor_centers = torch.stack(nor_center)
        if not args.use_cpu:
            nor_points, target = nor_points.cuda(), target.cuda()
        
        

        l2_temp = []
        target = target.cpu().detach().numpy()
        target = target.reshape([32,21,3])
        
        pred,_ = classifier(nor_points)
        pred = pred.unsqueeze(-1).view(32,21,3)
        pred_list = pred.float()
        #oBBLs = oBBLs.numpy()
        #nor_centers = nor_centers.numpy()
        for i in range(32):
            camCS = data_postprocess(pred_list[i],oBBLs[i],nor_centers[i],fst_rs[i],sec_rs[i],r_cens[i])
            for j in range(21):
                if j == 0:
                    l2_pm.append(np.linalg.norm(camCS[j]-target[i][j]))
                if j == 1:
                    l2_idR.append(np.linalg.norm(camCS[j]-target[i][j]))
                if j == 4:
                    l2_idT.append(np.linalg.norm(camCS[j]-target[i][j]))
                if j == 5:
                    l2_mdR.append(np.linalg.norm(camCS[j]-target[i][j]))
                if j == 8:
                    l2_mdT.append(np.linalg.norm(camCS[j]-target[i][j]))
                if j == 9:
                    l2_rnR.append(np.linalg.norm(camCS[j]-target[i][j]))
                if j == 12:
                    l2_rnT.append(np.linalg.norm(camCS[j]-target[i][j]))
                if j == 13:
                    l2_pkR.append(np.linalg.norm(camCS[j]-target[i][j]))
                if j == 16:
                    l2_pkT.append(np.linalg.norm(camCS[j]-target[i][j]))
                if j == 17:
                    l2_ThR.append(np.linalg.norm(camCS[j]-target[i][j]))
                if j == 20:
                    l2_ThT.append(np.linalg.norm(camCS[j]-target[i][j]))
                l2_norm = np.linalg.norm(camCS[j]-target[i][j])
                #print(l2_norm == np.mean(l2_norm))
                l2_temp.append(np.mean(l2_norm))
        l2_list.append(np.mean(l2_temp))

        
    print('wrist ',np.mean(l2_pm),' idr ',np.mean(l2_idR),' idt ',np.mean(l2_idT),' mdr ',np.mean(l2_mdR),' mdt ',np.mean(l2_mdT),' rnr ',np.mean(l2_rnR),' rnt ',np.mean(l2_rnT),' pkr ',np.mean(l2_pkR),' pkt ',np.mean(l2_pkT),' thr ',np.mean(l2_ThR),' tht ',np.mean(l2_ThT))
    l2_list = np.array(l2_list)
    avg_l2_norm = np.mean(l2_list)

    return avg_l2_norm


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    experiment_dir = 'log/classification/' + args.log_dir

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')


    #test_dataset = ContactPose('test')
    #test_dataset = MSRAhand(n_sample=1024,task = 'test')
    test_dataset = ICVLHand(task = 'test')

    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8,drop_last=True)
    '''MODEL LOADING'''
    num_class = args.num_category
    #model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    model_name = 'handpointnet2'
    model = importlib.import_module(model_name)

    #classifier = model.getPretrainedHandglobal()
    classifier = model.get_model()
    if not args.use_cpu:
        classifier = classifier.cuda()

    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/ICVLhandGlobal.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])

    with torch.no_grad():
        avg_l2= test(classifier.eval(), testDataLoader)
        log_string('Test average l2: %f' % avg_l2)





if __name__ == '__main__':
    args = parse_args()
    main(args)
