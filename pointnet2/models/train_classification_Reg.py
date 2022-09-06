"""
Author: Benny
Date: Nov 2019
"""

import os
import sys
import torch
import numpy as np

import datetime
import logging
import importlib
import shutil
import argparse
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import torch.nn as nn
from MSRAhand_dataset import MSRAhand_n
from HANDS17_dataset import Hands17data
from loss import ofstL1Loss
from ICVL_dataset import ICVL_n

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size in training')
    parser.add_argument('--model', default='handpointnet2_Reg', help='model name [default: pointnet_cls]')
    parser.add_argument('--num_category', default=63, type=float,  help='training on ModelNet10/40')
    parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default='..', help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--use_normals', action='store_true', default=True, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    return parser.parse_args()


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True


def test(model, loader):
    
    classifier = model.eval()
    l1_list = []
    gl1_list = []
    for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):
        if not args.use_cpu:
            points, target = points.cuda(), target.cuda()

        
        
        #points = points.transpose(2, 1)
        refinePC,pred,globalEst = classifier(points)
       
        #refinePC: B,21,N,6  pred:B,21,N,3

        voteMap = refinePC[...,:3] - pred
        #vote: B,21,N,3
        vote = torch.mean(voteMap,dim=2)
        #print(vote.shape)


        
        
        target = target.view(args.batch_size,21,3)
        
        #print(pred.size())
        total_loss = F.mse_loss(vote, target)
        globalLoss = F.mse_loss(globalEst, target)
        #print(F.mse_loss(vote,globalEst))
        l1_list.append(total_loss.cpu())

        gl1_list.append(globalLoss.cpu())
    
    avg_l1 = np.mean(l1_list)
    avg_gl1 = np.mean(gl1_list)

    return avg_l1,avg_gl1


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('classification')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    train_dataset = ICVL_n(task = 'train')
    test_dataset = ICVL_n(task = 'test')
    # train_dataset = MSRAhand(n_sample =1024, task = 'train')
    # test_dataset = MSRAhand(n_sample = 1024, task = 'test')
    # train_dataset = MSRAhand_n(task = 'train')
    # test_dataset = MSRAhand_n(task = 'test')
    #data = Hands17data(task = 'train')
    # test_dataset = Hands17data(task = 'test')
    #test_dataset,train_dataset = torch.utils.data.random_split(data, [20000, 937032], generator=torch.Generator().manual_seed(42))

    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=True)

    '''MODEL LOADING'''
    num_class = args.num_category
    model = importlib.import_module(args.model)
    shutil.copy('./%s.py' % args.model, str(exp_dir))
    #shutil.copy('./utils.py', str(exp_dir))
    #shutil.copy('./train_classification.py', str(exp_dir))

    classifier = model.get_model()#, normal_channel=args.use_normals)
    criterion = ofstL1Loss()

    classifier.apply(inplace_relu)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

    if not args.use_cpu:
        classifier = classifier.cuda()
        criterion = criterion.cuda()

    try:
        checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0
        #classifier = classifier.apply(weights_init)

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    global_epoch = 0
    global_step = 0
    lowest_avg_mse = 1
    current_ofstL1 = 0.0

    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        mse = []
        classifier = classifier.train()

        pbar = tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9)
        for batch_id, (points, target) in pbar:
            optimizer.zero_grad()
            

            #points = points.transpose(2, 1)

            if not args.use_cpu:
                points, target = points.cuda(), target.cuda()

            refinePC,pred,globalEst = classifier(points)
            
            loss = criterion(target.float(),refinePC,pred)
            
            

            current_ofstL1 = loss.item()

            mse.append(current_ofstL1)
            loss.backward()
            optimizer.step()
            global_step += 1
            pbar.set_description("Loss:%f" % (current_ofstL1))

        scheduler.step()
        train_mean_mse = np.mean(mse)
        log_string('Train MSE: %f' % train_mean_mse)

        with torch.no_grad():
            avg_mse,avg_gl1 = test(classifier.eval(), testDataLoader)

            if (avg_mse <= lowest_avg_mse):
                lowest_avg_mse = avg_mse
                best_epoch = epoch + 1


            log_string('Test Average MSE: %f, Best Average MSE: %f , Test GlobalL1: %f' % (avg_mse,lowest_avg_mse,avg_gl1))
            

            if (avg_mse <= lowest_avg_mse):
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': best_epoch,
                    'avg_mse': avg_mse,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            global_epoch += 1
            
            if (global_epoch%20 == 0):
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model'+ str(global_epoch)+'.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': best_epoch,
                    'avg_mse': avg_mse,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
    logger.info('End of training...')


if __name__ == '__main__':
    args = parse_args()
    main(args)
