'''
Author:Liyuan Qi
'''
from open3d import geometry as o3dg
from open3d import utility as o3du
from open3d import camera as o3dc
from open3d import io as o3dio
import numpy as np
from scipy.spatial.transform import Rotation as rot
import torch

datadir = '/home/liyuan/HandEstimation/dataset/MSRAhand_data_1024'

def rot_plus(rotation_mat):
    angle = rot.from_matrix(rotation_mat).as_euler('xyz', degrees=True)
    if angle[0]>0:
        if angle[2]>0:
            extra_r = rot.from_euler('xyz',[0,0,0],degrees=True)
            extra_r = extra_r.as_matrix()
        elif angle[2]<0:
            extra_r = rot.from_euler('xyz',[0,0,180],degrees=True)
            extra_r = extra_r.as_matrix()
    if angle[0]<0:
        if angle[2]<0:
            extra_r = rot.from_euler('xyz',[0,0,0],degrees=True)
            extra_r = extra_r.as_matrix()
        elif angle[2]>0:
            extra_r = rot.from_euler('xyz',[0,0,180],degrees=True)
            extra_r = extra_r.as_matrix()
    extra_r = np.eye(3)
    return extra_r

def getoBB(pointcloud,label):
    '''
    rotate the pointcloud & label from camera frame to oBB frame
    input: 
        pointcloud: hand pointcloud, open3d.geometry.pointcloud
        label: label pointcloud, open3d.geometry.pointcloud
    output:
        oBBCS_pointcloud:open3d.geometry.pointcloud
        oBBCS_label:open3d.geometry.pointcloud
        edgeLength: the length of the oBB edges,[x,y,z]

    '''
    oBB = pointcloud.get_oriented_bounding_box()
    R_mat = np.linalg.inv(oBB.R)
    center = oBB.center
    ''' pointcloud under obb coordinator system '''
    oBBCS_pointcloud = pointcloud.rotate(R_mat,center)
    oBBCS_label = label.rotate(R_mat,center)
    extra_rot = rot_plus(R_mat)
    oBBCS_pointcloud = oBBCS_pointcloud.rotate(extra_rot,center)
    oBBCS_label = oBBCS_label.rotate(extra_rot,center)
    ''' obb attr '''
    edgeLength = oBB.extent
    #return oBBCS_pointcloud,oBBCS_label,edgeLength.max(),R_mat,extra_rot,center
    return oBBCS_pointcloud,oBBCS_label,edgeLength,R_mat,extra_rot,center

def pointset_normalize(pointcloud,label,edgeLength):
    '''
    normalize the pointset to be within (-0.5,0.5)
    input:
        pointcloud/lable: open3d.geometry.pointcloud
        edgeLength: the length of the oBB edges,[x,y,z]
    output:
        points_norm,label_norm: normalized hand points and label points, numpy.array
    '''
    points_array = np.array(pointcloud.points)
    label_array = np.array(label.points)
    points_center = points_array.mean(0)
    points_norm = (points_array-points_center)/(1.2*edgeLength)
    label_norm = (label_array-points_center)/(1.2*edgeLength)
    return points_norm,label_norm,points_center

def get_normals(pointcloud):
    pointcloud.estimate_normals()
    pointcloud.normalize_normals()
    return np.array(np.array(pointcloud.normals))

def create_pc(pointsets,f_label):
    pointcloud = o3dg.PointCloud()
    pointcloud.points = o3du.Vector3dVector(pointsets.numpy().reshape([1024,3]))

    joints = o3dg.PointCloud()
    joints.points = o3du.Vector3dVector(f_label.numpy().reshape([21,3]))
    return pointcloud,joints


def data_process(data,label):
    data,label = create_pc(data,label)
    oBB_pc,oBB_label,Length,first_r, second_r,r_cen = getoBB(data,label)
    norms = get_normals(oBB_pc)
    nor_pc,nor_label,nor_center = pointset_normalize(oBB_pc,oBB_label,Length)
    xyz_n = np.hstack((nor_pc,norms))
    xyz_n_tenor = torch.from_numpy(xyz_n.copy()).float()
    labels_tensor = torch.flatten(torch.from_numpy(nor_label.copy()).float())
    return xyz_n_tenor, labels_tensor,Length,nor_center,first_r, second_r,r_cen

def data_postprocess(label,obbl,nor_center,fst_r,scd_r,r_cen):
    label = (label.cpu().numpy()*(1.2*obbl))+nor_center
    o3d_label = o3dg.PointCloud()
    o3d_label.points = o3du.Vector3dVector(label)
    o3d_label.rotate(np.linalg.inv(scd_r),r_cen)
    o3d_label.rotate(np.linalg.inv(fst_r),r_cen)
    post_label = np.array(o3d_label.points)
    
    #post_label = torch.from_numpy(post_label.reshape(21,3)).float()
    return post_label

def data_process_wPhase2(data,label):
    # joints = o3dg.PointCloud()
    # joints.points = o3du.Vector3dVector(label)
    joints = label
    oBB_pc,oBB_label,Length,first_r, second_r,r_cen = getoBB(data,joints)
    norms = get_normals(oBB_pc)
    nor_pc,nor_label,nor_center = pointset_normalize(oBB_pc,oBB_label,Length)
    xyz_n = np.hstack((nor_pc,norms))
    xyz_n_tenor = torch.from_numpy(xyz_n.copy()).float()
    labels_tensor = torch.flatten(torch.from_numpy(nor_label.copy()).float())
    return xyz_n_tenor, labels_tensor