import _init_paths
import argparse
import os
import random
import numpy as np
import yaml
import copy
import cv2
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from datasets.linemod.dataset import PoseDataset as PoseDataset_linemod
from datasets.linemod.linemod_dataset import Dataset as Dataset_linemod
from datasets.linemod.occ_dataset.py import OCCLinemodDataset as Dataset_linemod_occ
from lib.network import PoseNet, PoseRefineNet, PoseNet_trans, PoseNet_trans_mix
from lib.loss import Loss
from lib.loss_refiner import Loss_refine
from lib.transformations import euler_matrix, quaternion_matrix, quaternion_from_matrix
from lib.knn.__init__ import KNearestNeighbor

def project_PVNet(pts_3d, intrinsic_matrix):
    pts_2d = np.matmul(pts_3d, intrinsic_matrix.T)
    pts_2d = pts_2d[:, :2] / pts_2d[:, 2:]
    return pts_2d

def viz_target_on_img(target_, intrinsic_, img_, r ,g, b):
    img = img_.copy()
    uv_target_ = project_PVNet(target_, intrinsic_).astype(np.int32)
    uv = []
    for u_, v_ in uv_target_:
        uv.append((u_, v_))  # orgin
        if 0 < v_ < img.shape[0] and 0 < u_ < img.shape[1]:
            pass
            # img[v_, u_, 0] = b
            # img[v_, u_, 1] = g
            # img[v_, u_, 2] = r
    return img, uv

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', type=str, default = '/data1/zzh/DenseFusion-Pytorch-1.0/datasets/linemod/Linemod_preprocessed/',
                    help='dataset root dir')
parser.add_argument('--model', type=str, default = "/data1/zzh/DenseFusion-Pytorch-1.0/trained_models/linemod1/pose_model_30_0.008178278177831611.pth",
                    help='resume PoseNet model')
parser.add_argument('--refine_model', type=str, default = "/data1/zzh/DenseFusion-Pytorch-1.0/trained_models/linemod_my/pose_refine_model_121_0.007144017796867602.pth",
                    help='resume PoseRefineNet model')

opt = parser.parse_args()

num_objects = 13
objlist = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
num_points = 500
iteration = 4
bs = 1
dataset_config_dir = 'datasets/linemod/dataset_config'
output_result_dir = 'experiments/eval_result/linemod'
knn = KNearestNeighbor(1)

# estimator = PoseNet(num_points = num_points, num_obj = num_objects)
estimator = PoseNet_trans(num_points=num_points, num_obj=num_objects)
estimator.cuda()
refiner = PoseRefineNet(num_points=num_points, num_obj=num_objects)
refiner.cuda()
estimator.load_state_dict(torch.load(opt.model))
refiner.load_state_dict(torch.load(opt.refine_model))
estimator.eval()
refiner.eval()
obj_ = 'benchvise'
# first = False
save = True
testdataset = PoseDataset_linemod('eval', num_points, False, opt.dataset_root, 0.0, True, obj_)
# testdataset = Dataset_linemod('test', True, 0.0, 'ape', DEBUG=True)
testdataloader = torch.utils.data.DataLoader(testdataset, batch_size=1, shuffle=False, num_workers=10)

# sym_list = testdataset.get_sym_list()
# num_points_mesh = testdataset.get_num_points_mesh()
# criterion = Loss(num_points_mesh, sym_list)
# criterion_refine = Loss_refine(num_points_mesh, sym_list)
#
# diameter = []
# meta_file = open('{0}/models_info.yml'.format(dataset_config_dir), 'r')
# meta = yaml.safe_load(meta_file)
# for obj in objlist:

#     diameter.append(meta[obj]['diameter'] / 1000.0 * 0.1)
# print(diameter)
#
# success_count = [0 for i in range(num_objects)]
# num_count = [0 for i in range(num_objects)]
# fw = open('{0}/eval_result_logs.txt'.format(output_result_dir), 'w')

for i, data in enumerate(testdataloader, 0):
    try:
        cloud, choose, img_masked, target, model_points, obj_name, color_path, poses_gt = data
    except:
        continue
    points, choose, img, target, model_points, idx = Variable(cloud).cuda(), \
                                                    Variable(choose).cuda(), \
                                                    Variable(img_masked).cuda(), \
                                                    Variable(target).cuda(), \
                                                    Variable(model_points).cuda(), \
                                                    Variable(obj_name).cuda()
    # Variable(t).cuda()
    max_x, max_y, max_z = torch.max(model_points[0], 0)[0]
    min_x, min_y, min_z = torch.min(model_points[0], 0)[0]
    max_x = max_x.item()
    max_y = max_y.item()
    max_z = max_z.item()
    min_x = min_x.item()
    min_y = min_y.item()
    min_z = min_z.item()
    corners = torch.tensor([
        [max_x, max_y, min_z],
        [max_x, min_y, min_z],
        [min_x, max_y, min_z],
        [min_x, min_y, min_z],

        [max_x, max_y, max_z],
        [max_x, min_y, max_z],
        [min_x, max_y, max_z],
        [min_x, min_y, max_z],
    ])

    # pred_r, pred_t, pred_c, emb = estimator(img, points, choose, idx)
    pred_r, pred_t, pred_c, emb = estimator(img, points, model_points, choose, idx)
    pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, num_points, 1)
    pred_c = pred_c.view(bs, num_points)
    how_max, which_max = torch.max(pred_c, 1)
    pred_t = pred_t.view(bs * num_points, 1, 3)

    my_r = pred_r[0][which_max[0]].view(-1).cpu().data.numpy()
    my_t = (points.view(bs * num_points, 1, 3) + pred_t)[which_max[0]].view(-1).cpu().data.numpy()
    my_pred = np.append(my_r, my_t)

    for ite in range(0, iteration):
        T = Variable(torch.from_numpy(my_t.astype(np.float32))).cuda().view(1, 3).repeat(num_points,
                                                                                         1).contiguous().view(1,
                                                                                                              num_points,
                                                                                                              3)
        # print(my_r)
        my_mat = quaternion_matrix(my_r)
        R = Variable(torch.from_numpy(my_mat[:3, :3].astype(np.float32))).cuda().view(1, 3, 3)
        my_mat[0:3, 3] = my_t

        new_points = torch.bmm((points - T), R).contiguous()
        pred_r, pred_t = refiner(new_points, emb, idx)
        pred_r = pred_r.view(1, 1, -1)
        pred_r = pred_r / (torch.norm(pred_r, dim=2).view(1, 1, 1))
        my_r_2 = pred_r.view(-1).cpu().data.numpy()
        my_t_2 = pred_t.view(-1).cpu().data.numpy()
        my_mat_2 = quaternion_matrix(my_r_2)
        my_mat_2[0:3, 3] = my_t_2

        my_mat_final = np.dot(my_mat, my_mat_2)
        my_r_final = copy.deepcopy(my_mat_final)
        my_r_final[0:3, 3] = 0
        my_r_final = quaternion_from_matrix(my_r_final, True)
        my_t_final = np.array([my_mat_final[0][3], my_mat_final[1][3], my_mat_final[2][3]])

        my_pred = np.append(my_r_final, my_t_final)
        my_r = my_r_final
        my_t = my_t_final

    # Here 'my_pred' is the final pose estimation result after refinement ('my_r': quaternion, 'my_t': translation)

    model_points = model_points[0].cpu().detach().numpy()
    my_r = quaternion_matrix(my_r)[:3, :3]
    pred = np.dot(model_points, my_r.T) + my_t
    corners = corners.numpy()
    #
    # print(f"corner:\n{corners}")
    corners_pred = np.dot(corners, my_r.T) + my_t

    # print(f"poses_gt:\n{poses_gt}")
    poses_gt = poses_gt[0].numpy()
    corners_gt = np.dot(corners, poses_gt[:, :3].T) + poses_gt[:, 3]
    target = target[0].cpu().detach().numpy()

    color_path = color_path[0]


    img_idx = color_path[-8:-4]
    print(f"loading img {img_idx} from {color_path}")
    save_path = f"/data1/zzh/paper_img3/{obj_}/{img_idx}.png"
    # if first:
    #     color = cv2.imread(color_path)
    # else:
    #     if os.path.exists(f"/data1/zzh/paper_img2/{img_idx}.png"):
    #         color = cv2.imread(f"/data1/zzh/paper_img2/{img_idx}.png")
    #     else:
    #         color = cv2.imread(color_path)

    color = cv2.imread(color_path)

    # pred_target_DF = np.add(np.dot(model_points, pred[:, :3].T), pred[:, 3])
    # # viz DF
    intrinsics = np.array([[572.41140, 0, 325.26110], [0, 573.57043, 242.04899], [0, 0, 1]])
    img_ = color
    img_, uv_gt = viz_target_on_img(corners_gt, intrinsics, img_, r=0, g=255, b=0)
    img_, uv_pred = viz_target_on_img(corners_pred, intrinsics, img_, r=0, g=0, b=255)
    thickness = 2
    gt_color = (0, 0, 255)
    # pred_color = testdataset.color_list[obj_]
    pred_color = (160, 242, 189)
    # gt
    cv2.line(img_, uv_gt[0], uv_gt[1], gt_color, thickness)
    cv2.line(img_, uv_gt[0], uv_gt[2], gt_color, thickness)
    cv2.line(img_, uv_gt[1], uv_gt[3], gt_color, thickness)
    cv2.line(img_, uv_gt[2], uv_gt[3], gt_color, thickness)

    cv2.line(img_, uv_gt[4], uv_gt[5], gt_color, thickness)
    cv2.line(img_, uv_gt[4], uv_gt[6], gt_color, thickness)
    cv2.line(img_, uv_gt[5], uv_gt[7], gt_color, thickness)
    cv2.line(img_, uv_gt[6], uv_gt[7], gt_color, thickness)

    cv2.line(img_, uv_gt[0], uv_gt[4], gt_color, thickness)
    cv2.line(img_, uv_gt[1], uv_gt[5], gt_color, thickness)
    cv2.line(img_, uv_gt[2], uv_gt[6], gt_color, thickness)
    cv2.line(img_, uv_gt[3], uv_gt[7], gt_color, thickness)
    # pred
    cv2.line(img_, uv_pred[0], uv_pred[1], pred_color, thickness)
    cv2.line(img_, uv_pred[0], uv_pred[2], pred_color, thickness)
    cv2.line(img_, uv_pred[1], uv_pred[3], pred_color, thickness)
    cv2.line(img_, uv_pred[2], uv_pred[3], pred_color, thickness)

    cv2.line(img_, uv_pred[4], uv_pred[5], pred_color, thickness)
    cv2.line(img_, uv_pred[4], uv_pred[6], pred_color, thickness)
    cv2.line(img_, uv_pred[5], uv_pred[7], pred_color, thickness)
    cv2.line(img_, uv_pred[6], uv_pred[7], pred_color, thickness)

    cv2.line(img_, uv_pred[0], uv_pred[4], pred_color, thickness)
    cv2.line(img_, uv_pred[1], uv_pred[5], pred_color, thickness)
    cv2.line(img_, uv_pred[2], uv_pred[6], pred_color, thickness)
    cv2.line(img_, uv_pred[3], uv_pred[7], pred_color, thickness)
    
    cv2.imshow('project target', img_)
    cv2.waitKey(0)
    if save:
        cv2.imwrite(save_path, img_)
        print(f"save img to {save_path}")
