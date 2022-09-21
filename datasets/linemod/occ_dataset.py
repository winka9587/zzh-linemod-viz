
import math
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from os.path import join as pjoin
import cv2
import _pickle as cPickle
import random
import numpy.ma as ma
from plyfile import PlyData

border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
img_width = 480
img_length = 640


def mask_to_bbox(mask):
    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    x = 0
    y = 0
    w = 0
    h = 0
    for contour in contours:
        tmp_x, tmp_y, tmp_w, tmp_h = cv2.boundingRect(contour)
        if tmp_w * tmp_h > w * h:
            x = tmp_x
            y = tmp_y
            w = tmp_w
            h = tmp_h
    return [x, y, w, h]


def get_bbox(bbox):
    bbx = [bbox[1], bbox[1] + bbox[3], bbox[0], bbox[0] + bbox[2]]
    if bbx[0] < 0:
        bbx[0] = 0
    if bbx[1] >= 480:
        bbx[1] = 479
    if bbx[2] < 0:
        bbx[2] = 0
    if bbx[3] >= 640:
        bbx[3] = 639
    rmin, rmax, cmin, cmax = bbx[0], bbx[1], bbx[2], bbx[3]
    r_b = rmax - rmin
    for tt in range(len(border_list)):
        if r_b > border_list[tt] and r_b < border_list[tt + 1]:
            r_b = border_list[tt + 1]
            break
    c_b = cmax - cmin
    for tt in range(len(border_list)):
        if c_b > border_list[tt] and c_b < border_list[tt + 1]:
            c_b = border_list[tt + 1]
            break
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > 480:
        delt = rmax - 480
        rmax = 480
        rmin -= delt
    if cmax > 640:
        delt = cmax - 640
        cmax = 640
        cmin -= delt
    return rmin, rmax, cmin, cmax


def ply_vtx(path):
    f = open(path)
    assert f.readline().strip() == "ply"
    f.readline()
    f.readline()
    N = int(f.readline().split()[-1])
    while f.readline().strip() != "end_header":
        continue
    pts = []
    for _ in range(N):
        pts.append(np.float32(f.readline().split()[:3]))
    return np.array(pts)

def load_ply_model(model_path):
    ply = PlyData.read(model_path)
    data = ply.elements[0].data
    x = data['x']
    y = data['y']
    z = data['z']
    return np.stack([x, y, z], axis=-1)


def project_model(model, pose, intrinsic_matrix):
    camera_points_3d = np.dot(model, pose[:, :3].T) + pose[:, 3]
    print(f'camera_points_3d: {camera_points_3d[0]}')
    camera_points_3d = np.dot(camera_points_3d, intrinsic_matrix.T)
    return camera_points_3d[:, :2] / camera_points_3d[:, 2:]

def project_model_R(model, pose, intrinsic_matrix, R):
    model_tmp = np.dot(model, R)
    # pose_t_tmp = np.dot(pose[:, 3].T, R)
    pose_t_tmp = pose[:, 3].T
    camera_points_3d = np.dot(model_tmp, pose[:, :3].T) + pose_t_tmp
    print(f'camera_points_3d: {camera_points_3d[0]}')
    camera_points_3d = np.dot(camera_points_3d, intrinsic_matrix.T)
    return camera_points_3d[:, :2] / camera_points_3d[:, 2:]

def project_PVNet(pts_3d, intrinsic_matrix):
    pts_2d = np.matmul(pts_3d, intrinsic_matrix.T)
    pts_2d = pts_2d[:, :2] / pts_2d[:, 2:]
    return pts_2d

class OCCLinemodDataset(Dataset):
    # mode : train, test, all
    def __init__(self, mode, num, root_path):
        self.symmetry_obj_idx = [7, 8]
        self.change_list = ["cat", "duck", "holepuncher"]
        self.intrinsics = np.array([[572.41140, 0, 325.26110], [0, 573.57043, 242.04899], [0, 0, 1]])
        # self.obj_name_list = ['ape', 'can', 'cat', 'driller', 'duck', 'eggbox', 'glue', 'holepuncher']
        self.obj_name_list = ['ape']
        self.cam_cx = 325.26110
        self.cam_cy = 242.04899
        self.cam_fx = 572.41140
        self.cam_fy = 573.57043
        self.xmap = np.array([[j for i in range(640)] for j in range(480)])
        self.ymap = np.array([[i for i in range(640)] for j in range(480)])

        self.num = num
        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
        self.refine = False
        self.num_pt_mesh_large = 500
        self.num_pt_mesh_small = 500
        self.obj_folder = {
            'ape': 'Ape',
            'can': 'Can',
            'cat': 'Cat',
            'driller': 'Driller',
            'duck': 'Duck',
            'eggbox': 'Eggbox',
            'glue': 'Glue',
            'holepuncher': 'Holepuncher'
        }
        self.xyz_path = {
            'ape': '001',
            'can': '004',
            'cat': '005',
            'driller': '006',
            'duck': '007',
            'eggbox': '008',
            'glue': '009',
            'holepuncher': '010'
        }
        self.lm_obj_dict = {
            'ape': 1,
            'benchvise': 2,
            'cam': 4,
            'can': 5,
            'cat': 6,
            'driller': 8,
            'duck': 9,
            'eggbox': 10,
            'glue': 11,
            'holepuncher': 12,
            'iron': 13,
            'lamp': 14,
            'phone': 15,
        }

        self.mode = mode
        self.list_rgb = []
        self.list_mask = []
        self.list_depth = []
        self.list_pose = []
        self.list_blender_pose = []
        self.list_label = []
        self.list_obj = []
        self.linemod_model = {}  # 存储linemod数据集模型
        self.root = root_path

        self.xyz = np.loadtxt(pjoin(self.root, 'models', self.obj_folder[self.obj_name_list[0]], self.xyz_path[self.obj_name_list[0]] + '.xyz'))
        rotation = np.array([[0., 0., 1.],
                             [1., 0., 0.],
                             [0., 1., 0.]])
        self.xyz = np.dot(self.xyz, rotation.T)


        self.anns_pkl_list = []
        self.anns_list = []
        self.length = 0
        for obj_name in self.obj_name_list:
            anns_list_tmp = []
            if self.mode == 'train':
                self.anns_pkl_list.append(pjoin(root_path, f'anns/{obj_name}/train.pkl'))
            elif self.mode == 'test':
                self.anns_pkl_list.append(pjoin(root_path, f'anns/{obj_name}/test.pkl'))
            elif self.mode == 'all':
                self.anns_pkl_list.append(pjoin(root_path, f'anns/{obj_name}/train.pkl'))
                self.anns_pkl_list.append(pjoin(root_path, f'anns/{obj_name}/test.pkl'))

            for pkl_path in self.anns_pkl_list:
                anns_f = open(pjoin(self.root, pkl_path), 'rb')
                anns_list_tmp = cPickle.load(anns_f)

            for info in anns_list_tmp:
                # check_list = {
                #     "RGB-D/rgb_noseg/color_00966.png",
                #     "RGB-D/rgb_noseg/color_00970.png",
                #     "RGB-D/rgb_noseg/color_00967.png",
                #     "RGB-D/rgb_noseg/color_00977.png",
                #     "RGB-D/rgb_noseg/color_00974.png",
                #     "RGB-D/rgb_noseg/color_00958.png",
                #     "RGB-D/rgb_noseg/color_00995.png",
                #     "RGB-D/rgb_noseg/color_00968.png",
                #     "RGB-D/rgb_noseg/color_00972.png",
                #     "RGB-D/rgb_noseg/color_00971.png"
                # }
                # print('get '+info[0].replace('data/occ_linemod/', ''))
                # if info[0].replace('data/occ_linemod/', '') in check_list:
                # print(f"append color: {self.list_rgb[index]}")
                img_idx = info[0][info[0].find('color_')+6: -4]
                self.list_rgb.append(info[0].replace('data/occ_linemod/', ''))  # rgb
                self.list_depth.append(info[0].replace('data/occ_linemod/', '').replace('rgb_noseg', 'depth_noseg').replace('color', 'depth'))  # depth
                self.list_mask.append(info[2].replace('data/occ_linemod/', ''))  # mask
                self.list_pose.append(f'poses/{self.obj_folder[obj_name]}/info_{img_idx}.txt')
                self.list_blender_pose.append(f'blender_poses/{obj_name}/pose{int(img_idx)}.npy')
                self.list_obj.append(obj_name)
            # print("init end !!!!!")

            # 加载linemod模型
            self.linemod_model[obj_name] = ply_vtx('/data1/zzh/DenseFusion-Pytorch-1.0/datasets/linemod/Linemod_preprocessed/models/obj_{0}.ply'.format('%02d' % self.lm_obj_dict[obj_name]))

            self.length += len(anns_list_tmp)


    def get_translation_transform(self, xyz):
        # if self.class_type in self.translation_transforms:
        #     return self.translation_transforms[self.class_type]

        blender_model_path = '/data1/zzh/DenseFusion-Pytorch-1.0/datasets/linemod/Linemod_preprocessed/models/obj_{}.ply'.format('%02d' % self.lm_obj_dict[self.obj_name_list[0]])
        model = load_ply_model(blender_model_path)
        model = model/1000.0

        # xyz = np.loadtxt(self.xyz_pattern.format(
        #     self.class_type.title(), self.class_type_to_number[self.class_type]))
        # rotation = np.array([[0., 0., 1.],
        #                      [1., 0., 0.],
        #                      [0., 1., 0.]])
        # xyz = np.dot(xyz, rotation.T)
        translation_transform = np.mean(xyz, axis=0) - np.mean(model, axis=0)
        # self.translation_transforms[self.class_type] = translation_transform

        return translation_transform

    def occlusion_pose_to_blender_pose(self, pose, xyz):
        rot, tra = pose[:, :3], pose[:, 3]
        rotation = np.array([[0., 1., 0.],
                             [0., 0., 1.],
                             [1., 0., 0.]])
        rot = np.dot(rot, rotation)
        tra[1:] *= -1
        translation_transform = np.dot(rot, self.get_translation_transform(xyz))
        rot[1:] *= -1
        translation_transform[1:] *= -1
        tra += translation_transform
        pose = np.concatenate([rot, np.reshape(tra, newshape=[3, 1])], axis=-1)
        return pose

    def __getitem__(self, index):
        print(f"getitem list_rgb:{self.list_rgb[index]}")
        ret = {}
        color = cv2.imread(pjoin(self.root, self.list_rgb[index]))
        depth = cv2.imread(pjoin(self.root, self.list_depth[index]), -1)
        mask = cv2.imread(pjoin(self.root, self.list_mask[index]), -1)
        obj_name = self.list_obj[index]
        label = mask
        mask_label = ma.getmaskarray(ma.masked_equal(label, np.array(1)))
        rmin, rmax, cmin, cmax = get_bbox(mask_to_bbox(mask_label))



        mask_bool = (mask == 1)
        final_mask = np.logical_and(mask_bool, depth > 0)
        choose = final_mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
        if len(choose) == 0:
            cc = torch.LongTensor([0])
            return(cc, cc, cc, cc, cc, cc)
        if len(choose) > self.num:
            c_mask = np.zeros(len(choose), dtype=int)
            c_mask[:self.num] = 1
            np.random.shuffle(c_mask)
            choose = choose[c_mask.nonzero()]
        else:
            choose = np.pad(choose, (0, self.num - len(choose)), 'wrap')

        # masked
        img = np.array(color)[:, :, :3]
        img = np.transpose(img, (2, 0, 1))
        img_masked = img[:, rmin:rmax, cmin:cmax]
        depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        xmap_masked = self.xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        ymap_masked = self.ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        choose = np.array([choose])


        # cv2.imshow(f"color:{self.list_rgb[index]}", color)
        # cv2.waitKey(0)
        # cv2.imshow("color_masked", color[rmin:rmax, cmin:cmax, :])
        # cv2.waitKey(0)
        # print(f"len(choose):{len(choose)}")
        # print(f"r:{0},{1},{2},{3}".format(rmin, rmax, cmin, cmax))
        # print(f"choose: {choose}")

        cam_scale = 1.0
        pt2 = depth_masked / cam_scale
        pt0 = (ymap_masked - self.cam_cx) * pt2 / self.cam_fx
        pt1 = (xmap_masked - self.cam_cy) * pt2 / self.cam_fy
        cloud = np.concatenate((pt0, pt1, pt2), axis=1)
        cloud = cloud / 1000.0

        # 读取位姿

        # 读取模型
        # use model in occ-linemod
        # xyz = xyz / 1000.0  # 是否要除以？NO
        # dellist = [j for j in range(0, len(xyz))]
        # dellist = random.sample(dellist, len(xyz) - self.num_pt_mesh_small)
        # model_points = np.delete(xyz, dellist, axis=0)
        # model_points[:, 0] = -model_points[:, 0]
        if obj_name == "holepuncher":
            poses_R = np.loadtxt(pjoin(self.root, self.list_pose[index]), skiprows=6, max_rows=3)
            poses_t = np.loadtxt(pjoin(self.root, self.list_pose[index]), skiprows=10, max_rows=1)
        else:
            poses_R = np.loadtxt(pjoin(self.root, self.list_pose[index]), skiprows=4, max_rows=3)
            poses_t = np.loadtxt(pjoin(self.root, self.list_pose[index]), skiprows=8, max_rows=1)
        poses = np.hstack((poses_R, poses_t.reshape(3, 1)))


        poses = self.occlusion_pose_to_blender_pose(poses, self.xyz)
        # use model in linemod
        model_linemod = self.linemod_model[obj_name]
        model_linemod = model_linemod / 1000.0

        if obj_name in self.change_list:
            # cat, duck, holepuncher
            rotation_poses = np.array([
                [1., 0., 0.],
                [0., 1., 0.],
                [0., 0., 1.]])
        else:
            # ape, can, driller, eggbox(都可), glue
            rotation_poses = np.array([
                [-1., 0., 0.],
                [0., -1., 0.],
                [0., 0., 1.]])

        poses[:, :3] = np.dot(poses[:, :3], rotation_poses.T)
        dellist = [j for j in range(0, len(model_linemod))]
        dellist = random.sample(dellist, len(model_linemod) - self.num_pt_mesh_small)
        model_points = np.delete(model_linemod, dellist, axis=0)
        target = np.dot(model_points, poses[:, :3].T)
        target = np.add(target, poses[:, 3])




        # # blender pose
        # b_poses = np.load(pjoin(self.root, self.list_blender_pose[index]))

        # 投影linemod
        # uv_2 = project_model(model_linemod, poses_linemod, intrinsics).astype(np.int32)
        # for u, v in uv_2:
        #     color[v, u, 0] = 255
        #     color[v, u, 1:] = 0
        # cv2.imshow('project linemod model origin', color)
        # cv2.waitKey(0)


        # 可视化
        # self.cam_cx = 325.26110
        # self.cam_cy = 242.04899
        # self.cam_fx = 572.41140
        # self.cam_fy = 573.57043



        # intrinsic_matrix = {
        #     'linemod': np.array([[572.4114, 0., 325.2611],
        #                          [0., 573.57043, 242.04899],
        #                          [0., 0., 1.]]),
        # }
        # 可视化：投影cloud
        # uv = project_PVNet(cloud).astype(np.int32)
        # for u, v in uv:
        #     color[v, u, :] = 255
        # cv2.imshow('project cloud', color)
        # cv2.waitKey(0)
        # Project target
        # 可视化：投影target
        # uv_target = project_PVNet(target, self.intrinsics).astype(np.int32)
        # for u, v in uv_target:
        #     # 加判断，防止物体有一部分在画面外
        #     if 0 < v < color.shape[0] and 0 < u < color.shape[1]:
        #         color[v, u, 0:2] = 0
        #         color[v, u, 2] = 255
        # cv2.imshow('project target', color)
        # cv2.waitKey(0)
        #
        # # 可视化：投影poses变换后的model_points，应该与target重合
        # uv2 = project_model(model_points, poses, self.intrinsics).astype(np.int32)
        # for u, v in uv2:
        #     if 0 < v < color.shape[0] and 0 < u < color.shape[1]:
        #         color[v, u, 0] = 255
        #         color[v, u, 1:] = 0
        # cv2.imshow('project model', color)
        # cv2.waitKey(0)

        # cloud, choose, img_masked, target, model_points, obj_name, color_path, poses_gt

        return torch.from_numpy(cloud.astype(np.float32)), \
               torch.LongTensor(choose.transpose(0, 1).astype(np.int32)), \
               self.norm(torch.from_numpy(img_masked.astype(np.float32))), \
               torch.from_numpy(target.astype(np.float32)), \
               torch.from_numpy(model_points.astype(np.float32)), \
               torch.LongTensor([self.lm_obj_dict[obj_name]]), \
               pjoin(self.root, self.list_rgb[index]), \
               torch.from_numpy(model_linemod)

    def __len__(self):
        return self.length

    def get_sym_list(self):
        return self.symmetry_obj_idx

    def get_num_points_mesh(self):
        if self.refine:
            return self.num_pt_mesh_large
        else:
            return self.num_pt_mesh_small

if __name__ == "__main__":
    dataset = OCCLinemodDataset(mode='train',
                                num=1024,
                                root_path='/data1/zzh/OCCLUSION_LINEMOD'
                                )
    train_dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    for i, data in enumerate(train_dataloader):
        cloud, choose, img_masked, target, model_points, obj_name = data
        # img = (img_masked.squeeze(0).permute(1, 2, 0).numpy()).astype(np.uint8)
        print(i, data)

        # render_points_diff_color("cloud", [cloud.numpy().squeeze(0)], [np.array([0, 191, 255])], False, False)
        #
        # render_points_diff_color("target", [target.numpy().squeeze(0)], [np.array([238, 203, 173])], False, False)
        #
        # render_points_diff_color("target+cloud", [cloud.numpy().squeeze(0), target.numpy().squeeze(0)], [np.array([0, 191, 255]), np.array([238, 203, 173])], False, False)




