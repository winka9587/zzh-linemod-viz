#!/usr/bin/env python3
import os
import cv2
import torch
import os.path
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from .common import Config
import pickle as pkl
from .basic_utils import Basic_Utils
import yaml
import scipy.io as scio
import scipy.misc
from glob import glob
from termcolor import colored
import normalSpeed
import numpy.ma as ma
import time
import numpy
import random
from .FPS import farthest_point_sampling
try:
    from neupeak.utils.webcv2 import imshow, waitKey
except ImportError:
    from cv2 import imshow, waitKey


class Dataset():

    def __init__(self, dataset_name, add_noise, noise_trans, cls_type="duck", DEBUG=False):
        self.DEBUG = DEBUG
        self.config = Config(ds_name='linemod', cls_type=cls_type)
        self.bs_utils = Basic_Utils(self.config)
        self.dataset_name = dataset_name
        self.xmap = np.array([[j for i in range(640)] for j in range(480)])
        self.ymap = np.array([[i for i in range(640)] for j in range(480)])


        self.cam_cx = 325.26110
        self.cam_cy = 242.04899
        self.cam_fx = 572.41140
        self.cam_fy = 573.57043
        
        self.add_noise = add_noise
        self.noise_trans = noise_trans
        self.border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
        self.num_pt_mesh_large = 500
        self.num_pt_mesh_small = 500
        self.num = 500
        self.symmetry_obj_idx = [7, 8]

        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.224])
        self.resize = transforms.Resize([80,80])
        self.obj_dict = self.config.lm_obj_dict

        self.cls_type = cls_type
        self.cls_id = self.obj_dict[cls_type]
        print("cls_id in lm_dataset.py", self.cls_id)
        self.root = os.path.join(self.config.lm_root, 'Linemod_preprocessed')
        self.cls_root = os.path.join(self.root, "data/%02d/" % self.cls_id)
        self.rng = np.random
        meta_file = open(os.path.join(self.cls_root, 'gt.yml'), "r")
        self.meta_lst = yaml.safe_load(meta_file)
        self.pt = ply_vtx('{0}/models/obj_{1}.ply'.format(self.root, '%02d' % self.config.lm_obj_dict[cls_type]))
        if dataset_name == 'train':
            self.add_noise = True
            real_img_pth = os.path.join(
                self.cls_root, "train.txt"
            )
            self.real_lst = self.bs_utils.read_lines(real_img_pth)

            rnd_img_ptn = os.path.join(
                self.root, 'renders/%s/*.pkl' % cls_type
            )
            self.rnd_lst = glob(rnd_img_ptn)
            print("render data length: ", len(self.rnd_lst))
            if len(self.rnd_lst) == 0:
                warning = "Warning: "
                warning += "Trainnig without rendered data will hurt model performance \n"
                warning += "Please generate rendered data from https://github.com/ethnhe/raster_triangle.\n"
                print(colored(warning, "red", attrs=['bold']))

            fuse_img_ptn = os.path.join(
                self.root, 'fuse/%s/*.pkl' % cls_type
            )
            self.fuse_lst = glob(fuse_img_ptn)
            print("fused data length: ", len(self.fuse_lst))
            if len(self.fuse_lst) == 0:
                warning = "Warning: "
                warning += "Trainnig without fused data will hurt model performance \n"
                warning += "Please generate fused data from https://github.com/ethnhe/raster_triangle.\n"
                print(colored(warning, "red", attrs=['bold']))

            self.all_lst = self.real_lst + self.rnd_lst + self.fuse_lst
            self.minibatch_per_epoch = len(self.all_lst) // self.config.mini_batch_size
        else:
            self.add_noise = False
            tst_img_pth = os.path.join(self.cls_root, "test.txt")
            self.tst_lst = self.bs_utils.read_lines(tst_img_pth)
            self.all_lst = self.tst_lst
        print("{}_dataset_size: ".format(dataset_name), len(self.all_lst))

    def real_syn_gen(self, real_ratio=0.3):
        if len(self.rnd_lst+self.fuse_lst) == 0:
            real_ratio = 1.0
        if self.rng.rand() < real_ratio:  # real
            n_imgs = len(self.real_lst)
            idx = self.rng.randint(0, n_imgs)
            pth = self.real_lst[idx]
            return pth
        else:
            if len(self.fuse_lst) > 0 and len(self.rnd_lst) > 0:
                fuse_ratio = 0.4
            elif len(self.fuse_lst) == 0:
                fuse_ratio = 0.
            else:
                fuse_ratio = 1.
            if self.rng.rand() < fuse_ratio:
                idx = self.rng.randint(0, len(self.fuse_lst))
                pth = self.fuse_lst[idx]
            else:
                idx = self.rng.randint(0, len(self.rnd_lst))
                pth = self.rnd_lst[idx]
            return pth

    def real_gen(self):
        n = len(self.real_lst)
        idx = self.rng.randint(0, n)
        item = self.real_lst[idx]
        return item

    def rand_range(self, rng, lo, hi):
        return rng.rand()*(hi-lo)+lo

    def gaussian_noise(self, rng, img, sigma):
        """add gaussian noise of given sigma to image"""
        img = img + rng.randn(*img.shape) * sigma
        img = np.clip(img, 0, 255).astype('uint8')
        return img

    def linear_motion_blur(self, img, angle, length):
        """:param angle: in degree"""
        rad = np.deg2rad(angle)
        dx = np.cos(rad)
        dy = np.sin(rad)
        a = int(max(list(map(abs, (dx, dy)))) * length * 2)
        if a <= 0:
            return img
        kern = np.zeros((a, a))
        cx, cy = a // 2, a // 2
        dx, dy = list(map(int, (dx * length + cx, dy * length + cy)))
        cv2.line(kern, (cx, cy), (dx, dy), 1.0)
        s = kern.sum()
        if s == 0:
            kern[cx, cy] = 1.0
        else:
            kern /= s
        return cv2.filter2D(img, -1, kern)

    def rgb_add_noise(self, img):
        rng = self.rng
        # apply HSV augmentor
        if rng.rand() > 0:
            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.uint16)
            hsv_img[:, :, 1] = hsv_img[:, :, 1] * self.rand_range(rng, 1-0.25, 1+.25)
            hsv_img[:, :, 2] = hsv_img[:, :, 2] * self.rand_range(rng, 1-.15, 1+.15)
            hsv_img[:, :, 1] = np.clip(hsv_img[:, :, 1], 0, 255)
            hsv_img[:, :, 2] = np.clip(hsv_img[:, :, 2], 0, 255)
            img = cv2.cvtColor(hsv_img.astype(np.uint8), cv2.COLOR_HSV2BGR)

        if rng.rand() > 0.8:  # motion blur
            r_angle = int(rng.rand() * 360)
            r_len = int(rng.rand() * 15) + 1
            img = self.linear_motion_blur(img, r_angle, r_len)

        if rng.rand() > 0.8:
            if rng.rand() > 0.2:
                img = cv2.GaussianBlur(img, (3, 3), rng.rand())
            else:
                img = cv2.GaussianBlur(img, (5, 5), rng.rand())

        return np.clip(img, 0, 255).astype(np.uint8)

    def add_real_back(self, rgb, labels, dpt, dpt_msk):
        real_item = self.real_gen()
        with Image.open(os.path.join(self.cls_root, "depth", real_item+'.png')) as di:
            real_dpt = np.array(di)
        with Image.open(os.path.join(self.cls_root, "mask", real_item+'.png')) as li:
            bk_label = np.array(li)
        bk_label = (bk_label < 255).astype(rgb.dtype)
        if len(bk_label.shape) > 2:
            bk_label = bk_label[:, :, 0]
        with Image.open(os.path.join(self.cls_root, "rgb", real_item+'.png')) as ri:
            back = np.array(ri)[:, :, :3] * bk_label[:, :, None]
        dpt_back = real_dpt.astype(np.float32) * bk_label.astype(np.float32)

        if self.rng.rand() < 0.6:
            msk_back = (labels <= 0).astype(rgb.dtype)
            msk_back = msk_back[:, :, None]
            rgb = rgb * (msk_back == 0).astype(rgb.dtype) + back * msk_back

        dpt = dpt * (dpt_msk > 0).astype(dpt.dtype) + \
            dpt_back * (dpt_msk <= 0).astype(dpt.dtype)
        return rgb, dpt

    def dpt_2_pcld(self, dpt, cam_scale, K):
        if len(dpt.shape) > 2:
            dpt = dpt[:, :, 0]
        dpt = dpt.astype(np.float32) / cam_scale
        msk = (dpt > 1e-8).astype(np.float32)
        row = (self.ymap - K[0][2]) * dpt / K[0][0]
        col = (self.xmap - K[1][2]) * dpt / K[1][1]
        dpt_3d = np.concatenate(
            (row[..., None], col[..., None], dpt[..., None]), axis=2
        )
        dpt_3d = dpt_3d * msk[:, :, None]
        return dpt_3d

    def get_item(self, item_name):
        flag=1
        if "pkl" in item_name:
            data = pkl.load(open(item_name, "rb"))
            dpt_mm = data['depth'] * 1000.
            rgb = data['rgb']
            labels = data['mask']
            K = data['K']
            RT = data['RT']
            rnd_typ = data['rnd_typ']
            R = RT[:, :3]
            T = RT[:, 3]
            if rnd_typ == "fuse":
                labels = (labels == self.cls_id).astype("uint8")
            else:
                labels = (labels > 0).astype("uint8")
            flag=1
        else:
            with Image.open(os.path.join(self.cls_root, "depth/{}.png".format(item_name))) as di:
                dpt_mm = np.array(di)
            with Image.open(os.path.join(self.cls_root, "mask/{}.png".format(item_name))) as li:
                labels = np.array(li)
            with Image.open(os.path.join(self.cls_root, "rgb/{}.png".format(item_name))) as ri:
                if self.add_noise:
                    ri = self.trancolor(ri)
                rgb = np.array(ri)[:, :, :3]
            meta = self.meta_lst[int(item_name)]
            meta_f = meta
            if self.cls_id == 2:
                for i in range(0, len(meta)):
                    if meta[i]['obj_id'] == 2:
                        meta = meta[i]
                        meta_f = meta
                        break
            else:
                meta = meta[0]
                meta_f = meta
            R = np.resize(np.array(meta['cam_R_m2c']), (3, 3))
            T = np.array(meta['cam_t_m2c']) / 1000.0
            RT = np.concatenate((R, T[:, None]), axis=1)
            rnd_typ = 'real'
            K = self.config.intrinsic_matrix["linemod"]
            flag=2

        #print(labels.shape)

        mask_depth = ma.getmaskarray(ma.masked_not_equal(dpt_mm, 0))
        if flag == 1:
            mask_label = ma.getmaskarray(ma.masked_equal(labels, np.array(1)))
        else:
            if self.dataset_name != 'train':
                mask_label = ma.getmaskarray(ma.masked_equal(labels, np.array(255)))
            else:
                mask_label = ma.getmaskarray(ma.masked_equal(labels, np.array([255, 255, 255])))[:, :, 0]
        if len(mask_label.shape) == 3:
            mask_label = mask_label[:, :, 0]

        mask = mask_label * mask_depth
        labels = (labels > 0).astype("uint8")

        cam_scale = 1000.0
        if len(labels.shape) > 2:
            labels = labels[:, :, 0]
        rgb_labels = labels.copy()
        # if self.add_noise and rnd_typ == 'render':
        if self.add_noise and rnd_typ != 'real':
            if rnd_typ == 'render' or self.rng.rand() < 0.8:
                rgb = self.rgb_add_noise(rgb)
                rgb_labels = labels.copy()
                msk_dp = dpt_mm > 1e-6
                rgb, dpt_mm = self.add_real_back(rgb, rgb_labels, dpt_mm, msk_dp)
                if self.rng.rand() > 0.8:
                    rgb = self.rgb_add_noise(rgb)

        dpt_mm = dpt_mm.copy().astype(np.uint16)
        nrm_map = normalSpeed.depth_normal(
            dpt_mm, K[0][0], K[1][1], 5, 2000, 20, False
        )

        # if self.DEBUG:
        #     show_nrm_map = ((nrm_map + 1.0) * 127).astype(np.uint8)
        #     imshow("nrm_map", show_nrm_map)

        img = np.transpose(rgb, (2, 0, 1))
        img_masked = img
        nrm_map = np.array(nrm_map)[:, :, :3]
        nrm_map = np.transpose(nrm_map, (2, 0, 1))

        if self.dataset_name != 'train' or flag==1:
            rmin, rmax, cmin, cmax = get_bbox(mask_to_bbox(mask_label))
        else:
            rmin, rmax, cmin, cmax = get_bbox(meta_f['obj_bb'])


        img_masked = img_masked[:, rmin:rmax, cmin:cmax]
        nrm_map = nrm_map[:, rmin:rmax, cmin:cmax]
        rgb_size = img_masked.shape

        
        add_t = np.array([random.uniform(-self.noise_trans, self.noise_trans) for i in range(3)])

        dpt_m = dpt_mm.astype(np.float32) / cam_scale
        dpt_xyz = self.dpt_2_pcld(dpt_m, 1.0, K)
        dpt_xyz[np.isnan(dpt_xyz)] = 0.0
        dpt_xyz[np.isinf(dpt_xyz)] = 0.0

        msk_dp = dpt_mm > 1e-6
        choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
        #print(choose.shape)
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

        depth_masked = dpt_mm[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        xmap_masked = self.xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        ymap_masked = self.ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)

        choose = np.array([choose])
        rgb_size = np.array([rgb_size])
        # img_masked_choose = img_masked.reshape(-1, 3)[choose, :].astype(np.float32)
        # nrm_map_choose = nrm_map.reshape(-1, 3)[choose, :].astype(np.float32)

        cam_scale = 1.0
        pt2 = depth_masked / cam_scale
        pt0 = (ymap_masked - self.cam_cx) * pt2 / self.cam_fx
        pt1 = (xmap_masked - self.cam_cy) * pt2 / self.cam_fy
        cloud = np.concatenate((pt0, pt1, pt2), axis=1)
        cloud = cloud / 1000.0

        if self.add_noise:
            cloud = np.add(cloud, add_t)


        model_points = self.pt / 1000.0
        dellist = [j for j in range(0, len(model_points))]
        dellist = random.sample(dellist, len(model_points) - self.num_pt_mesh_small)
        model_points = np.delete(model_points, dellist, axis=0)

        h, w = rgb_labels.shape
        rgb = np.transpose(rgb, (2, 0, 1))  # hwc2chw

        xyz_lst = [dpt_xyz.transpose(2, 0, 1)]  # c, h, w
        
        msk_lst = [dpt_xyz[2, :, :] > 1e-8]

        for i in range(3):
            scale = pow(2, i+1)
            nh, nw = h // pow(2, i+1), w // pow(2, i+1)
            ys, xs = np.mgrid[:nh, :nw]
            xyz_lst.append(xyz_lst[0][:, ys*scale, xs*scale])
            msk_lst.append(xyz_lst[-1][2, :, :] > 1e-8)

        # xyz_lst = numpy.array(xyz_lst)
        # xyz_masked = xyz_lst[:, rmin:rmax, cmin:cmax]
        
        target_r = R
        target_t = T
        target = np.dot(model_points, target_r.T)
        if self.add_noise:
            target = np.add(target, target_t + add_t)
            out_t = target_t + add_t
        else:
            target = np.add(target, target_t )
            out_t = target_t


        # item_dict = dict(
        #     cloud=torch.from_numpy(cloud.astype(np.float32)),  # 
        #     choose=torch.LongTensor(choose.astype(np.int32)),  #
        #     img_masked=self.norm(torch.from_numpy(img_masked.astype(np.float32))),  # 
        #     target=torch.from_numpy(target.astype(np.float32)),  # 
        #     model_points=torch.from_numpy(model_points.astype(np.float32)),  # 
        #     idx = torch.LongTensor([self.cls_id]),
        #     nrm_map=torch.from_numpy(nrm_map.astype(np.float32)),  # 
        # )
        
        return torch.from_numpy(cloud.astype(np.float32)), \
               torch.LongTensor(choose.astype(np.int32)), \
               self.norm(torch.from_numpy(img_masked.astype(np.float32))), \
               torch.from_numpy(target.astype(np.float32)), \
               torch.from_numpy(model_points.astype(np.float32)), \
               torch.LongTensor([self.cls_id]), \
               torch.from_numpy(nrm_map.astype(np.float32))



    def __len__(self):
        return len(self.all_lst)

    def get_sym_list(self):
        return self.symmetry_obj_idx

    def get_num_points_mesh(self):
            return self.num_pt_mesh_small

    def __getitem__(self, idx):
        if self.dataset_name == 'train':
            item_name = self.real_syn_gen()
            data = self.get_item(item_name)
            while data is None:
                item_name = self.real_syn_gen()
                data = self.get_item(item_name)
            return data
        else:
            item_name = self.all_lst[idx]
            return self.get_item(item_name)


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

if __name__ == "__main__":
    cls = 'ape'
    ds = Dataset('train', True, 0.03, cls, DEBUG=True)
    datum = ds.__getitem__(0)
    cloud = datum['cloud']
    choose = datum['choose']
    img_masked = datum['img_masked']
    target = datum['target']
    model_points = datum['model_points']
    nrm_map = datum['nrm_map']
    print("cloud:", cloud, "\n",
        "choose:", choose, "\n",
        "img_masked:", img_masked, "\n",
        "target:", target, "\n",
        "model_points:", model_points, "\n",
        "nrm_map:", nrm_map, "\n"
        )

    

    #ds['test'] = Dataset('test', cls, DEBUG=True)
