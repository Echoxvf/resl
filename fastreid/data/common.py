# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from torch.utils.data import Dataset
import random
from .data_utils import read_image
import torch
from collections import defaultdict

class CommDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, img_items, transform=None, relabel=True):
        self.img_items = img_items
        self.transform = transform
        self.relabel = relabel
        self.cam_images = None
        pid_set = set()
        cam_set = set()
        for i in img_items:
            pid_set.add(i[1])
            cam_set.add(i[2])

        self.pids = sorted(list(pid_set))
        self.cams = sorted(list(cam_set))
        if relabel:
            self.pid_dict = dict([(p, i) for i, p in enumerate(self.pids)])
            self.cam_dict = dict([(p, i) for i, p in enumerate(self.cams)])

    def __len__(self):
        return len(self.img_items)

    def __getitem__(self, index):
        img_item = self.img_items[index]
        img_path = img_item[0]
        pid = img_item[1]
        camid = img_item[2]
        img = read_image(img_path)
        if self.transform is not None: img = self.transform(img)
        if self.relabel:
            pid = self.pid_dict[pid]
            camid = self.cam_dict[camid]
        return {
            "images": img,
            "targets": pid,
            "camids": camid,
            "img_paths": img_path,
            "indexs":index
        }

    @property
    def num_classes(self):
        return len(self.pids)

    @property
    def num_cameras(self):
        return len(self.cams)


class CommMBDataset(Dataset):
    """Image Person ReID Dataset for multiple input"""

    def __init__(self, img_items, transform=None, reliability=None,  cam_style=None, relabel=True):
        self.img_items = img_items
        self.transform = transform
        self.relabel = relabel
        self.cam_imgs = cam_style
        self.reliabilities = reliability
        pid_set = set()
        cam_set = set()
        for i in img_items:
            pid_set.add(i[1])
            cam_set.add(i[2])
        self.pids = sorted(list(pid_set))
        self.cams = sorted(list(cam_set))
        self.aug_prob = 0.1
        if relabel:
            self.pid_dict = dict([(p, i) for i, p in enumerate(self.pids)])
            self.cam_dict = dict([(p, i) for i, p in enumerate(self.cams)])
    def __len__(self):
        return len(self.img_items)

    def __getitem__(self, index):
        img_item = self.img_items[index]
        img_path = img_item[0]
        pid = img_item[1]
        camid = img_item[2]
        reliability = self.reliabilities[index]
        img = read_image(img_path)
        if self.relabel:
            pid = self.pid_dict[pid]
            camid = self.cam_dict[camid]
        if random.random()<self.aug_prob:
            img1 = random.choice(self.cam_imgs[img_path.split("/")[-1][:-4]]) #[-20:-4]
            img1 = read_image(img1)
        else:
            img1 = img
        if random.random()<self.aug_prob:
            img2 = random.choice(self.cam_imgs[img_path.split("/")[-1][:-4]]) #[-20:-4]
            img2 = read_image(img2)
        else:
            img2 = img
        if random.random()<self.aug_prob:
            img3 = random.choice(self.cam_imgs[img_path.split("/")[-1][:-4]]) #[-20:-4]
            img3 = read_image(img3)
        else:
            img3 = img
        if self.transform is not None: 
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
            imgs = torch.stack([img1,img2,img3])
        return {
            "images": imgs,
            "targets": pid,
            "camids": camid,
            "img_paths": img_path,
            "indexs":index,
            "reliability": reliability
        }

    @property
    def num_classes(self):
        return len(self.pids)

    @property
    def num_cameras(self):
        return len(self.cams)


class PseudoDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, img_items, transform=None, relabel=True):
        self.img_items = img_items
        self.transform = transform
        self.relabel = relabel

        pid_set = set()
        cam_set = set()
        for i in img_items:
            pid_set.add(i[1])
            cam_set.add(i[2])

        self.pids = sorted(list(pid_set))
        self.cams = sorted(list(cam_set))
        if relabel:
            self.pid_dict = dict([(p, i) for i, p in enumerate(self.pids)])
            self.cam_dict = dict([(p, i) for i, p in enumerate(self.cams)])

    def __len__(self):
        return len(self.img_items)

    def __getitem__(self, index):
        img_item = self.img_items[index]
        img_path = img_item[0]
        pid = img_item[1]
        camid = img_item[2]
        reliability=img_item[3]
        img = read_image(img_path)
        if self.transform is not None: img = self.transform(img)
        if self.relabel:
            pid = self.pid_dict[pid]
            camid = self.cam_dict[camid]
        return {
            "images": img,
            "targets": pid,
            "camids": camid,
            "img_paths": img_path,
            "indexs":index,
            "reliability":reliability
        }

    @property
    def num_classes(self):
        return len(self.pids)

    @property
    def num_cameras(self):
        return len(self.cams)