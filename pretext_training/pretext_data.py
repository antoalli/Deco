#!/usr/bin/python -tt
# -*- coding: utf-8 -*-
"""
@Author: Antonio Alliegro
@Contact: antonio.alliegro@polito.it
@File:
@Time:
"""

from __future__ import print_function
import torch.utils.data as data
import os
import os.path as osp
import json
import numpy as np
import torch


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


class PretextDataset(data.Dataset):
    def __init__(self,
                 root,
                 npoints=2048,
                 task='cls',
                 class_choice=None,
                 split='train',
                 normalize=True,
                 noise_mean=0.0,
                 noise_std=0.01,
                 centroids=None,
                 crop_point_num=512,
                 num_positive_samples=2,
                 transforms=None
                 ):
        self.npoints = npoints
        self.root = root
        self.catfile = osp.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        assert task in ['denoise', 'contrast'], f"Wrong task chosen: {task}"
        self.task = task
        self.normalize = normalize

        # Contrast pretext
        self.transforms = transforms
        self.crop_point_num = crop_point_num  # num points to be cropped
        self.num_positive_samples = num_positive_samples  # num positive samples for contrastive learning

        # Denoise pretext
        self.noise_mean = noise_mean
        self.noise_std = noise_std

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]

        if class_choice is not None:
            self.cat = {k: v for k, v in self.cat.items() if k in class_choice}

        self.meta = {}
        with open(osp.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
            train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(osp.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
            val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(osp.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str(d.split('/')[2]) for d in json.load(f)])

        for item in self.cat:
            self.meta[item] = []
            dir_point = osp.join(self.root, self.cat[item], 'points')
            dir_seg = osp.join(self.root, self.cat[item], 'points_label')
            fns = sorted(os.listdir(dir_point))
            if split == 'trainval':
                fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
            elif split == 'train':
                fns = [fn for fn in fns if fn[0:-4] in train_ids]
            elif split == 'val':
                fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif split == 'test':
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                raise ValueError(f"Unknown split: {split}")

            for fn in fns:
                token = (osp.splitext(osp.basename(fn))[0])
                self.meta[item].append((osp.join(dir_point, token + '.pts'), osp.join(dir_seg, token + '.seg'),
                                        self.cat[item], token))
        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn[0], fn[1], fn[2], fn[3]))

        self.classes = dict(zip(sorted(self.cat), range(len(self.cat))))
        print('PretextDataset task: ', self.task)
        print('PretextDataset classes: ', self.classes)
        self.cache = {}
        self.cache_size = 18000

    def __getitem__(self, index):
        if index in self.cache:
            point_set, seg, cls, foldername, filename = self.cache[index]
        else:
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            point_set = np.loadtxt(fn[1]).astype(np.float32)
            if self.normalize:
                point_set = pc_normalize(point_set)
            seg = np.loadtxt(fn[2]).astype(np.int64) - 1
            foldername = fn[3]
            filename = fn[4]
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, seg, cls, foldername, filename)

        # sampling
        choice = np.random.choice(len(seg), self.npoints, replace=True)
        point_set = point_set[choice, :]
        seg = seg[choice]

        if self.task == 'denoise':
            # return (noised, clean, cls)
            noise = np.random.normal(loc=self.noise_mean, scale=self.noise_std, size=np.shape(point_set))
            noised = point_set + noise
            noised = torch.from_numpy(noised).float()
            point_set = torch.from_numpy(point_set).float()
            return noised, point_set, cls

        elif self.task == 'contrast':
            assert self.transforms is not None
            """
            for _ in num_positive_samples:
                apply transformations (crop + other spatial + appearance) 
            """
            views = []
            for i in range(self.num_positive_samples):
                views.append(self.transforms(point_set))

            return views, cls

        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.datapath)
