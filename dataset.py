import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils import data
from PIL import Image
import numpy as np
import os
from collections import OrderedDict
import util.util as util
import cv2

from PIL import Image
from base_dataset import BaseDataset, get_params, get_transform
import json

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, stop=10000):
    images = []
    count = 0
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
                count += 1
            if count >= stop:
                return images
    return images

class UnpairedDepthDataset(data.Dataset):
    def __init__(self, root, root2, opt, transforms_r=None, mode='train', midas=False, depthroot=''):

        self.root = root
        self.mode = mode
        self.midas = midas

        all_img = make_dataset(self.root)

        self.depth_maps = 0
        if self.midas:

            depth = []
            print(depthroot)
            if os.path.exists(depthroot):
                depth = make_dataset(depthroot)
            else:
                print('could not find %s'%depthroot)
                import sys
                sys.exit(0)

            newimages = []
            self.depth_maps = []

            for dmap in depth:
                lastname = os.path.basename(dmap)
                trainName1 = os.path.join(self.root, lastname)
                trainName2 = os.path.join(self.root, lastname.split('.')[0] + '.jpg')
                if (os.path.exists(trainName1)):
                    newimages += [trainName1]
                elif (os.path.exists(trainName2)):
                    newimages += [trainName2]
            print('found %d correspondences' % len(newimages))

            self.depth_maps = depth
            all_img = newimages

        self.data = all_img
        self.mode = mode

        self.transform_r = transforms.Compose(transforms_r)

        self.opt = opt
        
        if mode == 'train':
            
            self.img2 = make_dataset(root2)

            if len(self.data) > len(self.img2):
                howmanyrepeat = (len(self.data) // len(self.img2)) + 1
                self.img2 = self.img2 * howmanyrepeat
            elif len(self.img2) > len(self.data):
                howmanyrepeat = (len(self.img2) // len(self.data)) + 1
                self.data = self.data * howmanyrepeat
                self.depth_maps = self.depth_maps * howmanyrepeat
            

            cutoff = min(len(self.data), len(self.img2))

            self.data = self.data[:cutoff] 
            self.img2 = self.img2[:cutoff] 

            self.min_length =cutoff
        else:
            self.min_length = len(self.data)


    def __getitem__(self, index):

        img_path = self.data[index]

        basename = os.path.basename(img_path)
        base = basename.split('.')[0]

        img_r = Image.open(img_path).convert('RGB')
        transform_params = get_params(self.opt, img_r.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.opt.input_nc == 1), norm=False)
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.opt.output_nc == 1), norm=False)        

        if self.mode != 'train':
            A_transform = self.transform_r

        img_r = A_transform(img_r )

        B_mode = 'L'
        if self.opt.output_nc == 3:
            B_mode = 'RGB'

        img_depth = 0
        if self.midas:
            img_depth = cv2.imread(self.depth_maps[index])
            img_depth = A_transform(Image.fromarray(img_depth.astype(np.uint8)).convert('RGB'))


        img_normals = 0
        label = 0

        input_dict = {'r': img_r, 'depth': img_depth, 'path': img_path, 'index': index, 'name' : base, 'label': label}

        if self.mode=='train':
            cur_path = self.img2[index]
            cur_img = B_transform(Image.open(cur_path).convert(B_mode))
            input_dict['line'] = cur_img

        return input_dict

    def __len__(self):
        return self.min_length

