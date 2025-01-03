# Copyright (c) 2025 JD.com, Inc. and affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
from torchvision import transforms
import cv2
from PIL import Image
from torchvision.transforms import functional as F
from torchvision.transforms import RandomResizedCrop
import random
import time

def add_transform():
    transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0),
    #transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2)),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor()
    ])

    return transform

# 此处一定不能有水平翻转的，与上面的区别
# 要使用水平翻转，要与下面的RandomHorizontalFlip联合使用
def add_transform_wo_flip():
    transform = transforms.Compose([
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0),
    #transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2)),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor()
    ])

    return transform


class RandomResizedCropWithParams(RandomResizedCrop):
    def forward(self, img):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        self.params = (i, j, h, w)
        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation)

    def apply_with_params(self, img):
        #print(self.params)
        i, j, h, w = self.params
        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation)


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, gt, ref, mask):
        if random.random() < self.p:
            return gt.transpose(Image.FLIP_LEFT_RIGHT), ref.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
        return gt, ref, mask


mouth_point_indices = [61,185,40,39,37,0,267,269,270,409,291,
                76,184,74,73,72,11,302,303,304,408,306,
                62,183,42,41,38,12,268,271,272,407,292,
                78,191,80,81,82,13,312,311,310,415,308,
                   95,88,178,87,14,317,402,318,324,
                   96,89,179,86,15,316,403,319,325,
                   77,90,180,85,16,315,404,320,307,
                  146,91,181,84,17,314,405,321,375
                ]

mouth_region_indices = [61,185,40,39,37,0,267,269,270,409,291,375,321,405,314,17,84,181,91,146
                ]
def create_mouth_mask(image, mouth_landmarks):
    mask = np.zeros_like(image, dtype=np.uint8)
    points = np.array(mouth_landmarks, dtype=np.int32)
    cv2.fillPoly(mask, [points], (1, 1, 1))
    return mask

class FaceDataset(Dataset):
    def __init__(self, root_dir, index_file, img_size=256):
        super(FaceDataset, self).__init__()
        self.root_dir = root_dir
        self.index_file = index_file
        # 256 or 512
        self.img_size = img_size
        self.transform = add_transform_wo_flip()
        with open(os.path.join(self.root_dir, self.index_file), 'r') as fid:
            self.train_data = fid.readlines()
            self.train_data = [ele.strip() for ele in self.train_data]

    def __getitem__(self, index):
        # current frame
        face_file = os.path.join(self.root_dir, self.train_data[index])
        gt_face = cv2.resize(cv2.imread(face_file), (self.img_size, self.img_size))
        num = int(face_file.split('/')[-1].replace("_face.jpg",""))
        whisper_file = face_file.replace("_face.jpg", "_whisper.npy")
        whisper = np.load(whisper_file)
        # depth
        #if random.choice([True, False]):
        if False:
            depth_file = face_file.replace("_face.jpg", "_depth.jpg")
            lmk_file = face_file.replace("_face.jpg", "_lm.npy")
            
            if os.path.exists(depth_file) and os.path.exists(lmk_file):
                try:
                    gt_depth = cv2.imread(depth_file)
                    lmk = np.load(lmk_file)
                    lip_mask = create_mouth_mask(gt_depth, lmk[mouth_region_indices,:])
                    gt_depth = gt_depth * lip_mask
                    gt_depth = cv2.resize(gt_depth,(self.img_size, self.img_size))
                    gt_depth = random_translate(gt_depth, int(self.img_size*0.02), int(self.img_size*0.02))
                except Exception as e:
                    print(f"error:{e}, {depth_file}")
                    gt_depth = np.zeros((self.img_size, self.img_size, 3), np.uint8)  
            else:
                gt_depth = np.zeros((self.img_size, self.img_size, 3), np.uint8)  
        else:
            gt_depth = np.zeros((self.img_size, self.img_size, 3), np.uint8)
        
        # random frame
        cur_path = face_file
        cur_ind = num
        valid_indices = np.load(os.path.join(os.path.dirname(face_file),"valid_indices.npy"))
        rand_ind = np.random.choice(valid_indices)
        rand_path = os.path.join(os.path.dirname(cur_path), f"{rand_ind:08d}_face.jpg")
        while abs(rand_ind-cur_ind) < 5:
            rand_ind = np.random.choice(valid_indices)
            rand_path = os.path.join(os.path.dirname(cur_path), f"{rand_ind:08d}_face.jpg")
            #print(index, cur_path, rand_path, cur_ind, rand_ind)
        
        ref_face = cv2.resize(cv2.imread(rand_path), (self.img_size, self.img_size))

        # to tensor
        whisper = torch.tensor(whisper)
        #gt_face = torch.tensor(gt_face).to(torch.float)
        #ref_face = torch.tensor(ref_face).to(torch.float)

         
        # S1: 随机裁剪
        transform_crop_resize = RandomResizedCropWithParams(size=self.img_size, scale=(0.8, 1.0), ratio=(1, 1))


        # gt_face 
        gt_face_convert = cv2.cvtColor(gt_face, cv2.COLOR_BGR2RGB)
        gt_face_pil = Image.fromarray(gt_face_convert)
        gt_face_pil = transform_crop_resize(gt_face_pil)

        # ref_face
        ref_face_convert = cv2.cvtColor(ref_face, cv2.COLOR_BGR2RGB)
        ref_face_pil = Image.fromarray(ref_face_convert)
        ref_face_pil = transform_crop_resize.apply_with_params(ref_face_pil)

        # depth同样也要对应随机裁剪与resize
        depth_convert = cv2.cvtColor(gt_depth, cv2.COLOR_BGR2RGB)
        depth_pil = Image.fromarray(depth_convert)
        depth_pil = transform_crop_resize.apply_with_params(depth_pil)
        # depth = cv2.cvtColor(np.array(depth_pil), cv2.COLOR_RGB2BGR)
        depth = transforms.ToTensor()(depth_pil)
        
        # S2: 随机水平翻转。 gt_face, ref_face, mask统一水平翻转或者保持不变
        #gt_face_pil, ref_face_pil, mask_pil = transform_flip(gt_face_pil, ref_face_pil, mask_pil)

        # S3: 其它增强
        # 图片上下拼接为了可以水平翻转(此处水平或者上下拼接都可以) 
        cat_face = Image.new('RGB', (self.img_size,2*self.img_size))
        cat_face.paste(gt_face_pil, (0,0))
        cat_face.paste(ref_face_pil, (0,self.img_size))
        gt_ref_face_enhance = self.transform(cat_face)

        return whisper, gt_ref_face_enhance, depth

    def __len__(self):
        return len(self.train_data)
