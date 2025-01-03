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

import numpy as np
import os
import glob
import pickle
import random
import shutil

def get_subdirectories(directory):
    entries = os.listdir(directory)
    subdirectories = [entry for entry in entries if os.path.isdir(os.path.join(directory, entry))]
    return subdirectories


import time
def check_batch_not_resizable(root_dir):
    """
    1. The folders are named according to the corresponding video file names.
    2. If the number of images exceeds the length of the segmented whisper features, the surplus images will be deleted.
    3. Folders containing an insufficient number of video frames will be deleted.
    """
    dirs = get_subdirectories(root_dir)
    for ind, dir_ in enumerate(dirs):
        depth_list = sorted(glob.glob(os.path.join(root_dir, dir_, "*_depth.jpg")))
        if len(depth_list) == 0:
            print("error: no images", ind, dir_)
            shutil.rmtree(os.path.join(root_dir, dir_))
            print("had deleted dir: ", os.path.join(root_dir, dir_))
            continue
        if len(depth_list) < 5+2:
            print("error: number of images less than 7", ind, dir_)
            shutil.rmtree(os.path.join(root_dir, dir_))
            print("had deleted dir: ", os.path.join(root_dir, dir_))
            continue

        num_imgs = int(depth_list[-1].split("/")[-1].split("_")[0])+1
        sub_dir = os.path.join(root_dir, dir_)
        whisper = np.load(os.path.join(root_dir, dir_, "whisper.pkl"), allow_pickle=True)
        num_chunks = len(whisper)
        print(ind, dir_)

        if num_imgs > num_chunks:
           print('error: imgs_len > whisper_len', ind, dir_, num_imgs, num_chunks)
           for i in range(num_chunks, num_imgs):
               if os.path.exists(os.path.join(sub_dir, f"{i:08d}_depth.jpg")):
                   os.remove(os.path.join(sub_dir, f"{i:08d}_depth.jpg"))
               if os.path.exists(os.path.join(sub_dir, f"{i:08d}.jpg")):
                   os.remove(os.path.join(sub_dir, f"{i:08d}.jpg"))
           print("have deleted files")


def check_batch_valid_indices(root_dir):
    """
    Generate a valid index file that specifies the video frames used for training
    """
    dirs = get_subdirectories(root_dir)
    for ind, dir_ in enumerate(dirs):
        depth_list = sorted(glob.glob(os.path.join(root_dir, dir_, "*_depth.jpg")))
        num = len(depth_list)
        if num == 0:
            print("warning: no images", ind, dir_)
            shutil.rmtree(os.path.join(root_dir, dir_))
            print("have deleted dir: ", os.path.join(root_dir, dir_))
            continue
        num_imgs = int(depth_list[-1].split("/")[-1].split("_")[0])+1
        print(dir_, num, num_imgs)
        depth_rgb_list = glob.glob(os.path.join(root_dir, dir_, "*.jpg"))
        if num*2 != len(depth_rgb_list):
            print("error: not match", dir_)
        if num < num_imgs:
            print("warning: failed to detect faces in some frames", dir_)

        valid_indices = []
        for depth_file in depth_list:
            depth_ind = int(depth_file.split("/")[-1].split("_")[0])
            valid_indices.append(depth_ind)
        np.save(os.path.join(root_dir, dir_, "valid_indices.npy"), np.array(valid_indices))


def check_split_whisper(root_dir):
    """
    Splitting the chunked Whisper features into smaller files can further improve data loading speed, thereby reducing the overall training time.
    """
    dirs = get_subdirectories(root_dir)
    for ind, dir_ in enumerate(dirs):
        t1 = time.time()
        valid_indices = np.load(os.path.join(root_dir, dir_, "valid_indices.npy"))
        whisper = np.array(np.load(os.path.join(root_dir, dir_, 'whisper.pkl'), allow_pickle=True))
        for ind in valid_indices:
            whisper_file = os.path.join(root_dir, dir_, f"{ind:08d}_whisper.npy")
            np.save(whisper_file, whisper[ind].squeeze())
        t2 = time.time()
        print(dir_, t2-t1)


def get_list(root_dir, face_file):
    dirs = get_subdirectories(root_dir)
    fid = open(face_file, "w")
    for dir_ in dirs:
        sub_dir = os.path.join(root_dir, dir_)
        face_files = glob.glob(os.path.join(sub_dir, "*_face.jpg"))
        for f in face_files:
            fid.write(f"{f}\n")
    fid.close()

if __name__ == "__main__":
    root_dir = "xx"
    face_list = "face_list.txt"
    check_batch_not_resizable(root_dir)
    check_batch_valid_indices(root_dir)
    check_split_whisper(root_dir)
    get_list(root_dir, face_list)