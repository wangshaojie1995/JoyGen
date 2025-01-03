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
from deep3d_facerecon.options.test_options import TestOptions
from deep3d_facerecon.models import create_model
from deep3d_facerecon.util.preprocess import align_img

import numpy as np
from deep3d_facerecon.util.load_mats import load_lm3d
import torch
from scipy.io import loadmat, savemat
from PIL import Image, ImageDraw
from facenet_pytorch import MTCNN
import time
import concurrent.futures
import glob
import shutil
import sys
import cv2
import logging
from datetime import datetime
import random
import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh
import copy

from src.audio2feature import Audio2Feature
import pickle

from mmpose.apis import inference_topdown, init_model
from mmpose.structures import merge_data_samples
from utils.face_detection import FaceAlignment,LandmarksType
import argparse


def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, height, width, num_frames

def draw_points(img, coordinates, point_color=(255,0,0)):
    # point_color = (255,0,0) # 红色(RGB)
    image = copy.deepcopy(img)
    draw = ImageDraw.Draw(image)
    point_radius = 3
    for x, y in coordinates:
        draw.ellipse((int(x) - point_radius, int(y) - point_radius, int(x) + point_radius, int(y) + point_radius), fill=point_color)
    return image


def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interWidth = max(0, xB - xA)
    interHeight = max(0, yB - yA)
    interArea = interWidth * interHeight

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    unionArea = boxAArea + boxBArea - interArea

    iou = interArea / unionArea if unionArea > 0 else 0
    return iou

def merge_bbox_landmark(f, face_land_mark):
    half_face_coord =  face_land_mark[29]#np.mean([face_land_mark[28], face_land_mark[29]], axis=0)
    half_face_dist = np.max(face_land_mark[:,1]) - half_face_coord[1]
    upper_bond = half_face_coord[1]-half_face_dist

    f_landmark = (np.min(face_land_mark[:, 0]), int(upper_bond), np.max(face_land_mark[:, 0]), np.max(face_land_mark[:,1]))
    x1, y1, x2, y2 = f_landmark

    if y2-y1<=0 or x2-x1<=0 or x1<0: # if the landmark bbox is not suitable, reuse the bbox
        return f
    else:
        return f_landmark


def read_data_beta(im_path, lm, lm3d_std, to_tensor=True):
    # to RGB
    im = Image.open(im_path).convert('RGB')
    W, H = im.size
    lm = lm.reshape([-1, 2])
    lm[:, -1] = H - 1 - lm[:, -1]
    trans_params, im, lm, _, crop_params = align_img(im, lm, lm3d_std)
    if to_tensor:
        im = torch.tensor(np.array(im) / 255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        lm = torch.tensor(lm).unsqueeze(0)
    return im, lm, trans_params, crop_params


def do_deep3DRecon_for_UNet_beta(opt, video_path, sub_indices, imgs_list, saved_dir):
    try:
        # 3D landmark
        lm3d_std = load_lm3d(opt.bfm_folder)
        # device
        device = torch.device(f"cuda:{opt.gpu_id}")

        time1 = time.time()
        # mtcnn
        mtcnn = MTCNN(image_size=224, margin=0, min_face_size=50, selection_method='largest', keep_all=False, device=device)

        # model for predicting 3DMM coefficients
        model_3drecon = create_model(opt)
        model_3drecon.setup(opt)
        model_3drecon.device = device
        model_3drecon.parallelize()
        model_3drecon.eval()

        # dwpose
        pose_model = init_model(opt.dwpose_config_file, opt.dwpose_model_path, device=device)
        fa = FaceAlignment(LandmarksType._2D, flip_input=False, device=device)

        time2 = time.time()
        print(f"model init: {time2-time1} s")


        for i, img_path in zip(sub_indices, imgs_list):
            t1_total = time.time()
            img_name = os.path.basename(img_path).split(".")[0]
            crop_img_path = os.path.join(saved_dir, img_name+"_face.jpg")
            depth_img_path = os.path.join(saved_dir, img_name + "_depth.jpg")
            lm468_path = os.path.join(saved_dir, img_name+"_lm.npy")
            combine_img_path = os.path.join(saved_dir, img_name + "_combine.jpg")
            coeff_path = os.path.join(saved_dir, img_name + ".mat")

            t1_mtcnn = time.time()
            img = Image.open(img_path)
            try:
                boxes, probs, points = mtcnn.detect(img, landmarks=True)
            except Exception as e:
                print(f"error: {str(e)}, {img_path}")
                continue
            t2_mtcnn = time.time()

            if len(boxes) == 0:
                print(f"detect no face, {img_path}")
                continue

            # select the largest face
            p5_mtcnn = np.array(points[0], np.float32)
            box_mtcnn = np.array(boxes[0], np.float32)
            box_select = copy.deepcopy(box_mtcnn)

            t1_mmpose = time.time()
            # musetalk's landmark & bbox
            try:
                frame = cv2.imread(img_path)
                results = inference_topdown(pose_model, np.array(frame))
                results = merge_data_samples(results)
                keypoints = results.pred_instances.keypoints
                face_land_mark= keypoints[0][23:91]
                face_land_mark = face_land_mark.astype(np.int32)
                bbox = fa.get_detections_for_batch(np.expand_dims(frame, axis=0))
                iou = calculate_iou(box_mtcnn, bbox[0])
                if iou > 0.7:
                    box_mmpose = merge_bbox_landmark(bbox[0], face_land_mark)
                    box_select = copy.deepcopy(box_mmpose)
            except Exception as e:
                print(f"error: {e}, {img_path}")
            t2_mmpose = time.time()


            try:
                t1_rec = time.time()
                im_tensor, lm_tensor, trans_params, crop_params = read_data_beta(img_path, np.array(p5_mtcnn), lm3d_std)

                # input
                data = {
                    'imgs': im_tensor,
                    'lms': lm_tensor
                }
                model_3drecon.set_input(data)  # unpack data from data loader
                model_3drecon.test()  # run inference

                # output
                pred_coeffs = model_3drecon.get_coeff()
                pred_depth = model_3drecon.get_depth()
                pred_lm468 = pred_coeffs['lm468']

                s = trans_params[2]
                w0, h0 = img.size  
                w_scaled = int(w0 * s)
                h_scaled = int(h0 * s) 

                img_croped_for_unet = img.crop(
                    (int(box_select[0]),
                     int(box_select[1]),
                     int(box_select[2]),
                     int(box_select[3])))

                w_crop = int(crop_params[5]) - int(crop_params[3])
                h_crop = int(crop_params[6]) - int(crop_params[4])
                depth_original_scale = pred_depth.crop(
                    (-int(crop_params[3]),
                     -int(crop_params[4]),
                     w_crop + w_scaled - int(crop_params[5]),
                     h_crop + h_scaled - int(crop_params[6])))
                depth_original_scale = depth_original_scale.resize((w0, h0))
                depth_croped_for_unet = depth_original_scale.crop(
                    (int(box_select[0]),
                     int(box_select[1]),
                     int(box_select[2]),
                     int(box_select[3])))


                # 468 landmark
                tt = np.tile(np.array([crop_params[3], crop_params[4]]), (468,1))
                tt = np.expand_dims(tt, axis=0)
                lm468_original_scale = (pred_lm468+tt)/s
                lm468_original_scale_croped_for_unet = lm468_original_scale - np.expand_dims(np.tile(np.array([box_select[0], box_select[1]]), (468,1)), axis=0)

                # combined
                #combine_image = Image.blend(img_croped_for_unet, depth_croped_for_unet.convert("RGB"), alpha=0.5)

                # savemat(coeff_path, pred_coeffs)
                depth_croped_for_unet.save(depth_img_path)
                img_croped_for_unet.save(crop_img_path)
                np.save(lm468_path, lm468_original_scale_croped_for_unet[0])

                #combine_image = draw_points(combine_image, lm468_original_scale_croped_for_unet[0], (255,0,0))
                #combine_image.save(combine_img_path)                
                #combine_image.save(combine_img_path)

                t2_rec = time.time()
                t2_total = time.time()
                if i % 5 == 0:
                    print(f"extract depth and rgb image successfully, mtcnn:{t2_mtcnn-t1_mtcnn:.3f}, dwpose:{t2_mmpose-t1_mmpose:.3f}, deep3d_rec: {t2_rec-t1_rec:.3f}, total time: {t2_total-t1_total:.2f}, {i}/{img_path}")
            except Exception as e:
                print(f"error: {str(e)}, {img_path}")
    except Exception as e:
        print(f"error:{str(e)}, {video_path}")



def deal_single_video(opt, video_path, tmp_dir, saved_dir, video_dir_depth_keep=0, multi_processes=False):
    video_file_name = os.path.basename(video_path)
    video_name = video_file_name.split(".")[0]
    #
    if video_dir_depth_keep > 0:
        sub_dirs = os.path.dirname(video_path).split("/")[-video_dir_depth_keep:]
        sub_dir = "".join(sub_dirs)
        tmp_dir = os.path.join(tmp_dir, sub_dir, video_name)
        saved_dir = os.path.join(saved_dir, sub_dir, video_path)
    else:
        tmp_dir = os.path.join(tmp_dir, video_name)
        saved_dir = os.path.join(saved_dir, video_name)
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)

    dst_video_path = os.path.join(tmp_dir, "video.mp4")

    
    # convert the video to 25fps
    cmd = f"ffmpeg -y -loglevel quiet -i {video_path} -r 25 -c:v libx264 {dst_video_path}"
    print(cmd)
    res = os.system(cmd)
    if res != 0:
        print(f"[deal_single_video]failed to convert 25-fps video, {dst_video_path}")
        return
    
    # extract frames of the video
    cmd = f"ffmpeg -y -loglevel quiet -i {dst_video_path} -start_number 0 {tmp_dir}/%08d.png"
    print(cmd)
    res = os.system(cmd)
    if res != 0:
        print(f"[deal_single_video]failed to extract frames from {dst_video_path}")
        return

    input_img_list = sorted(glob.glob(os.path.join(tmp_dir, '*.[jpJP][pnPN]*[gG]')))
    # fps = get_video_fps(dst_video_audio_path)
    # print(f"frames:{len(input_img_list)}, fps:{fps}, {dst_video_audio_path}")

    # extract audio of the video
    cmd = f"ffmpeg -y -loglevel quiet -i {dst_video_path} -q:a 0 -map a {tmp_dir}/audio.mp3"
    print(cmd)
    res = os.system(cmd)
    if res != 0:
        print(f"[deal_single_video]failed to extract audio from {dst_video_path}")
        return

    t1 = time.time()
    num_imgs = len(input_img_list)
    do_deep3DRecon_for_UNet_beta(opt, video_path, range(0,num_imgs), input_img_list, saved_dir)

    
    # extract feature from an audio
    fps,height,width,frames = get_video_info(dst_video_path)
    if fps != 25:
        print(f"[deal_single_videl]failed to covert video to 25 fps")
    audio_processor = Audio2Feature(model_path=opt.whisper_model_path, device=torch.device(f"cuda:{opt.gpu_id}"))
    whisper_feature = audio_processor.audio2feat(f"{tmp_dir}/audio.mp3")
    whisper_chunks = audio_processor.feature2chunks(feature_array=whisper_feature,fps=25)
    with open(f"{saved_dir}/whisper.pkl", 'wb') as f:
        pickle.dump(whisper_chunks, f)
    print(f"whisper feature: {whisper_feature.shape}, whisper feature by chunk: {len(whisper_chunks)}, {whisper_chunks[0].shape}, {video_path}")
    
    t2 = time.time()
    print(f"time: {t2-t1}, frames: {num_imgs}, frame/sec: {num_imgs/(t2-t1)}, {video_path}")
 
    # save disk 
    shutil.rmtree(tmp_dir)



if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    print('arguments: ', opt)
    video_list = glob.glob(os.path.join(opt.video_dir, '*.mp4'))  
    random.seed(0) 
    random.shuffle(video_list)
    print('video files:', video_list)
    print('total videos:', len(video_list))
    for video_path in video_list:
        tmp_dir = os.path.join(opt.result_dir, "tmp_dir")
        saved_dir = opt.result_dir
        deal_single_video(opt, video_path, tmp_dir, saved_dir, video_dir_depth_keep=0)
