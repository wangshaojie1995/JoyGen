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

import argparse
import os
from omegaconf import OmegaConf
import numpy as np
import cv2
import torch
import glob
from tqdm import tqdm
import copy
import time
from PIL import Image
from utils.blending import get_image
import shutil
import json
from diffusers import AutoencoderKL, UNet2DConditionModel
from src.audio2feature import Audio2Feature
from src.pe import PositionalEncoding
from src.model_util import load_model_weight
from src.modules.vae import VAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

def bbox_check(box, height, width):
    clx, cly, crx, cry = box

    if clx<0:
        clx = 0
    if cly<0:
        cly = 0
    if crx > width:
        crx = width
    if cry > height:
        cry = height
    return clx, cly, crx, cry


def create_mouth_mask(image, mouth_landmarks):
    mask = np.zeros_like(image, dtype=np.uint8)
    points = np.array(mouth_landmarks, dtype=np.int32)
    cv2.fillPoly(mask, [points], (1, 1, 1))
    return mask


def data_generator(whisper_chunks, vae_encode_latents, batch_size=8, delay_frame = 0):
    whisper_batch, latent_batch = [], []
    for i, (whisper, latent) in enumerate(zip(whisper_chunks, vae_encode_latents)):
        idx = (i+delay_frame)%len(vae_encode_latents)
        whisper_batch.append(whisper)
        latent_batch.append(latent)

        if len(latent_batch) >= batch_size:
            whisper_batch = np.asarray(whisper_batch)
            latent_batch = torch.cat(latent_batch, dim=0)
            yield whisper_batch, latent_batch
            whisper_batch, latent_batch = [], []

    # the last batch may smaller than batch size
    if len(latent_batch) > 0:
        whisper_batch = np.asarray(whisper_batch)
        latent_batch = torch.cat(latent_batch, dim=0)

        yield whisper_batch, latent_batch



@torch.no_grad()
def main(args):
    # load model weights
    device = torch.device("cuda", args.gpu_id)

    audio_processor = Audio2Feature(model_path=args.whisper_model_path)

    # vae
    vae = VAE(model_path=args.vae_model_path, 
              resized_img=args.img_size, 
              device=device)

    # unet 
    unet = UNet2DConditionModel.from_pretrained(args.unet_model_path).to(device=device)
    #load_model_weight(unet, args.unet_model_path)
        
    pe = PositionalEncoding(d_model=384)
    timesteps = torch.tensor([0], device=device)



    t1 = time.time()
    video_path = args.video_path
    audio_path = args.audio_path
    
    video_basename = os.path.basename(video_path).split('.')[0]
    audio_basename  = os.path.basename(audio_path).split('.')[0]
    pose_path = os.path.join(args.intermediate_dir, video_basename, audio_basename, video_basename)
    print(video_path, audio_path, pose_path)
    if not os.path.exists(pose_path):
        exit(0)

    output_basename = f"{video_basename}#{audio_basename}"
    result_img_save_path = os.path.join(args.result_dir, output_basename) # related to video & audio inputs
    os.makedirs(result_img_save_path,exist_ok =True)
    output_vid_name = os.path.join(args.result_dir, output_basename+".mp4")

    ############################################## extract audio feature ##############################################
    try:
        fps = args.fps
        whisper_feature = audio_processor.audio2feat(audio_path)
        whisper_chunks = audio_processor.feature2chunks(feature_array=whisper_feature,fps=fps)
        print(type(whisper_feature), type(whisper_chunks))
        print(whisper_feature.shape, len(whisper_chunks), whisper_chunks[0].shape)
    except Exception as e:
        print(f"{e}")
        exit(0)
    
    input_imgs_list = glob.glob(os.path.join(pose_path, "*_ori.jpg"))
    indices_img = []
    for img_file in input_imgs_list:
        ind = int(img_file.split("/")[-1].split("_")[0])
        indices_img.append(ind)

    # index start from 0
    ind_max = np.max(indices_img)
    print("number of generated video's frames: ", len(input_imgs_list))
    if ind_max+1 != len(input_imgs_list):
        print("failed to detect face in some frame of input video")
        return

    whisper_feature = whisper_feature[0:ind_max+1]

    
    if not args.enable_pose_driven:
        depth_img = np.zeros((args.img_size, args.img_size, 3), np.uint8)
        latent_depth = vae.get_latents_for_nomask(depth_img)

    sub_dir = os.path.dirname(input_imgs_list[0])
    ori_img_list = []
    crop_img_list = []
    box_list = []
    latent_list = []
    for i in range(ind_max+1):
        ori_img = cv2.imread(os.path.join(sub_dir, f"{i}_ori.jpg"))
        crop_img = cv2.imread(os.path.join(sub_dir, f"{i}_face.jpg"))
        depth_img = cv2.imread(os.path.join(sub_dir, f"{i}_depth_edit_exp.jpg"))
        box = np.load(os.path.join(sub_dir, f"{i}_box.npy"))
        lmk = np.load(os.path.join(sub_dir, f"{i}_lm.npy"))
        lmk = np.array(lmk, np.int32)
        ori_img_list.append(ori_img)
        crop_img_list.append(crop_img)
        box_list.append(box)
        
        crop_img = cv2.resize(crop_img,(args.img_size, args.img_size),interpolation = cv2.INTER_LANCZOS4)
        latent,_  = vae.get_latents_for_unet(crop_img)
        
        if args.enable_pose_driven:
            # lip mask
            lip_mask = create_mouth_mask(depth_img, lmk[mouth_region_indices,:])
            depth_img = depth_img * lip_mask
            depth_img = cv2.resize(depth_img,(args.img_size, args.img_size),interpolation = cv2.INTER_LANCZOS4)
            depth_img[:args.img_size//2,...] = 0
            latent_depth_mask = vae.get_latents_for_nomask(depth_img)
        else:
            latent_depth_mask = latent_depth
        
        latent_list.append(torch.cat([latent, latent_depth_mask], axis=1))
    

    gen = data_generator(whisper_chunks, latent_list, args.batch_size)
    res_frame_list = []
    for i, (whisper_batch,latent_batch) in enumerate(tqdm(gen,total=int(np.ceil(float(ind_max+1)/args.batch_size)))):
        
        tensor_list = [torch.FloatTensor(arr) for arr in whisper_batch]
        audio_feature_batch = torch.stack(tensor_list).to(unet.device) # torch, B, 5*N,384
        audio_feature_batch = pe(audio_feature_batch)
        latent_batch = latent_batch.to(dtype=unet.dtype)

        pred_latents = unet(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch).sample
        recon = vae.decode_latents(pred_latents)
        for res_frame in recon:
            res_frame_list.append(res_frame)
    
    num_frames = min(ind_max+1, len(res_frame_list))
    print(f"original video frames: {ind_max+1}, gerenated video's frames: {len(res_frame_list)}")
    ############################################## pad to full image ##############################################
    print("pad talking image to original video")
    for i in tqdm(range(num_frames)):
    #for i, res_frame in enumerate(tqdm(res_frame_list)):
        box = box_list[i]
        box = [int(e) for e in box]
        res_crop_img = res_frame_list[i]
        ori_img = ori_img_list[i]
        x1, y1, x2, y2 = box
        try:
            res_crop_img = cv2.resize(res_crop_img.astype(np.uint8),(x2-x1,y2-y1))
        except:
#                 print(bbox)
            continue
        
        combine_img = get_image(ori_img, res_crop_img, box)
        cv2.imwrite(f"{result_img_save_path}/{i+1}_edit.png",combine_img)
        
    cmd_img2video = f"ffmpeg -y -v fatal -r {fps} -f image2 -i {result_img_save_path}/%d_edit.png -vcodec libx264 -vf format=rgb24,scale=out_color_matrix=bt709,format=yuv420p -crf 18 temp_{output_basename}.mp4"
    print(cmd_img2video)
    os.system(cmd_img2video)
    
    cmd_combine_audio = f"ffmpeg -y -v fatal -i {audio_path} -i temp_{output_basename}.mp4 {output_vid_name}"
    print(cmd_combine_audio)
    os.system(cmd_combine_audio)
    os.remove(f"temp_{output_basename}.mp4")
    t2 = time.time() 
    shutil.rmtree(result_img_save_path)
    print(f"result is save to {output_vid_name}, time:{t2-t1}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_path", type=str, default="")
    parser.add_argument("--video_path", type=str, default="")
    parser.add_argument("--intermediate_dir", type=str, default="")
    parser.add_argument("--vae_model_path", type=str, default="pretrained_models/sd-vae-ft-mse")
    parser.add_argument("--whisper_model_path", type=str, default="pretrained_models/whisper/tiny.pt")
    parser.add_argument("--unet_model_path", type=str, default="pretrained_models/joygen")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--enable_pose_driven", action='store_true', help="")
    parser.add_argument("--result_dir", default='./results', help="path to output")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=8)

    args = parser.parse_args()
    print(args)
    main(args)
    
