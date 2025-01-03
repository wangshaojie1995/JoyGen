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
#
# This file contains code from MuseTalk (Copyright (c) 2024 Tencent Music Entertainment Group),
# licensed under the MIT License, available at https://github.com/TMElyralab/MuseTalk.

import os
import cv2
import numpy as np
import torch
import yaml

ffmpeg_path = os.getenv('FFMPEG_PATH')
if ffmpeg_path is None:
    print("please download ffmpeg-static and export to FFMPEG_PATH. \nFor example: export FFMPEG_PATH=/musetalk/ffmpeg-4.4-amd64-static")
elif ffmpeg_path not in os.getenv('PATH'):
    print("add ffmpeg to path")
    os.environ["PATH"] = f"{ffmpeg_path}:{os.environ['PATH']}"

    
from musetalk.whisper.audio2feature import Audio2Feature
from musetalk.models.vae import VAE
from musetalk.models.unet import UNet,PositionalEncoding

def load_all_model(device):
    audio_processor = Audio2Feature(model_path="./models/whisper/tiny.pt")
    vae = VAE(model_path = "./models/sd-vae-ft-mse/", device=device)
    unet = UNet(unet_config="./models/musetalk/musetalk.json",
                model_path ="./models/musetalk/pytorch_model.bin", device=device)
    pe = PositionalEncoding(d_model=384)
    return audio_processor,vae,unet,pe

def load_config(config_file):
    with open(config_file, 'r') as ifs:
        config = yaml.safe_load(ifs)
    return config


def get_file_type(video_path):
    _, ext = os.path.splitext(video_path)

    if ext.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
        return 'image'
    elif ext.lower() in ['.avi', '.mp4', '.mov', '.flv', '.mkv']:
        return 'video'
    else:
        return 'unsupported'

def get_video_fps(video_path):
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    video.release()
    return fps

def datagen(whisper_chunks,vae_encode_latents,batch_size=8,delay_frame = 0):
    whisper_batch, latent_batch = [], []
    for i, w in enumerate(whisper_chunks):
        idx = (i+delay_frame)%len(vae_encode_latents)
        latent = vae_encode_latents[idx]
        whisper_batch.append(w)
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


def datagen_beta(whisper_chunks,vae_encode_latents,crop_face_list, batch_size=8,delay_frame = 0):
    whisper_batch, latent_batch, face_batch = [], [], []
    for i, w in enumerate(whisper_chunks):
        idx = (i+delay_frame)%len(vae_encode_latents)
        latent = vae_encode_latents[idx]
        whisper_batch.append(w)
        latent_batch.append(latent)
        face_batch.append(crop_face_list[idx])

        if len(latent_batch) >= batch_size:
            whisper_batch = np.asarray(whisper_batch)
            latent_batch = torch.cat(latent_batch, dim=0)
            face_batch = np.asarray(face_batch)
            yield whisper_batch, latent_batch, face_batch
            whisper_batch, latent_batch, face_batch = [], [], []

    """
    # drop the last batch
    # the last batch may smaller than batch size
    if len(latent_batch) > 0:
        whisper_batch = np.asarray(whisper_batch)
        latent_batch = torch.cat(latent_batch, dim=0)
        face_batch = np.asarray(face_batch)

        yield whisper_batch, latent_batch, face_batch
    """

def datagen_gamma(whisper_chunks, input_latent_list, input_face_list, gt_latent_list, gt_face_list, batch_size=8, delay_frame = 0):
    whisper_batch, input_latent_batch, input_face_batch, gt_latent_batch, gt_face_batch = [], [], [], [], []
    for i, w in enumerate(whisper_chunks):
        idx = (i+delay_frame)%len(input_latent_list)
        whisper_batch.append(w)
        input_latent_batch.append(input_latent_list[idx])
        input_face_batch.append(input_face_list[idx])
        gt_latent_batch.append(gt_latent_list[idx])
        gt_face_batch.append(gt_face_list[idx])

        if len(input_latent_batch) >= batch_size:
            whisper_batch = np.asarray(whisper_batch)
            input_latent_batch = torch.cat(input_latent_batch, dim=0)
            gt_latent_batch = torch.cat(gt_latent_batch, dim=0)
            input_face_batch = np.asarray(input_face_batch)
            gt_face_batch = np.asarray(gt_face_batch)
            #print(whisper_batch.shape, input_latent_batch.shape, input_face_batch.shape, gt_latent_batch.shape, gt_face_batch.shape)
            yield whisper_batch, input_latent_batch, input_face_batch, gt_latent_batch, gt_face_batch
            whisper_batch, input_latent_batch, input_face_batch, gt_latent_batch, gt_face_batch = [], [], [], [], []




