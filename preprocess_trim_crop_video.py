import os
import torch
from torch import nn
from copy import deepcopy
import numpy as np
from utils.download import load_file_from_url
from utils.download import download_pretrained_models
from utils.retinaface.retinaface import RetinaFace
import cv2
import time
import pickle
import math
import random
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_retinaface_model(model_name, half=False):
    if model_name == 'retinaface_resnet50':
        model = RetinaFace(network_name='resnet50', half=half)
        model_url = 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/detection_Resnet50_Final.pth'
    elif model_name == 'retinaface_mobile0.25':
        model = RetinaFace(network_name='mobile0.25', half=half)
        model_url = 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/detection_mobilenet0.25_Final.pth'
    else:
        raise NotImplementedError(f'{model_name} is not implemented.')

    model_path = load_file_from_url(url=model_url, model_dir='pretrained_models', progress=True, file_name=None)
    load_net = torch.load(model_path, map_location=lambda storage, loc: storage)
    # remove unnecessary 'module.'
    for k, v in deepcopy(load_net).items():
        if k.startswith('module.'):
            load_net[k[7:]] = v
            load_net.pop(k)
    model.load_state_dict(load_net, strict=True)
    model.eval()
    model = model.to(device)

    return model


def read_image(img):
    """img can be image path or cv2 loaded image."""
    # self.input_img is Numpy array, (h, w, c), BGR, uint8, [0, 255]
    if isinstance(img, str):
        img = cv2.imread(img)

    if np.max(img) > 256:  # 16-bit image
        img = img / 65535 * 255
    if len(img.shape) == 2:  # gray image
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:  # BGRA image with alpha channel
        img = img[:, :, 0:3]

    return img

def make_detector():
    det_model = "retinaface_mobile0.25"
    face_detector = init_retinaface_model(det_model, half=False)
    return face_detector

def get_faces(face_detector, img):
    resize = 300
    input_img = read_image(img) 
    if resize is None:
        scale = 1
    else:
        h, w = input_img.shape[0:2]
        scale = resize / min(h, w)
        scale = min(1, scale) # always downsample to speed up
        h, w = int(h * scale), int(w * scale)
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        input_img = cv2.resize(input_img, (w, h), interpolation=interp)

    with torch.no_grad():
        bboxes = face_detector.detect_faces(input_img)
    
    rects = []
    for bbox in bboxes:
        x1, y1, x2, y2 = int(bbox[0]/scale), int(bbox[1]/scale), int(bbox[2]/scale), int(bbox[3]/scale)
        rects.append([x1,y1,x2,y2])
    return rects


def extract_rects(video_path, save_dir): 
    face_detector = make_detector()

    cap = cv2.VideoCapture(video_path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    video_name = os.path.basename(video_path).split(".")[0]
    rect_file = os.path.join(save_dir, f"{video_name}.pkl")

    video_rects = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rects = get_faces(face_detector, frame)
        video_rects.append(rects)
    cap.release()
    
    with open(rect_file, "wb") as pickle_file:
            pickle.dump(video_rects, pickle_file)

    return video_rects, rect_file

    
def get_startend(rect_file):
    with open(rect_file, "rb") as pickle_file:
            video_rects = pickle.load(pickle_file)
    print(f"rect file of video: {rect_file}")
    video_rects = [[]] + video_rects + [[]]
    start = []
    end = []
    for ind in range(len(video_rects)-1):
        rect1 = video_rects[ind]
        rect2 = video_rects[ind+1]
        if (len(rect1) == 0 or len(rect1) > 1) and len(rect2) == 1:
            start.append(ind+1)
        if len(rect1) == 1 and (len(rect2) == 0 or len(rect2) > 1):
            end.append(ind)
    if len(start) != len(end):
        print('error')
        return []
    intervals = []
    for s,e in zip(start, end):
        intervals.append((s,e))    

    return intervals


def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, height, width, num_frames


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


def get_bbox_from_bboxes(rects, height, width, ratio=0.5):
    x1s = []
    y1s = []
    x2s = []
    y2s = []
    ws, hs = [], []
    for rect in rects:
        #print(rect)
        x1, y1, x2, y2 = rect[0]
        #x1, y1, x2, y2 = rect[0], rect[1], rect[2], rect[3]
        x1s.append(x1)
        y1s.append(y1)
        x2s.append(x2)
        y2s.append(y2)
        ws.append(x2-x1)
        hs.append(y2-y1)

    x1_ = np.min(x1s)
    y1_ = np.min(y1s)
    x2_ = np.max(x2s)
    y2_ = np.max(y2s)

    w = np.mean(ws) 
    h = np.mean(hs)

    x1_ = x1_ - 0.5 * ratio * w
    x2_ = x2_ + 0.5 * ratio * w

    y1_ = y1_ - 0.5 * ratio * h
    y2_ = y2_ + 0.5 * ratio * h

    return bbox_check([x1_,y1_,x2_,y2_], height, width)



def clip_video(video_path, save_dir):
    t1 = time.time()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    fps, height, width, num_frames = get_video_info(video_path) 
    print(f"{video_path}, fps:{fps}, height:{height}, width:{width}, frames:{num_frames}")
    print(f"{video_path}, start to detect faces in video")
    video_rects, rect_file = extract_rects(video_path, save_dir)
    print(f"{video_path}, finish to detect faces in video")
    
    intervals = get_startend(rect_file)
    video_name = os.path.basename(video_path).split(".")[0]
    format_name = os.path.basename(video_path).split(".")[1]

    print(f"{video_path}, start to seg&crop video")
    bias = 10 # drop at least 0.x sec
    num_thresh = 300 # about 10 sec video
    for k,interval in enumerate(intervals):
        start_ind = interval[0]
        end_ind = interval[1]
        if end_ind - start_ind < num_thresh:
            continue

        start_ind = start_ind + bias
        end_ind = end_ind - bias 

        start_time = math.ceil(float(start_ind)/fps)
        end_time = math.floor(float(end_ind)/fps)
        during_time = end_time - start_time 


        rects_clip = video_rects[start_ind:end_ind]
        crop_rect = get_bbox_from_bboxes(rects_clip, height, width, 0.5)
        x1,y1,x2,y2 = crop_rect
        h,w = y2-y1, x2-x1 
        
        cmd = f"ffmpeg -y -loglevel quiet -i \"{video_path}\" -ss {start_time} -t {during_time} -r 25 -acodec aac -vcodec h264 -strict -2 -vf \"crop={w}:{h}:{x1}:{y1}\" \"{save_dir}\"/{k}.{format_name}"
        print(f"{video_path}, {k}-th clip, {cmd}")
        os.system(cmd)
    t2 = time.time()
    print(f"{video_path}, finish to crop video, time:{t2-t1}")
    
def find_videos(directory, extensions=['.mp4', '.avi', '.mov']):
    videos = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if os.path.splitext(file)[1].lower() in extensions:
                videos.append(os.path.join(root, file))
    return videos


def bat(src_dir, dst_dir, start_idx, end_idx):
    video_paths = find_videos(src_dir)
    video_paths = sorted(video_paths)
    print(video_paths)
    num = len(video_paths)
    for i,video_path in enumerate(video_paths[start_idx:end_idx]):
        print(f"{i}/{num}, {video_path}")
        video_name = os.path.basename(video_path).split(".")[0]
        save_dir = os.path.join(dst_dir, video_name)
        clip_video(video_path, save_dir)



if __name__ == "__main__":
    src_dir = sys.argv[1]
    dst_dir = sys.argv[2]
    bat(src_dir, dst_dir, start_idx, end_idx)
