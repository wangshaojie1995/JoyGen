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
import sys
import torch
import random
import time
import numpy as np
import copy
from audio2motion.utils.commons.hparams import hparams, set_hparams
from audio2motion.utils.commons.ckpt_utils import load_ckpt, get_last_checkpoint
from audio2motion.modules.vae import VAEModel, PitchContourVAEModel

from audio2motion.data_gen.utils.process_audio.extract_hubert import get_hubert_from_16k_wav
from audio2motion.utils.audio import librosa_wav2mfcc
from audio2motion.data_gen.utils.process_audio.extract_mel_f0 import extract_mel_from_fname, extract_f0_from_wav_and_mel


class Audio2Motion:
    def __init__(self, audio2secc_dir, device=None, inp=None):
        if torch.cuda.is_available():
            print("CUDA is available!")
        else:
            print("CUDA is not available.")
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.audio2secc_model = self.load_audio2secc(audio2secc_dir)
        self.audio2secc_model.to(device).eval()

    def load_audio2secc(self, audio2secc_dir):
        config_name = f"{audio2secc_dir}/config.yaml" if not audio2secc_dir.endswith(".ckpt") else f"{os.path.dirname(audio2secc_dir)}/config.yaml"
        set_hparams(f"{config_name}", print_hparams=False)
        self.audio2secc_dir = audio2secc_dir
        self.audio2secc_hparams = copy.deepcopy(hparams)
        if self.audio2secc_hparams['audio_type'] == 'hubert':
            audio_in_dim = 1024
        elif self.audio2secc_hparams['audio_type'] == 'mfcc':
            audio_in_dim = 13

        if 'icl' in hparams['task_cls']:
            self.use_icl_audio2motion = True
            model = InContextAudio2MotionModel(hparams['icl_model_type'], hparams=self.audio2secc_hparams)
        else:
            self.use_icl_audio2motion = False
            if hparams.get("use_pitch", False) is True:
                model = PitchContourVAEModel(hparams, in_out_dim=64, audio_in_dim=audio_in_dim)
            else:
                model = VAEModel(in_out_dim=64, audio_in_dim=audio_in_dim)
        load_ckpt(model, f"{audio2secc_dir}", model_name='model', strict=True)
        return model

    def infer_once(self, inp):
        self.inp = inp
        samples = self.prepare_batch_from_inp(inp)
        seed = inp['seed'] if inp['seed'] is not None else int(time.time())
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        result = self.forward_audio2secc(samples, inp)
        return result
    
    def prepare_batch_from_inp(self, inp):
        """
        :param inp: {'audio_source_name': (str)}
        :return: a dict that contains the condition feature of NeRF
        """
        sample = {}
        # Process Driving Motion
        if inp['drv_audio_name'][-4:] in ['.wav', '.mp3']:
            self.save_wav16k(inp['drv_audio_name'])
            if self.audio2secc_hparams['audio_type'] == 'hubert':
                hubert = self.get_hubert(self.wav16k_name, inp['hubert_path'])
            elif self.audio2secc_hparams['audio_type'] == 'mfcc':
                hubert = self.get_mfcc(self.wav16k_name) / 100

            f0 = self.get_f0(self.wav16k_name)
            if f0.shape[0] > len(hubert):
                f0 = f0[:len(hubert)]
            else:
                num_to_pad = len(hubert) - len(f0)
                f0 = np.pad(f0, pad_width=((0,num_to_pad), (0,0)))    
            t_x = hubert.shape[0]
            x_mask = torch.ones([1, t_x]).float() # mask for audio frames
            y_mask = torch.ones([1, t_x//2]).float() # mask for motion/image frames
            sample.update({
                'hubert': torch.from_numpy(hubert).float().unsqueeze(0).cuda(),
                'f0': torch.from_numpy(f0).float().reshape([1,-1]).cuda(),
                'x_mask': x_mask.cuda(),
                'y_mask': y_mask.cuda(),
                })
            sample['blink'] = torch.zeros([1, t_x, 1]).long().cuda()
            sample['audio'] = sample['hubert']
            sample['eye_amp'] = torch.ones([1, 1]).cuda() * 1.0
            sample['mouth_amp'] = torch.ones([1, 1]).cuda() * inp['mouth_amp']

        return sample

    @torch.no_grad()
    def get_hubert(self, wav16k_name, model_path):
        hubert = get_hubert_from_16k_wav(wav16k_name, model_path).detach().numpy()
        len_mel = hubert.shape[0]
        x_multiply = 8
        if len_mel % x_multiply == 0:
            num_to_pad = 0
        else:
            num_to_pad = x_multiply - len_mel % x_multiply
        hubert = np.pad(hubert, pad_width=((0,num_to_pad), (0,0)))
        return hubert

    def get_mfcc(self, wav16k_name):
        hparams['fft_size'] = 1200
        hparams['win_size'] = 1200
        hparams['hop_size'] = 480
        hparams['audio_num_mel_bins'] = 80
        hparams['fmin'] = 80
        hparams['fmax'] = 12000
        hparams['audio_sample_rate'] = 24000
        mfcc = librosa_wav2mfcc(wav16k_name,
            fft_size=hparams['fft_size'],
            hop_size=hparams['hop_size'],
            win_length=hparams['win_size'],
            num_mels=hparams['audio_num_mel_bins'],
            fmin=hparams['fmin'],
            fmax=hparams['fmax'],
            sample_rate=hparams['audio_sample_rate'],
            center=True)
        mfcc = np.array(mfcc).reshape([-1, 13])
        len_mel = mfcc.shape[0]
        x_multiply = 8
        if len_mel % x_multiply == 0:
            num_to_pad = 0
        else:
            num_to_pad = x_multiply - len_mel % x_multiply
        mfcc = np.pad(mfcc, pad_width=((0,num_to_pad), (0,0)))
        return mfcc

    @torch.no_grad()
    def forward_audio2secc(self, batch, inp=None):
        if inp['drv_audio_name'][-4:] in ['.wav', '.mp3']:
            # audio-to-exp
            ret = {}
            pred = self.audio2secc_model.forward(batch, ret=ret,train=False, temperature=inp['temperature'],)
            print("| audio-to-motion finished")
            if pred.shape[-1] == 144:
                exp = ret['pred'][0][:,80:]
            else:
                exp = ret['pred'][0]
            batch['exp'] = exp
        else:
            raise "format not be supported"
        return batch
    
    @classmethod
    def example_run(cls, inp):
        infer_instance = cls(inp['a2m_ckpt'], inp=inp)
        result = infer_instance.infer_once(inp)
        return result

    ##############
    # IO-related
    ##############
    def save_wav16k(self, audio_name):
        supported_types = ('.wav', '.mp3', '.mp4', '.avi')
        assert audio_name.endswith(supported_types), f"Now we only support {','.join(supported_types)} as audio source!"
        wav16k_name = audio_name[:-4] + '_16k.wav'
        self.wav16k_name = wav16k_name
        extract_wav_cmd = f"ffmpeg -i {audio_name} -f wav -ar 16000 -v quiet -y {wav16k_name} -y"
        os.system(extract_wav_cmd)
        print(f"Extracted wav file (16khz) from {audio_name} to {wav16k_name}.")

    def get_f0(self, wav16k_name):
        wav, mel = extract_mel_from_fname(self.wav16k_name)
        f0, f0_coarse = extract_f0_from_wav_and_mel(wav, mel)
        f0 = f0.reshape([-1,1])
        return f0


if __name__ == '__main__':
    import argparse, glob, tqdm
    parser = argparse.ArgumentParser()
    parser.add_argument("--a2m_ckpt", default='./pretrained_models/audio2motion/240210_real3dportrait_orig/audio2secc_vae', type=str) 
    parser.add_argument("--hubert_path", default='pretrained_models/hubert', type=str) 
    parser.add_argument("--drv_aud", default='demo/xinwen02_crop.mp3', type=str) # data/raw/examples/Obama_5s.wav
    parser.add_argument("--blink_mode", default='period', type=str) # none | period
    parser.add_argument("--temperature", default=0.2, type=float) # sampling temperature in audio2motion, higher -> more diverse, less accurate
    parser.add_argument("--mouth_amp", default=0.45, type=float) # scale of predicted mouth, enabled in audio-driven
    parser.add_argument("--seed", default=None, type=int) # random seed, default None to use time.time()
    parser.add_argument("--result_dir", default='./results/a2m', type=str)
    parser.add_argument("--exp_file", default='./exp.npy', type=str)

    args = parser.parse_args()

    inp = {
            'a2m_ckpt': args.a2m_ckpt,
            'hubert_path': args.hubert_path,
            'drv_audio_name': args.drv_aud,
            'blink_mode': args.blink_mode,
            'temperature': args.temperature,
            'mouth_amp': args.mouth_amp,
            'seed': args.seed,
            }
    result = Audio2Motion.example_run(inp)
    exp = result['exp'].detach().cpu().numpy()
    print(exp)   
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    np.save(os.path.join(args.result_dir, args.exp_file), exp)
