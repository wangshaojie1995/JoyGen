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
# This file contains code from Real3DPortrait (Copyright (c) 2024 ZhenhuiYe),
# licensed under the MIT License, available at https://github.com/yerfor/Real3DPortrait.

import subprocess

import numpy as np
from scipy.io import wavfile
import pyloudnorm as pyln


def save_wav(wav, path, sr, norm=False):
    wav = wav.astype(float)
    if norm:
        meter = pyln.Meter(sr)  # create BS.1770 meter
        loudness = meter.integrated_loudness(wav)
        wav = pyln.normalize.loudness(wav, loudness, -18.0)
        if np.abs(wav).max() >= 1:
            wav = wav / np.abs(wav).max() * 0.95
    wav = wav * 32767
    wavfile.write(path[:-4] + '.wav', sr, wav.astype(np.int16))
    if path[-4:] == '.mp3':
        to_mp3(path[:-4])


def to_mp3(out_path):
    if out_path[-4:] == '.wav':
        out_path = out_path[:-4]
    subprocess.check_call(
        f'ffmpeg -threads 1 -loglevel error -i "{out_path}.wav" -vn -b:a 192k -y -hide_banner -async 1 "{out_path}.mp3"',
        shell=True, stdin=subprocess.PIPE)
    subprocess.check_call(f'rm -f "{out_path}.wav"', shell=True)
