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

import numpy as np
import torch

f0_bin = 256
f0_max = 1100.0
f0_min = 50.0
f0_mel_min = 1127 * np.log(1 + f0_min / 700)
f0_mel_max = 1127 * np.log(1 + f0_max / 700)

def coarse_to_f0(coarse):
    uv = coarse == 1
    f0_mel = (coarse - 1) * (f0_mel_max - f0_mel_min) / (f0_bin - 2) + f0_mel_min
    f0 = ((f0_mel / 1127).exp() - 1) * 700
    f0[uv] = 0
    return f0

def f0_to_coarse(f0):
    is_torch = isinstance(f0, torch.Tensor)
    f0_mel = 1127 * (1 + f0 / 700).log() if is_torch else 1127 * np.log(1 + f0 / 700)
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * (f0_bin - 2) / (f0_mel_max - f0_mel_min) + 1

    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > f0_bin - 1] = f0_bin - 1
    f0_coarse = (f0_mel + 0.5).long() if is_torch else np.rint(f0_mel).astype(np.int_)
    assert f0_coarse.max() <= 255 and f0_coarse.min() >= 1, (f0_coarse.max(), f0_coarse.min(), f0.min(), f0.max())
    return f0_coarse


def norm_f0(f0, uv, hparams):
    is_torch = isinstance(f0, torch.Tensor)
    if hparams['pitch_norm'] == 'standard':
        f0 = (f0 - hparams['f0_mean']) / hparams['f0_std']
    if hparams['pitch_norm'] == 'log':
        f0 = torch.log2(f0 + 1e-8) if is_torch else np.log2(f0 + 1e-8)
    if uv is not None and hparams['use_uv']:
        f0[uv > 0] = 0
    return f0