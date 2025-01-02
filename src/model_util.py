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

import torch
import torch.nn as nn
from safetensors.torch import load_file
import os


def load_model_weight(model, model_weight_file):
    if os.path.exists(model_weight_file):
        if model_weight_file.endswith("safetensors"):
            state_dict = load_file(model_weight_file)
            model.load_state_dict(state_dict)
        else:
            state_dict = torch.load(model_weight_file)
            new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(new_state_dict)
        print(f"initialize model's weights from {model_weight_file}", main_process_only=True)
    else:
        print(f"not initialize model's weight", main_process_only=True)


def get_mask_tensor(self):
    """
    Creates a mask tensor for image processing.
    :return: A mask tensor.
    """
    mask_tensor = torch.zeros((self.cfg.dataset.img_size,self.cfg.dataset.img_size))
    mask_tensor[:self.cfg.dataset.img_size//2,:] = 1
    mask_tensor[mask_tensor< 0.5] = 0
    mask_tensor[mask_tensor>= 0.5] = 1
    return mask_tensor