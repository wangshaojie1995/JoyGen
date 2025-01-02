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
# This file contains code from Deep3DFaceRecon_pytorch (Copyright (c) 2022 Sicheng Xu),
# licensed under the MIT License, available at https://github.com/sicxu/Deep3DFaceRecon_pytorch.

"""This script contains the test options for Deep3DFaceRecon_pytorch
"""

from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        parser.add_argument('--dataset_mode', type=str, default=None, help='chooses how datasets are loaded. [None | flist]')
        parser.add_argument('--img_folder', type=str, default='examples', help='folder for test images.')
        
        ###########################################  preprocess dataset ######################################################
        parser.add_argument('--video_dir', type=str, default='demo')
        parser.add_argument('--result_dir', type=str, default='results')
        parser.add_argument('--whisper_model_path', type=str, default='pretrained_models/whisper/tiny.pt')
        parser.add_argument('--dwpose_model_path', type=str, default='./pretrained_models/dwpose/dw-ll_ucoco_384.pth')
        parser.add_argument('--dwpose_config_file', type=str, default='./pretrained_models/dwpose/rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py')
        parser.add_argument('--start_idx', type=int, default=0)
        parser.add_argument('--end_idx', type=int, default=1)
        parser.add_argument('--num_workers', type=int, default=1)
        parser.add_argument('--gpu_id', type=int, default=0)
        
        ###########################################  edit expression ######################################################
        parser.add_argument('--infer_video_path', type=str, default='demo/examples01_crop_25fps.mp4')
        parser.add_argument('--infer_exp_coeff_path', type=str, default='results/a2m/xinwen01_crop_exp.npy')
        parser.add_argument('--infer_result_dir', type=str, default='results/edit_exppression')

        # Dropout and Batchnorm has different behavior during training and test.
        self.isTrain = False
        return parser
