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

python -u preprocess_musetalk_train_single_process.py --name face_recon_feat0.2_augment --epoch=20 --use_opengl False --start_idx 2 --end_idx 3 --num_workers 3 --gpu_id 0
