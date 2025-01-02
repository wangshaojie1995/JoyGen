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

# depth by 3dmm 
# 256 pix
# with depth

 CUDA_VISIBLE_DEIVCES=0 python -u inference_joygen.py \
 --unet_model_path pretrained_models/joygen \
 --vae_model_path pretrained_models/sd-vae-ft-mse \
 --intermediate_dir ./results/edit_expression \
 --audio_path demo/xinwen_5s.mp3 \
 --video_path demo/example_5s.mp4 \
 --enable_pose_driven \
 --result_dir results/talk \
 --img_size 256  \
 --gpu_id 0 \
