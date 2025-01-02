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

audio_file=$1
video_file=$2
result_dir=$3
audio_name=$(basename "$audio_file")
audio_name=${audio_name%.*}
exp_file="${audio_name}.npy"

python inference_audio2motion.py \
    --a2m_ckpt ./pretrained_models/audio2motion/240210_real3dportrait_orig/audio2secc_vae \
    --hubert_path ./pretrained_models/audio2motion/hubert \
    --drv_aud $audio_file \
    --seed 0 \
    --result_dir "${result_dir}/a2m" \
    --exp_file $exp_file


python -u inference_edit_expression.py \
    --name face_recon_feat0.2_augment \
    --epoch=20 \
    --use_opengl False \
    --checkpoints_dir ./pretrained_models \
    --bfm_folder ./pretrained_models/BFM \
    --infer_video_path $video_file \
    --infer_exp_coeff_path "${result_dir}/a2m/${exp_file}" \
    --infer_result_dir "${result_dir}/edit_exp"


 python -u inference_joygen.py \
 --unet_model_path pretrained_models/joygen \
 --vae_model_path pretrained_models/sd-vae-ft-mse \
 --intermediate_dir "${result_dir}/edit_exp" \
 --audio_path $audio_file \
 --video_path $video_file \
 --enable_pose_driven \
 --result_dir "${result_dir}/talk" \
 --img_size 256  \
 --gpu_id 0 \
