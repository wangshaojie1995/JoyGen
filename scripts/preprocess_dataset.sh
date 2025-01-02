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

python -u preprocess_dataset.py \
    --checkpoints_dir ./pretrained_models \
    --name face_recon_feat0.2_augment \
    --epoch=20 \
    --use_opengl False \
    --bfm_folder ./pretrained_models/BFM \
    --video_dir ./demo \
    --result_dir ./results/preprocessed_dataset \
