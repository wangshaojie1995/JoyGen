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

import os, glob
from audio2motion.utils.commons.os_utils import multiprocess_glob
from audio2motion.utils.commons.multiprocess_utils import multiprocess_run_tqdm


def extract_wav16k_job(audio_name:str):
    out_path = audio_name.replace("/audio_raw/","/audio/",1)
    assert out_path != audio_name # prevent inplace
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ffmpeg_path = "/usr/bin/ffmpeg"

    cmd = f'{ffmpeg_path} -i {audio_name} -ar 16000 -v quiet -y {out_path}'
    os.system(cmd)

if __name__ == '__main__':
    import argparse, glob, tqdm, random
    parser = argparse.ArgumentParser()
    parser.add_argument("--aud_dir", default='/home/tiger/datasets/raw/CMLR/audio_raw/')
    parser.add_argument("--ds_name", default='CMLR')
    parser.add_argument("--num_workers", default=64, type=int)
    parser.add_argument("--process_id", default=0, type=int)
    parser.add_argument("--total_process", default=1, type=int)
    args = parser.parse_args()
    print(f"args {args}")

    aud_dir = args.aud_dir
    ds_name = args.ds_name
    if ds_name in ['CMLR']:
        aud_name_pattern = os.path.join(aud_dir, "*/*/*.wav")
        aud_names = multiprocess_glob(aud_name_pattern)
    else:
        raise NotImplementedError()
    aud_names = sorted(aud_names)
    print(f"total audio number : {len(aud_names)}")
    print(f"first {aud_names[0]} last {aud_names[-1]}")
    # exit()
    process_id = args.process_id
    total_process = args.total_process
    if total_process > 1:
        assert process_id <= total_process -1
        num_samples_per_process = len(aud_names) // total_process
        if process_id == total_process:
            aud_names = aud_names[process_id * num_samples_per_process : ]
        else:
            aud_names = aud_names[process_id * num_samples_per_process : (process_id+1) * num_samples_per_process]
    
    for i, res in multiprocess_run_tqdm(extract_wav16k_job, aud_names, num_workers=args.num_workers, desc="resampling videos"):
        pass

