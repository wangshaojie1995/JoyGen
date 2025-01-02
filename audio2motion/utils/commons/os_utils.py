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

import os
import subprocess
import glob
from audio2motion.utils.commons.multiprocess_utils import multiprocess_run_tqdm


def link_file(from_file, to_file):
    subprocess.check_call(
        f'ln -s "`realpath --relative-to="{os.path.dirname(to_file)}" "{from_file}"`" "{to_file}"', shell=True)


def move_file(from_file, to_file):
    subprocess.check_call(f'mv "{from_file}" "{to_file}"', shell=True)


def copy_file(from_file, to_file):
    subprocess.check_call(f'cp -r "{from_file}" "{to_file}"', shell=True)


def remove_file(*fns):
    for f in fns:
        subprocess.check_call(f'rm -rf "{f}"', shell=True)

def glob_job(d, f):
    pattern = os.path.join(d, f)
    return glob.glob(pattern)

def multiprocess_glob(pattern, num_workers=None):
    split_pattern = pattern.split("/")
    recursive_depth = 0 # number of recursive depth
    for split in split_pattern:
        if '*' in split:
            recursive_depth += 1
    if recursive_depth == 1:
        return glob.glob(pattern)
    else:
        dirs = glob.glob('/'.join(split_pattern[:-1]))
        ret = []
        args = [(d, split_pattern[-1]) for d in dirs]
        for (i,res) in multiprocess_run_tqdm(glob_job, args=args, desc=f"globing {pattern}", num_workers=num_workers):
            ret += res
        return ret
