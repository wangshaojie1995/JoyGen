[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) ![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)
# JoyGen
JoyGen: Audio-Driven 3D Depth-Aware Talking-Face Video Editing

## ShowCase
### Driving Audio(Chinese)
<table class="center">
<tr>
    <td width=25% style="border: none">
        <video controls loop src="https://github.com/user-attachments/assets/bad0b728-c0f1-46d8-b608-49ebb6d438a8" muted="false"></video>
    </td>
    <td width=25% style="border: none">
        <video controls loop src="https://github.com/user-attachments/assets/a1697fd7-3f1a-42ba-ac89-79ff73a339f0" muted="false"></video>
    </td>
    <td width=25% style="border: none">
        <video controls loop src="https://github.com/user-attachments/assets/c34e7d4f-ffad-42df-a458-e873cba6a5dc" muted="false"></video>
    </td>
</tr>
</table>

### Driving Audio(English)
<table class="center">
<tr>
    <td width=25% style="border: none">
        <video controls loop src="https://github.com/user-attachments/assets/da497c52-82de-4df6-a0e0-b2289f860776" muted="false"></video>
    </td>
    <td width=25% style="border: none">
        <video controls loop src="https://github.com/user-attachments/assets/f320832f-31cd-443e-9721-ee78955e64ba" muted="false"></video>
    </td>
    <td width=25% style="border: none">
        <video controls loop src="https://github.com/user-attachments/assets/4e9e3089-2b14-4ced-9a96-210ac6c56072" muted="false"></video>
    </td>
</tr>
</table>


## Installation
### Python Environment
- Tested GPUS: V100, A800
- Tested Python Version: 3.8.19

Create conda environment and install packages with pip:
```
conda create -n joygen python=3.8.19 ffmpeg
conda activate joygen
pip install -r requirements.txt
```

Install Nvdiffrast library:
```
git clone https://github.com/NVlabs/nvdiffrast
cd nvdiffrast
pip install .
```

### Download pretrained weights
These pretrained models should be organized as follows:
```text
./pretrained_models/
├── BFM
│   ├── 01_MorphableModel.mat
│   ├── BFM_exp_idx.mat
│   ├── BFM_front_idx.mat
│   ├── BFM_model_front.mat
│   ├── Exp_Pca.bin
│   ├── facemodel_info.mat
│   ├── index_mp468_from_mesh35709.npy
│   ├── select_vertex_id.mat
│   ├── similarity_Lm3D_all.mat
│   └── std_exp.txt
├── audio2motion
│   ├── 240210_real3dportrait_orig
│   │   └── audio2secc_vae
│   │       ├── config.yaml
│   │       └── model_ckpt_steps_400000.ckpt
│   └── hubert
│       ├── config.json
│       ├── preprocessor_config.json
│       ├── pytorch_model.bin
│       ├── special_tokens_map.json
│       ├── tokenizer_config.json
│       └── vocab.json
├── joygen
│   ├── config.json
│   └── diffusion_pytorch_model.safetensors
├── dwpose
│   ├── default_runtime.py
│   ├── dw-ll_ucoco_384.pth
│   └── rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py
├── face-parse-bisent
│   ├── 79999_iter.pth
│   └── resnet18-5c106cde.pth
├── face_recon_feat0.2_augment
│   ├── epoch_20.pth
│   ├── loss_log.txt
│   ├── test_opt.txt
│   └── train_opt.txt
├── sd-vae-ft-mse
│   ├── README.md
│   ├── config.json
│   ├── diffusion_pytorch_model.bin
│   └── diffusion_pytorch_model.safetensors
└── whisper
    └── tiny.pt
```
Or you can download them separately:
- [audio2motion](https://github.com/yerfor/Real3DPortrait)
- [hubert](https://huggingface.co/facebook/hubert-large-ls960-ft/tree/main)
- [BFM](https://github.com/sicxu/Deep3DFaceRecon_pytorch?tab=readme-ov-file#prepare-prerequisite-models)
- [joygen]()
- [dwpose](https://github.com/IDEA-Research/DWPose)
- [face_recon_feat0.2_augment](https://github.com/sicxu/Deep3DFaceRecon_pytorch?tab=readme-ov-file#prepare-prerequisite-models)
- [face-parse-bisent](https://github.com/zllrunning/face-parsing.PyTorch)
- [sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse)
- [whisper](https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt)


## Inference
Run the inference script:
```
bash scripts/inference_pipeline.sh args1 args2 args3
```
- args1: driving audio file
- args2: video file
- args3: result directory

Run the inference script step by step
1. Obtain a sequence of facial expression coefficients from the audio.
```
python inference_audio2motion.py \
    --a2m_ckpt ./pretrained_models/audio2motion/240210_real3dportrait_orig/audio2secc_vae \
    --hubert_path ./pretrained_models/audio2motion/hubert \
    --drv_aud ./demo/xinwen_5s.mp3 \
    --seed 0 \
    --result_dir ./results/a2m \
    --exp_file xinwen_5s.npy
```
2. Render the depth map frame by frame using the new expression coefficients.
```
python -u inference_edit_expression.py \
    --name face_recon_feat0.2_augment \
    --epoch=20 \
    --use_opengl False \
    --checkpoints_dir ./pretrained_models \
    --bfm_folder ./pretrained_models/BFM \
    --infer_video_path ./demo/example_5s.mp4 \
    --infer_exp_coeff_path ./results/a2m/xinwen_5s.npy \
    --infer_result_dir ./results/edit_expression
```
3. Generate the facial animation based on the audio features and the facial depth map.
```
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
```


## Training

### Preprocess Training Data

```
python -u preprocess_dataset.py \
    --checkpoints_dir ./pretrained_models \
    --name face_recon_feat0.2_augment \
    --epoch=20 \
    --use_opengl False \
    --bfm_folder ./pretrained_models/BFM \
    --video_dir ./demo \  # The directory for storing video files.
    --result_dir ./results/preprocessed_dataset \
```

Check the preprocessed data and generate a list file for training.
```
python -u preprocess_dataset_extra.py  data_dir
```


### Training

Modify the config.yaml file according to the specific requirements, such as the dataset section.

```
accelerate launch --main_process_port 29501 --config_file config/accelerate_config.yaml train_joygen.py
```


## Acknowledgement

We would like to thank the contributors to the [Deep3DFaceRecon_pytorch](https://github.com/sicxu/Deep3DFaceRecon_pytorch), [Real3DPortrait](https://github.com/yerfor/Real3DPortrait), [MuseTalk](https://github.com/TMElyralab/MuseTalk) for their open research and exploration.

## Citation
```
@misc{xx,
      title={JoyGen: Audio-Driven 3D Depth-Aware Talking-Face Video Editing}, 
      author={xx,xx},
      year={2024},
      eprint={24},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```