python inference_audio2motion.py \
    --a2m_ckpt ./pretrained_models/audio2motion/240210_real3dportrait_orig/audio2secc_vae \
    --hubert_path ./pretrained_models/audio2motion/hubert \
    --drv_aud ./demo/xinwen_34s.mp3 \
    --seed 0 \
    --result_dir ./results/a2m \
    --exp_file xinwen_34s.npy
