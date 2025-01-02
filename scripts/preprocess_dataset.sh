python -u preprocess_dataset.py \
    --checkpoints_dir ./pretrained_models \
    --name face_recon_feat0.2_augment \
    --epoch=20 \
    --use_opengl False \
    --bfm_folder ./pretrained_models/BFM \
    --video_dir ./demo \
    --result_dir ./results/preprocessed_dataset \
