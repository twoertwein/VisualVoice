python test.py \
--audio1_path /vision/vision_rhgao/VisualVoice/VoxCeleb2/audio/seen_heard_test/id06688/akPwstwDxjE/00023.wav \
--audio2_path /vision/vision_rhgao/VisualVoice/VoxCeleb2/audio/seen_heard_test/id08606/0o-ZBLLLjXE/00002.wav \
--mouthroi1_path /vision/vision_rhgao/VisualVoice/VoxCeleb2/mouth_roi/seen_heard_test/id06688/akPwstwDxjE/00023.h5 \
--mouthroi2_path /vision/vision_rhgao/VisualVoice/VoxCeleb2/mouth_roi/seen_heard_test/id08606/0o-ZBLLLjXE/00002.h5 \
--video1_path /vision/vision_rhgao/VisualVoice/VoxCeleb2/mp4/seen_heard_test/id06688/akPwstwDxjE/00023.mp4 \
--video2_path /vision/vision_rhgao/VisualVoice/VoxCeleb2/mp4/seen_heard_test/id08606/0o-ZBLLLjXE/00002.mp4 \
--offscreen_audio_path /vision/vision_rhgao/VisualVoice/audioset/nonspeech_eval/ZZKV1pR4Ptg_30000_40000.wav \
--num_frames 64 \
--audio_length 2.55 \
--hop_size 160 \
--window_size 400 \
--n_fft 512 \
--weights_lipreadingnet pretrained_models/lipreading_best.pth \
--weights_facial pretrained_models/facial_best.pth \
--weights_unet pretrained_models/unet_best.pth \
--weights_vocal pretrained_models/vocal_best.pth \
--lipreading_config_path configs/lrw_snv1x_tcn2x.json \
--unet_output_nc 2 \
--normalization \
--mask_to_use pred \
--visual_feature_type both \
--identity_feature_dim 128 \
--audioVisual_feature_dim 1152 \
--visual_pool maxpool \
--audio_pool maxpool \
--compression_type none \
--mask_clip_threshold 5 \
--hop_length 2.55 \
--audio_normalization \
--lipreading_extract_feature \
--number_of_identity_frames 1 \
--output_dir_root test ; python evaluateSeparation.py --results_dir test/id06688_akPwstwDxjE_00023VSid08606_0o-ZBLLLjXE_00002