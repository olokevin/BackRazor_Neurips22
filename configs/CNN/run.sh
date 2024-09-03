# ### MCUNet test
# CUDA_VISIBLE_DEVICES=0 python CNN/eval_torch.py --net_id mcunet-in2 --dataset imagenet --data-dir /home/zyq123/dataset/ImageNet2012/val

# ### test
# CUDA_VISIBLE_DEVICES=0 python CNN/tinytl_fgvc_train.py configs/CNN/test.yaml --eval_only

# ### pretrain
# CUDA_VISIBLE_DEVICES=0 python CNN/tinytl_fgvc_train.py configs/CNN/pretrain.yaml
# CUDA_VISIBLE_DEVICES=0 nohup python CNN/tinytl_fgvc_train.py configs/CNN/pretrain.yaml >/dev/null 2>&1 &

# ### transfer
# CUDA_VISIBLE_DEVICES=0 python CNN/tinytl_fgvc_train.py configs/CNN/transfer.yaml
# CUDA_VISIBLE_DEVICES=0 nohup python CNN/tinytl_fgvc_train.py configs/CNN/transfer.yaml >/dev/null 2>&1 &

### cross-val
# for i in {1..13}
# do
#   gpu=$((i % 4))
#   # CUDA_VISIBLE_DEVICES=0 python CNN/tinytl_fgvc_train.py configs/CNN/transfer.yaml --trainable_blocks [1]
#   CUDA_VISIBLE_DEVICES=$gpu nohup python CNN/tinytl_fgvc_train.py configs/CNN/transfer.yaml --trainable_blocks [$i] --n_epochs 50 >/dev/null 2>&1 &
# done


# corruption_types=(gaussian_noise impulse_noise shot_noise fog frost snow elastic_transform brightness contrast defocus_blur)
# corruption_types=(gaussian_noise impulse_noise shot_noise fog frost)
# corruption_types=(snow elastic_transform brightness contrast defocus_blur)

# corruption_types=(gaussian_noise shot_noise impulse_noise speckle_noise gaussian_blur defocus_blur glass_blur motion_blur zoom_blur) 
# corruption_types=(snow frost fog brightness contrast elastic_transform pixelate jpeg_compression saturate spatter)
corruption_types=(gaussian_noise shot_noise impulse_noise speckle_noise gaussian_blur defocus_blur glass_blur motion_blur zoom_blur snow frost fog brightness contrast elastic_transform pixelate jpeg_compression saturate spatter) 

# Initialize a counter for the GPU index
gpu_index=0
# gpu=0

# Iterate through the list of corruption types
for corruption in "${corruption_types[@]}"
do
  # Set the GPU to be used for the current experiment
  gpu=$((gpu_index % 4))
  
  # Run the experiment with the specified GPU and corruption type
  # CUDA_VISIBLE_DEVICES=$gpu python CNN/tinytl_fgvc_train.py configs/CNN/transfer.yaml --corruption_type $corruption
  CUDA_VISIBLE_DEVICES=$gpu nohup python CNN/tinytl_fgvc_train.py configs/CNN/transfer.yaml --corruption_type $corruption >/dev/null 2>&1 &
  
  # Increment the GPU index for the next experiment
  gpu_index=$((gpu_index + 1))
done