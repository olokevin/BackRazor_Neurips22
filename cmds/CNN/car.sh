gpu=0
lr=3e-4
pruneRatio=0.9
backRazor_pruneRatio=${pruneRatio}
backRazor_pruneRatio_head=${pruneRatio}

# BackRazor
# python CNN/tinytl_fgvc_train.py --transfer_learning_method full \
#     --train_batch_size 8 --test_batch_size 100 \
#     --n_epochs 50 --init_lr ${lr} --opt_type adam \
#     --label_smoothing 0.7 --distort_color torch --frozen_param_bits 8 --origin_network  \
#     --gpu ${gpu} --dataset car --path .exp/batch8/car_backRazor${backRazor_pruneRatio}HeadR${backRazor_pruneRatio_head} \
#     --backRazor --backRazor_pruneRatio ${backRazor_pruneRatio} --backRazor_pruneRatio_head ${backRazor_pruneRatio_head}

# TinyTL
# CUDA_VISIBLE_DEVICES=0 python CNN/tinytl_fgvc_train.py --transfer_learning_method tinytl-bias \
#     --train_batch_size 8 --test_batch_size 100 \
#     --n_epochs 50 --init_lr ${lr} --opt_type adam \
#     --label_smoothing 0.7 --distort_color torch --frozen_param_bits 8 \
#     --gpu ${gpu} --dataset car --path .exp/batch8/car/tinytl 

# MCUNet
CUDA_VISIBLE_DEVICES=0 python CNN/tinytl_fgvc_train.py --transfer_learning_method full \
    --train_batch_size 8 --test_batch_size 100 \
    --n_epochs 50 --init_lr ${lr} --opt_type adam \
    --label_smoothing 0.7 --distort_color torch --frozen_param_bits 8 --origin_network  \
    --gpu ${gpu} --dataset car --path .exp/batch8/car/mcunet \
    --net mcunet-in1 --disable_weight_quantization

# ImageNet validation
# CUDA_VISIBLE_DEVICES=0 python CNN/tinytl_fgvc_train.py --transfer_learning_method full \
#     --train_batch_size 8 --test_batch_size 100 \
#     --n_epochs 50 --init_lr ${lr} --opt_type adam \
#     --label_smoothing 0.7 --distort_color torch --frozen_param_bits 8 --origin_network  \
#     --gpu ${gpu} --dataset ImageNet --path .exp/batch8/car/mcunet \
#     --net mcunet-in1 --disable_weight_quantization --disable_weight_standardization --config configs/CNN/mcunet.yaml