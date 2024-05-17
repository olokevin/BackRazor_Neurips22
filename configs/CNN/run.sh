### mcunet test
# CUDA_VISIBLE_DEVICES=0 python CNN/tinytl_fgvc_train.py configs/CNN/test.yaml --path .exp/batch8/ImageNet/mcunet/test --eval_only

### mcunet pretrain
# CUDA_VISIBLE_DEVICES=0 python CNN/tinytl_fgvc_train.py configs/CNN/pretrain.yaml

### mcunet transfer
# CUDA_VISIBLE_DEVICES=0 python CNN/tinytl_fgvc_train.py configs/CNN/transfer.yaml