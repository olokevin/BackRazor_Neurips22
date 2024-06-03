### mcunet test
CUDA_VISIBLE_DEVICES=0 python CNN/tinytl_fgvc_train.py configs/CNN/test.yaml --eval_only

### mcunet pretrain
CUDA_VISIBLE_DEVICES=0 python CNN/tinytl_fgvc_train.py configs/CNN/pretrain.yaml
CUDA_VISIBLE_DEVICES=0 nohup python CNN/tinytl_fgvc_train.py configs/CNN/pretrain.yaml >/dev/null 2>&1 &

### mcunet transfer
CUDA_VISIBLE_DEVICES=0 python CNN/tinytl_fgvc_train.py configs/CNN/transfer.yaml
CUDA_VISIBLE_DEVICES=0 nohup python CNN/tinytl_fgvc_train.py configs/CNN/transfer.yaml >/dev/null 2>&1 &