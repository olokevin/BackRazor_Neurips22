### MCUNet test
CUDA_VISIBLE_DEVICES=0 python CNN/eval_torch.py --net_id mcunet-in2 --dataset imagenet --data-dir /home/zyq123/dataset/ImageNet2012/val

### test
CUDA_VISIBLE_DEVICES=0 python CNN/tinytl_fgvc_train.py configs/CNN/test.yaml --eval_only

### pretrain
CUDA_VISIBLE_DEVICES=0 python CNN/tinytl_fgvc_train.py configs/CNN/pretrain.yaml
CUDA_VISIBLE_DEVICES=0 nohup python CNN/tinytl_fgvc_train.py configs/CNN/pretrain.yaml >/dev/null 2>&1 &

### transfer
CUDA_VISIBLE_DEVICES=0 python CNN/tinytl_fgvc_train.py configs/CNN/transfer.yaml
CUDA_VISIBLE_DEVICES=0 nohup python CNN/tinytl_fgvc_train.py configs/CNN/transfer.yaml >/dev/null 2>&1 &