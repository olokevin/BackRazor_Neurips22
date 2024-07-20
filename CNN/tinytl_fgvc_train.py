import argparse
import os
import inspect
import sys
sys.path.append(".")
import numpy as np
import json
import random
import time
import torch
from torch import nn

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, os.path.join(parent_dir, 'once-for-all'))

from ofa.utils.layers import LinearLayer
from ofa.model_zoo import proxylessnas_mobile
# from ofa.imagenet_classification.run_manager import RunManager
from CNN.run_manager import RunManager
from ofa.utils import init_models, download_url, list_mean

### Use my own replace_conv2d_with_my_conv2d
# from ofa.utils import replace_conv2d_with_my_conv2d, replace_bn_with_gn
from ofa.utils import replace_bn_with_gn
from CNN.utils import replace_conv2d_with_my_conv2d

from CNN.data_providers import FGVCRunConfig
from CNN.utils import set_module_grad_status, enable_bn_update, disable_bn_update, enable_bias_update, weight_quantization
from CNN.utils import profile_memory_cost
from CNN.model import LiteResidualModule, build_network_from_config

# from custom_functions.custom_conv import SparseConv2d
# from custom_functions.masker import Masker
# from custom_functions.custom_fc import LinearSparse

from torchvision import models

from pdb import set_trace

### my add
from CNN.mcunet.model_zoo import net_id_list, build_model, download_tflite
from CNN.utils import fuse_bn, fuse_bn_add_gn, replace_bn_with_insn
from tools.config import configs, load_config_from_file, update_config_from_args, update_config_from_unknown_args

from torchvision import datasets, transforms

from ZO_Estim.ZO_Estim_entry import build_ZO_Estim

parser = argparse.ArgumentParser()
parser.add_argument('config', metavar='FILE', help='config file')
parser.add_argument('--path', type=str, metavar='DIR', help='run directory')
parser.add_argument('--eval_only', action='store_true')
parser.add_argument('--debug', action='store_true')

# parser = argparse.ArgumentParser()
# parser.add_argument('--path', type=str, default=None)
# parser.add_argument('--gpu', help='gpu available', default='0')
# parser.add_argument('--resume', action='store_true')
# parser.add_argument('--manual_seed', default=0, type=int)

# """ RunConfig: dataset related """
# parser.add_argument('--dataset', type=str, default='flowers102', choices=[
#     'aircraft', 'car', 'flowers102',
#     'food101', 'cub200', 'pets',
#     'cifar10', 'cifar100', 'ImageNet',
# ])
# parser.add_argument('--train_batch_size', type=int, default=8)
# parser.add_argument('--test_batch_size', type=int, default=100)
# parser.add_argument('--valid_size', type=float, default=None)

# parser.add_argument('--n_worker', type=int, default=10)
# parser.add_argument('--resize_scale', type=float, default=0.22)
# parser.add_argument('--distort_color', type=str, default='tf', choices=['tf', 'torch', 'None'])
# parser.add_argument('--image_size', type=int, default=224)

# """ RunConfig: optimization related """
# parser.add_argument('--n_epochs', type=int, default=50)
# parser.add_argument('--init_lr', type=float, default=0.05)
# parser.add_argument('--lr_schedule_type', type=str, default='cosine')

# parser.add_argument('--opt_type', type=str, default='adam', choices=['sgd', 'adam'])
# parser.add_argument('--momentum', type=float, default=0.9)  # opt_param
# parser.add_argument('--no_nesterov', action='store_true')  # opt_param
# parser.add_argument('--weight_decay', type=float, default=0)
# parser.add_argument('--no_decay_keys', type=str, default='bn#bias', choices=['None', 'bn', 'bn#bias', 'bias'])
# parser.add_argument('--label_smoothing', type=float, default=0)

# """ net config """
# parser.add_argument('--net', type=str, default='proxyless_mobile')
# parser.add_argument('--dropout', type=float, default=0.2)
# parser.add_argument('--ws_eps', type=float, default=1e-5)
# parser.add_argument('--net_path', type=str, default=None)
# parser.add_argument('--fix_bn_stat', action="store_true", help="if stop bn from summery the statistics")

# """ transfer learning configs """
# parser.add_argument('--transfer_learning_method', type=str, default='tinytl-lite_residual+bias', choices=[
#   'full', 'full_noBN', 'bn+last', 'last',
#   'tinytl-bias', 'tinytl-lite_residual', 'tinytl-lite_residual+bias'
# ])
# parser.add_argument('--origin_network', action="store_true")
# parser.add_argument('--backRazor', action="store_true")
# parser.add_argument('--backRazor_pruneRatio', type=float, default=0.8)
# parser.add_argument('--backRazor_pruneRatio_head', type=float, default=-1)
# parser.add_argument('--backRazor_act_prune', action="store_true", help="also prune the forward activation for ablation")

# """ lite residual module configs """
# parser.add_argument('--lite_residual_downsample', type=int, default=2)
# parser.add_argument('--lite_residual_expand', type=int, default=1)
# parser.add_argument('--lite_residual_groups', type=int, default=2)
# parser.add_argument('--lite_residual_ks', type=int, default=5)
# parser.add_argument('--random_init_lite_residual', action='store_true')

# """ weight quantization """
# parser.add_argument('--disable_weight_quantization', action="store_true")
# parser.add_argument('--frozen_param_bits', type=int, default=8)

# """ weight standardization """
# parser.add_argument('--disable_weight_standardization', action="store_true")

# def replace_conv2d_with_back_razor_conv2d(net, masker, act_prune):
#   for m in net.modules():
#     to_update_dict = {}
#     for name, sub_module in m.named_children():
#       if isinstance(sub_module, nn.Conv2d):
#         to_update_dict[name] = sub_module

#     for name, sub_module in to_update_dict.items():
#       m._modules[name] = SparseConv2d(
#         sub_module.in_channels,
#         sub_module.out_channels,
#         sub_module.kernel_size,
#         sub_module.stride,
#         sub_module.padding,
#         sub_module.dilation,
#         sub_module.groups,
#         sub_module.bias,
#         masker=masker,
#         act_prune=act_prune
#       )
#       # load weight
#       m._modules[name].load_state_dict(sub_module.state_dict(), strict=False)
#       # load requires_grad
#       m._modules[name].weight.requires_grad = sub_module.weight.requires_grad
#       if sub_module.bias is not None:
#         m._modules[name].bias.requires_grad = sub_module.bias.requires_grad

if __name__ == '__main__':
  # args = parser.parse_args()
  arguments, unknown = parser.parse_known_args()
  load_config_from_file(arguments.config)
  update_config_from_args(arguments)
  update_config_from_unknown_args(unknown)
  args = configs
  args.path = arguments.path
  args.eval_only = arguments.eval_only
  args.debug = arguments.debug  

  if args.path is None:
    args.path = os.path.join(
      "./exp",
      'batch'+str(args.train_batch_size),
      args.dataset,
      args.net, 
      args.transfer_learning_method,
      time.strftime("%Y%m%d-%H%M%S")+'-'+str(os.getpid())
    )
  
  os.makedirs(args.path, exist_ok=True)
  json.dump(args.__dict__, open(os.path.join(args.path, 'args.txt'), 'w'), indent=4)
  print(args)

  # setup transfer learning
  args.enable_feature_extractor_update = False
  args.enable_bn_update = False
  args.enable_bias_update = False
  args.enable_lite_residual = False
  args.disable_bn_update = False
  args.disable_head_update = False
  if args.transfer_learning_method == 'full':
    args.enable_feature_extractor_update = True
  elif args.transfer_learning_method == 'full_noBN':
    args.enable_feature_extractor_update = True
    args.disable_bn_update = True
  elif args.transfer_learning_method == 'bn+last':
    args.enable_bn_update = True
  elif args.transfer_learning_method == 'last':
    pass
  elif args.transfer_learning_method == 'bn_only':
    args.enable_bn_update = True
    args.disable_head_update = True
  elif args.transfer_learning_method == 'customize':
    args.disable_bn_update = True
    args.disable_head_update = True
  elif args.transfer_learning_method == 'tinytl-bias':
    args.enable_bias_update = True
  elif args.transfer_learning_method == 'tinytl-lite_residual':
    args.enable_lite_residual = True
  elif args.transfer_learning_method == 'tinytl-lite_residual+bias':
    args.enable_bias_update = True
    args.enable_lite_residual = True
  else:
    raise ValueError('Do not support %s' % args.transfer_learning_method)

  # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
  if args.resume:
    args.manual_seed = int(time.time())  # set new manual seed
  torch.manual_seed(args.manual_seed)
  torch.cuda.manual_seed_all(args.manual_seed)
  np.random.seed(args.manual_seed)
  random.seed(args.manual_seed)

  # run config
  if isinstance(args.valid_size, float) and args.valid_size > 1:
    args.valid_size = int(args.valid_size)
  args.no_decay_keys = None if args.no_decay_keys == 'None' else args.no_decay_keys
  args.opt_param = {'momentum': args.momentum, 'nesterov': not args.no_nesterov}

  run_config = FGVCRunConfig(**args.__dict__)
  print('Run config:')
  for k, v in run_config.config.items():
    print('\t%s: %s' % (k, v))

  # network
  classification_head = []
  if args.net == 'proxyless_mobile':
    if args.origin_network:
      net = proxylessnas_mobile(pretrained=True)
    else:
      net = proxylessnas_mobile(pretrained=False)
      LiteResidualModule.insert_lite_residual(
        net, args.lite_residual_downsample, 'bilinear', args.lite_residual_expand, args.lite_residual_ks,
        'relu', args.lite_residual_groups,
      )
      # replace bn layers with gn layers
      replace_bn_with_gn(net, gn_channel_per_group=8)
      # load pretrained model
      init_file = download_url('https://hanlab18.mit.edu/projects/tinyml/tinyTL/files/'
                   'proxylessnas_mobile+lite_residual@imagenet@ws+gn', model_dir='./assets/')
      net.load_state_dict(torch.load(init_file, map_location='cpu')['state_dict'])

    if args.backRazor:
      assert args.origin_network
      # if args.backRazor_pruneRatio_head < -0.5:
      #   head_prune_ratio = 0
      # else:
      #   head_prune_ratio = args.backRazor_pruneRatio_head
      # print("backRazor head prune ratio is {}".format(head_prune_ratio))
      # masker = Masker(prune_ratio=head_prune_ratio)
      # net.classifier = nn.Sequential(nn.Dropout(args.dropout),
      #                  LinearSparse(net.classifier.in_features, run_config.data_provider.n_classes,
      #                       masker=masker, act_prune=args.backRazor_act_prune))
      # classification_head.append(net.classifier)
      # init_models(classification_head)
    else:
      if args.eval_only or args.init_new_head==False:
        pass
      else:
        net.classifier = LinearLayer(net.classifier.in_features, run_config.data_provider.n_classes, dropout_rate=args.dropout)
        classification_head.append(net.classifier)
        init_models(classification_head)
  elif "resnet" in args.net:
    net = models.__dict__[args.net](pretrained=True)
    if args.backRazor:
      assert args.origin_network
      # if args.backRazor_pruneRatio_head < -0.5:
      #   head_prune_ratio = 0
      # else:
      #   head_prune_ratio = args.backRazor_pruneRatio_head
      # print("backRazor head prune ratio is {}".format(head_prune_ratio))
      # masker = Masker(prune_ratio=head_prune_ratio)
      # net.fc = nn.Sequential(nn.Dropout(args.dropout),
      #              LinearSparse(net.fc.in_features, run_config.data_provider.n_classes,
      #                   masker=masker, act_prune=args.backRazor_act_prune))
    else:
      net.fc = LinearLayer(net.fc.in_features, run_config.data_provider.n_classes, dropout_rate=args.dropout)
    classification_head.append(net.fc)
    init_models(classification_head)
  elif 'mcunet' in args.net:
    net, image_size, description = build_model(net_id=args.net, pretrained=args.pretrained)

    # replace bn layers with gn layers
    if args.origin_network == False:
      replace_bn_with_gn(net, gn_channel_per_group=8)
    else:
      if args.replace_norm_layer == 'fuse_bn':
        fuse_bn(net)
      elif args.replace_norm_layer == 'fuse_bn_add_gn':
        fuse_bn_add_gn(net, gn_channel_per_group=args.gn_channel_per_group)
      elif args.replace_norm_layer == 'replace_bn_with_insn':
        replace_bn_with_insn(net)
      else:
        pass
    
    # load pretrained model
    if args.pretrain_model_path is not None:
      net.load_state_dict(torch.load(args.pretrain_model_path, map_location='cpu')['state_dict'])
    
    if args.eval_only or args.init_new_head==False:
      run_config.data_provider.assign_active_img_size(image_size)
    else:
      net.classifier = LinearLayer(net.classifier.in_features, run_config.data_provider.n_classes, dropout_rate=args.dropout)
      classification_head.append(net.classifier)
      init_models(classification_head)
  else:
    assert False
    if args.net_path is not None:
      net_config_path = os.path.join(args.net_path, 'net.config')
      init_path = os.path.join(args.net_path, 'init')
    else:
      base_url = 'https://hanlab18.mit.edu/projects/tinyml/tinyTL/files/specialized/%s/' % args.dataset
      net_config_path = download_url(base_url + 'net.config',
                                     model_dir='~/.tinytl/specialized/%s' % args.dataset)
      init_path = download_url(base_url + 'init', model_dir='~/.tinytl/specialized/%s' % args.dataset)
    net_config = json.load(open(net_config_path, 'r'))
    net = build_network_from_config(net_config)

    net.classifier = LinearLayer(net.classifier.in_features, run_config.data_provider.n_classes, dropout_rate=args.dropout)
    classification_head.append(net.classifier)

    # load init (weight quantization already applied)
    init = torch.load(init_path, map_location='cpu')
    if 'state_dict' in init:
      init = init['state_dict']
    net.load_state_dict(init)

  # set transfer learning configs
  set_module_grad_status(net, args.enable_feature_extractor_update)
  set_module_grad_status(classification_head, True)

  if args.enable_bn_update:
    enable_bn_update(net)
  if args.disable_bn_update:
    assert not args.enable_bn_update
    disable_bn_update(net)
  if args.disable_head_update:
    set_module_grad_status(classification_head, False)
  if args.enable_bias_update:
    enable_bias_update(net)
  if args.enable_lite_residual:
    for m in net.modules():
      if isinstance(m, LiteResidualModule):
        set_module_grad_status(m.lite_residual, True)
        if args.enable_bias_update or args.enable_bn_update:
          m.lite_residual.final_bn.bias.requires_grad = False
        if args.random_init_lite_residual:
          init_models(m.lite_residual)
          m.lite_residual.final_bn.weight.data.zero_()
  
  ##### Test weight size and output size #####
  # for name, layer in net.named_modules():
  #   if isinstance(layer, nn.Conv2d):
  #     print(f'{layer.weight.data.numel()}')

  # print('output size')
  # def print_output_size(self, input, output):
  #   print(f"{output.data.numel() / output.data.shape[0]}")

  # def register_hooks(module):
  #   for name, layer in module.named_children():
  #       if isinstance(layer, nn.Conv2d):
  #         layer.register_forward_hook(print_output_size)
  #       register_hooks(layer)
  
  # register_hooks(net)
  # images, targets = next(iter(run_config.train_loader))
  # output = net(images)
  ##### Test weight size and output size #####
  
  if args.train_first_conv == True:
    net.first_conv.conv.weight.requires_grad = True
    if hasattr(net.first_conv, 'bn'):
      net.first_conv.bn.weight.requires_grad = True
      net.first_conv.bn.bias.requires_grad = True
    
    if args.net == 'proxyless_mobile':
      net.blocks[0].conv.depth_conv.conv.weight.requires_grad = True
      net.blocks[0].conv.point_linear.conv.weight.requires_grad = True
    elif 'mcunet' in args.net:
      net.blocks[0].mobile_inverted_conv.depth_conv.conv.weight.requires_grad = True
      net.blocks[0].mobile_inverted_conv.point_linear.conv.weight.requires_grad = True
  
  if args.trainable_blocks is not None and args.trainable_layers is not None:
    if args.net == 'proxyless_mobile':
      for layer_num in args.trainable_blocks:
          if args.trainable_layers == 'first':
            net.blocks[layer_num].conv.main_branch.inverted_bottleneck.conv.weight.requires_grad = True
          elif args.trainable_layers == 'no_dw':
            net.blocks[layer_num].conv.main_branch.inverted_bottleneck.conv.weight.requires_grad = True
            net.blocks[layer_num].conv.main_branch.depth_conv.conv.weight.requires_grad = False
            net.blocks[layer_num].conv.main_branch.point_linear.conv.weight.requires_grad = True
          elif args.trainable_layers == 'all':
            net.blocks[layer_num].conv.main_branch.inverted_bottleneck.conv.weight.requires_grad = True
            net.blocks[layer_num].conv.main_branch.depth_conv.conv.weight.requires_grad = True
            net.blocks[layer_num].conv.main_branch.point_linear.conv.weight.requires_grad = True
    elif 'mcunet' in args.net:
      for layer_num in args.trainable_blocks:
        if args.trainable_layers == 'first':
          net.blocks[layer_num].mobile_inverted_conv.inverted_bottleneck.conv.weight.requires_grad = True
        elif args.trainable_layers == 'no_dw':
          net.blocks[layer_num].mobile_inverted_conv.inverted_bottleneck.conv.weight.requires_grad = True
          net.blocks[layer_num].mobile_inverted_conv.depth_conv.conv.weight.requires_grad = False
          net.blocks[layer_num].mobile_inverted_conv.point_linear.conv.weight.requires_grad = True
        elif args.trainable_layers == 'all':
          net.blocks[layer_num].mobile_inverted_conv.inverted_bottleneck.conv.weight.requires_grad = True
          net.blocks[layer_num].mobile_inverted_conv.depth_conv.conv.weight.requires_grad = True
          net.blocks[layer_num].mobile_inverted_conv.point_linear.conv.weight.requires_grad = True
    else:
      raise NotImplementedError(f'not supported {args.net} for sparse training')

  # weight quantization on frozen parameters
  if not args.resume and args.weight_quantization:
    weight_quantization(net, bits=args.frozen_param_bits, max_iter=20)

  # setup weight standardization
  if args.backRazor:
    assert args.origin_network
    # mask = Masker(prune_ratio=args.backRazor_pruneRatio)
    # replace_conv2d_with_back_razor_conv2d(net, mask, act_prune=args.backRazor_act_prune)
  else:
    if args.weight_standardization:
      replace_conv2d_with_my_conv2d(net, args.ws_eps)
  
  # ZO estimator
  if args.ZO_Estim.en:
    ZO_Estim = build_ZO_Estim(args.ZO_Estim, model=net, )
  else:
    ZO_Estim = None

  # build run manager
  run_manager = RunManager(args.path, net, run_config, ZO_Estim=ZO_Estim, init=False)
  run_manager.write_log(str(os.getpid()))

  # profile memory cost
  require_backward = args.enable_feature_extractor_update or args.enable_bn_update or args.enable_bias_update \
                     or args.enable_lite_residual
  input_size = (1, 3, run_config.data_provider.active_img_size, run_config.data_provider.active_img_size)

  # memory_cost, detailed_info = profile_memory_cost(
  #   net, input_size, require_backward, activation_bits=32, trainable_param_bits=32,
  #   frozen_param_bits=args.frozen_param_bits, batch_size=run_config.train_batch_size,
  # )
  # net_info = {
  #   'memory_cost': memory_cost / 1e6,
  #   'param_size': detailed_info['param_size'] / 1e6,
  #   'act_size': detailed_info['act_size'] / 1e6,
  # }
  # print(net_info)
  # with open('%s/net_info.txt' % run_manager.path, 'a') as fout:
  #   fout.write(json.dumps(net_info, indent=4) + '\n')

  # information of parameters that will be updated via gradient
  run_manager.write_log('Updated params:', 'grad_params', False, 'w')
  for i, param_group in enumerate(run_manager.optimizer.param_groups):
    run_manager.write_log(
      'Group %d: %d params with wd %f' % (i + 1, len(param_group['params']), param_group['weight_decay']),
      'grad_params', True, 'a')
  # for name, param in net.named_parameters():
  #   if param.requires_grad:
  #     run_manager.write_log('%s: %s' % (name, list(param.data.size())), 'grad_params', False, 'a')

  run_manager.save_config()
  if args.resume:
    run_manager.load_model()
  else:
    init_path = '%s/init' % args.path
    if os.path.isfile(init_path):
      checkpoint = torch.load(init_path, map_location='cpu')
      if 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
      run_manager.network.load_state_dict(checkpoint)

  # train
  if args.eval_only:
    pass
  else:
    args.teacher_model = None
    run_manager.train(args)
  
  # test
  img_size, loss, acc1, acc5 = run_manager.validate_all_resolution(is_test=True)
  log = 'test_loss: %f\t test_acc1: %f\t test_acc5: %f\t' % (list_mean(loss), list_mean(acc1), list_mean(acc5))
  for i_s, v_a in zip(img_size, acc1):
    log += '(%d, %.3f), ' % (i_s, v_a)
  run_manager.write_log(log, prefix='test')
