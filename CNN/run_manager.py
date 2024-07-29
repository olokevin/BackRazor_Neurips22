# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import os
import random
import time
import json
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from tqdm import tqdm

from ofa.utils import (
    get_net_info,
    cross_entropy_loss_with_soft_target,
    cross_entropy_with_label_smoothing,
)
from ofa.utils import (
    AverageMeter,
    accuracy,
    write_log,
    mix_images,
    mix_labels,
    init_models,
)
from ofa.utils import MyRandomResizedCrop

from ZO_Estim.ZO_Estim_entry import build_obj_fn
import wandb

__all__ = ["RunManager"]

def fwd_hook_save_value(module, input, output):
    module.in_value = input[0].detach().clone()
    module.out_value = output.detach().clone()

def bwd_hook_save_grad(module, grad_input, grad_output):
    module.in_grad = grad_input[0].detach().clone()
    module.out_grad = grad_output[0].detach().clone()


class RunManager:
    def __init__(
        self, path, net, run_config, ZO_Estim=None, init=True, measure_latency=None, no_gpu=False
    ):
        self.path = path
        self.net = net
        self.run_config = run_config
        self.ZO_Estim = ZO_Estim

        self.best_acc = 0
        self.start_epoch = 0

        os.makedirs(self.path, exist_ok=True)

        # move network to GPU if available
        if torch.cuda.is_available() and (not no_gpu):
            self.device = torch.device("cuda:0")
            self.net = self.net.to(self.device)
            cudnn.benchmark = True
        else:
            self.device = torch.device("cpu")
        # initialize model (default)
        if init:
            init_models(run_config.model_init)

        # net info
        net_info = get_net_info(
            self.net, self.run_config.data_provider.data_shape, measure_latency, print_info=False
        )
        with open("%s/net_info.txt" % self.path, "w") as fout:
            fout.write(json.dumps(net_info, indent=4) + "\n")
            # noinspection PyBroadException
            try:
                fout.write(self.network.module_str + "\n")
            except Exception:
                pass
            fout.write("%s\n" % self.run_config.data_provider.train.dataset.transform)
            fout.write("%s\n" % self.run_config.data_provider.test.dataset.transform)
            fout.write("%s\n" % self.network)

        # criterion
        if isinstance(self.run_config.mixup_alpha, float):
            self.train_criterion = cross_entropy_loss_with_soft_target
        elif self.run_config.label_smoothing > 0:
            self.train_criterion = (
                lambda pred, target: cross_entropy_with_label_smoothing(
                    pred, target, self.run_config.label_smoothing
                )
            )
        else:
            self.train_criterion = nn.CrossEntropyLoss()
        self.test_criterion = nn.CrossEntropyLoss()

        # optimizer
        if self.run_config.no_decay_keys:
            keys = self.run_config.no_decay_keys.split("#")
            net_params = [
                self.network.get_parameters(
                    keys, mode="exclude"
                ),  # parameters with weight decay
                self.network.get_parameters(
                    keys, mode="include"
                ),  # parameters without weight decay
            ]
        else:
            # noinspection PyBroadException
            try:
                net_params = self.network.weight_parameters()
            except Exception:
                net_params = []
                for param in self.network.parameters():
                    if param.requires_grad:
                        net_params.append(param)
        self.optimizer = self.run_config.build_optimizer(net_params)

        self.net = torch.nn.DataParallel(self.net)

    """ save path and log path """

    @property
    def save_path(self):
        if self.__dict__.get("_save_path", None) is None:
            save_path = os.path.join(self.path, "checkpoint")
            os.makedirs(save_path, exist_ok=True)
            self.__dict__["_save_path"] = save_path
        return self.__dict__["_save_path"]

    @property
    def logs_path(self):
        if self.__dict__.get("_logs_path", None) is None:
            logs_path = os.path.join(self.path, "logs")
            os.makedirs(logs_path, exist_ok=True)
            self.__dict__["_logs_path"] = logs_path
        return self.__dict__["_logs_path"]

    @property
    def network(self):
        return self.net.module if isinstance(self.net, nn.DataParallel) else self.net

    def write_log(self, log_str, prefix="valid", should_print=True, mode="a"):
        write_log(self.logs_path, log_str, prefix, should_print, mode)

    """ save and load models """

    def save_model(self, checkpoint=None, is_best=False, model_name=None):
        if checkpoint is None:
            checkpoint = {"state_dict": self.network.state_dict()}

        if model_name is None:
            model_name = "checkpoint.pth.tar"

        checkpoint[
            "dataset"
        ] = self.run_config.dataset  # add `dataset` info to the checkpoint
        latest_fname = os.path.join(self.save_path, "latest.txt")
        model_path = os.path.join(self.save_path, model_name)
        with open(latest_fname, "w") as fout:
            fout.write(model_path + "\n")
        torch.save(checkpoint, model_path)

        if is_best:
            best_path = os.path.join(self.save_path, "model_best.pth.tar")
            torch.save({"state_dict": checkpoint["state_dict"]}, best_path)

    def load_model(self, model_fname=None):
        latest_fname = os.path.join(self.save_path, "latest.txt")
        if model_fname is None and os.path.exists(latest_fname):
            with open(latest_fname, "r") as fin:
                model_fname = fin.readline()
                if model_fname[-1] == "\n":
                    model_fname = model_fname[:-1]
        # noinspection PyBroadException
        try:
            if model_fname is None or not os.path.exists(model_fname):
                model_fname = "%s/checkpoint.pth.tar" % self.save_path
                with open(latest_fname, "w") as fout:
                    fout.write(model_fname + "\n")
            print("=> loading checkpoint '{}'".format(model_fname))
            checkpoint = torch.load(model_fname, map_location="cpu")
        except Exception:
            print("fail to load checkpoint from %s" % self.save_path)
            return {}

        self.network.load_state_dict(checkpoint["state_dict"])
        if "epoch" in checkpoint:
            self.start_epoch = checkpoint["epoch"] + 1
        if "best_acc" in checkpoint:
            self.best_acc = checkpoint["best_acc"]
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

        print("=> loaded checkpoint '{}'".format(model_fname))
        return checkpoint

    def save_config(self, extra_run_config=None, extra_net_config=None):
        """dump run_config and net_config to the model_folder"""
        run_save_path = os.path.join(self.path, "run.config")
        if not os.path.isfile(run_save_path):
            run_config = self.run_config.config
            if extra_run_config is not None:
                run_config.update(extra_run_config)
            json.dump(run_config, open(run_save_path, "w"), indent=4)
            print("Run configs dump to %s" % run_save_path)

        try:
            net_save_path = os.path.join(self.path, "net.config")
            net_config = self.network.config
            if extra_net_config is not None:
                net_config.update(extra_net_config)
            json.dump(net_config, open(net_save_path, "w"), indent=4)
            print("Network configs dump to %s" % net_save_path)
        except Exception:
            print("%s do not support net config" % type(self.network))

    """ metric related """

    def get_metric_dict(self):
        return {
            "top1": AverageMeter(),
            "top5": AverageMeter(),
        }

    def update_metric(self, metric_dict, output, labels):
        acc1, acc5 = accuracy(output, labels, topk=(1, 5))
        metric_dict["top1"].update(acc1[0].item(), output.size(0))
        metric_dict["top5"].update(acc5[0].item(), output.size(0))

    def get_metric_vals(self, metric_dict, return_dict=False):
        if return_dict:
            return {key: metric_dict[key].avg for key in metric_dict}
        else:
            return [metric_dict[key].avg for key in metric_dict]

    def get_metric_names(self):
        return "top1", "top5"

    """ train and test """

    def validate(
        self,
        epoch=0,
        is_test=False,
        run_str="",
        net=None,
        data_loader=None,
        no_logs=False,
        train_mode=False,
    ):
        if net is None:
            net = self.net
        if not isinstance(net, nn.DataParallel):
            net = nn.DataParallel(net)

        if data_loader is None:
            data_loader = (
                self.run_config.test_loader if is_test else self.run_config.valid_loader
            )

        if train_mode:
            net.train()
        else:
            net.eval()

        losses = AverageMeter()
        metric_dict = self.get_metric_dict()

        with torch.no_grad():
            with tqdm(
                total=len(data_loader),
                desc="Validate Epoch #{} {}".format(epoch + 1, run_str),
                disable=no_logs,
            ) as t:
                for i, (images, labels) in enumerate(data_loader):
                    images, labels = images.to(self.device), labels.to(self.device)
                    # compute output
                    output = net(images)
                    loss = self.test_criterion(output, labels)
                    # measure accuracy and record loss
                    self.update_metric(metric_dict, output, labels)

                    losses.update(loss.item(), images.size(0))
                    t.set_postfix(
                        {
                            "loss": losses.avg,
                            **self.get_metric_vals(metric_dict, return_dict=True),
                            "img_size": images.size(2),
                        }
                    )
                    t.update(1)
        return losses.avg, self.get_metric_vals(metric_dict)

    def validate_all_resolution(self, epoch=0, is_test=False, net=None):
        if net is None:
            net = self.network
        if isinstance(self.run_config.data_provider.image_size, list):
            img_size_list, loss_list, top1_list, top5_list = [], [], [], []
            for img_size in self.run_config.data_provider.image_size:
                img_size_list.append(img_size)
                self.run_config.data_provider.assign_active_img_size(img_size)
                self.reset_running_statistics(net=net)
                loss, (top1, top5) = self.validate(epoch, is_test, net=net)
                loss_list.append(loss)
                top1_list.append(top1)
                top5_list.append(top5)
            return img_size_list, loss_list, top1_list, top5_list
        else:
            loss, (top1, top5) = self.validate(epoch, is_test, net=net)
            return (
                [self.run_config.data_provider.active_img_size],
                [loss],
                [top1],
                [top5],
            )

    def train_one_epoch(self, args, epoch, warmup_epochs=0, warmup_lr=0):
        # switch to train mode
        self.net.train()

        if args.fix_bn_stat:
            for name, module in self.net.named_modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()

        MyRandomResizedCrop.EPOCH = epoch  # required by elastic resolution

        nBatch = len(self.run_config.train_loader)

        losses = AverageMeter()
        metric_dict = self.get_metric_dict()
        data_time = AverageMeter()

        with tqdm(
            total=nBatch,
            desc="{} Train Epoch #{}".format(self.run_config.dataset, epoch + 1),
        ) as t:
            end = time.time()
            for i, (images, labels) in enumerate(self.run_config.train_loader):
                MyRandomResizedCrop.BATCH = i
                data_time.update(time.time() - end)
                if epoch < warmup_epochs:
                    new_lr = self.run_config.warmup_adjust_learning_rate(
                        self.optimizer,
                        warmup_epochs * nBatch,
                        nBatch,
                        epoch,
                        i,
                        warmup_lr,
                    )
                else:
                    new_lr = self.run_config.adjust_learning_rate(
                        self.optimizer, epoch - warmup_epochs, i, nBatch
                    )

                images, labels = images.to(self.device), labels.to(self.device)
                target = labels
                if isinstance(self.run_config.mixup_alpha, float):
                    # transform data
                    lam = random.betavariate(
                        self.run_config.mixup_alpha, self.run_config.mixup_alpha
                    )
                    images = mix_images(images, lam)
                    labels = mix_labels(
                        labels,
                        lam,
                        self.run_config.data_provider.n_classes,
                        self.run_config.label_smoothing,
                    )

                # soft target
                if args.teacher_model is not None:
                    args.teacher_model.train()
                    with torch.no_grad():
                        soft_logits = args.teacher_model(images).detach()
                        soft_label = F.softmax(soft_logits, dim=1)

                if args.teacher_model is None:
                    loss_type = "ce"
                else:
                    if args.kd_type == "ce":
                        kd_loss = cross_entropy_loss_with_soft_target(
                            output, soft_label
                        )
                    else:
                        kd_loss = F.mse_loss(output, soft_logits)
                    loss = args.kd_ratio * kd_loss + loss
                    loss_type = "%.1fkd+ce" % args.kd_ratio

                if args.debug:
                    from ofa.utils.layers import MBConvLayer, ZeroLayer
                    from CNN.model.modules import LiteResidualModule
                    layer_name_list = ['inverted_bottleneck', 'depth_conv', 'point_linear']

                    ##### Add hook to save input/output value & grad #####
                    hook_handle_list = []
                    hook_handle_list.append(self.network.first_conv.register_forward_hook(fwd_hook_save_value))
                    # hook_handle_list.append(self.network.first_conv.register_full_backward_hook(bwd_hook_save_grad))
                    for block in self.network.blocks:
                        if args.net == 'proxyless_mobile':
                            if isinstance(block.conv, MBConvLayer):
                                layer=block.conv
                            elif isinstance(block.conv, LiteResidualModule):
                                layer=block.conv.main_branch
                            elif isinstance(block.conv, ZeroLayer):
                                continue
                        elif 'mcunet' in args.net:
                            layer = block.mobile_inverted_conv
                        
                        for layer_name in layer_name_list:
                            conv_layer = getattr(layer, layer_name)
                            if conv_layer is None:
                                continue
                            else:
                                hook_handle_list.append(conv_layer.register_forward_hook(fwd_hook_save_value))
                                hook_handle_list.append(conv_layer.register_full_backward_hook(bwd_hook_save_grad))
                
                if self.ZO_Estim is None:
                    # compute output
                    output = self.net(images)
                    loss = self.train_criterion(output, labels)
                    # compute gradient
                    # self.net.zero_grad()  # or self.optimizer.zero_grad()
                    loss.backward()

                    """
                         grad_output sparsity
                    """
                    # if self.run_config.grad_output_prune_ratio is not None:
                    #     grad_output_prune_ratio = self.run_config.grad_output_prune_ratio
                    #     for layer_num in self.run_config.trainable_blocks:
                            
                    #         dw_channelwise = self.network.blocks[layer_num].mobile_inverted_conv.depth_conv.conv.weight.abs().sum([1,2,3])
                    #         topk_dim = int((1.0-grad_output_prune_ratio) * dw_channelwise.numel())
                    #         _, indices = torch.topk(dw_channelwise, topk_dim)

                    #         pw1_grad_w = self.network.blocks[layer_num].mobile_inverted_conv.inverted_bottleneck.conv.weight.grad

                    #         pruned_pw1_grad_w = torch.zeros_like(pw1_grad_w)
                    #         for index in indices:
                    #             pruned_pw1_grad_w[index] = pw1_grad_w.data[index]
                            
                    #         mask = torch.zeros_like(pw1_grad_w, dtype=torch.bool)
                            
                    #         self.network.blocks[layer_num].mobile_inverted_conv.inverted_bottleneck.conv.weight.grad.data = pruned_pw1_grad_w
                    #         # print(layer_num)

                    if args.debug:
                        
                        """
                            Output Norm
                        """
                        # pw1_in_value = self.network.blocks[block_idx].mobile_inverted_conv.inverted_bottleneck.in_value
                        # pw1_out_value = self.network.blocks[block_idx].mobile_inverted_conv.inverted_bottleneck.out_value

                        # log_dict = {
                        #     f"first_conv/norm": torch.norm(self.network.first_conv.conv.weight),
                        #     f"first_conv/max": torch.max(self.network.first_conv.conv.weight.abs()),
                        # }

                        # for block_idx, block in enumerate(self.network.blocks[0:2]):
                        #     if args.net == 'proxyless_mobile':
                        #         if isinstance(block.conv, MBConvLayer):
                        #             layer=block.conv
                        #         elif isinstance(block.conv, LiteResidualModule):
                        #             layer=block.conv.main_branch
                        #         elif isinstance(block.conv, ZeroLayer):
                        #             continue
                        #     elif 'mcunet' in args.net:
                        #         layer = block.mobile_inverted_conv
                            
                        #     for layer_name in layer_name_list:
                        #         conv_layer = getattr(layer, layer_name)
                        #         if conv_layer is None:
                        #             continue
                        #         else:
                        #             log_dict.update({
                        #                 f"block_{block_idx}/{layer_name}/w_norm": torch.norm(conv_layer.conv.weight),
                        #                 f"block_{block_idx}/{layer_name}/w_max": torch.max(conv_layer.conv.weight.abs()),
                        #                 f"block_{block_idx}/{layer_name}/out_norm": torch.norm(conv_layer.out_value),
                        #                 f"block_{block_idx}/{layer_name}/out_max": torch.max(conv_layer.out_value.abs()),
                        #             })

                        # wandb.log(log_dict)
                        """
                            Residual gradient similarity
                        """
                        # last_block_out_grad = None
                        # cos_sim_log = ''
                        # block_idx_list = [-2, -3, -5, -6]
                        # for block_idx in block_idx_list:
                        #     block_out_grad = self.network.blocks[block_idx].mobile_inverted_conv.point_linear.out_grad
                        #     block_in_grad = self.network.blocks[block_idx-1].mobile_inverted_conv.point_linear.out_grad
                            
                        #     cos_sim = F.cosine_similarity(block_out_grad.view(-1), block_in_grad.view(-1), dim=0)
                        #     cos_sim_log += f'block {block_idx}: {cos_sim}'
                        
                        # self.write_log(cos_sim_log, prefix="cos_sim", should_print=False)

                        ##### save input/output value & grad #####
                        """
                            save input/output value & grad (Explore activation sparsity)
                        """
                        # block_idx = -2

                        # pw1_w_grad = self.network.blocks[block_idx].mobile_inverted_conv.inverted_bottleneck.conv.weight.grad.data
                        # pw1_in_grad = self.network.blocks[block_idx].mobile_inverted_conv.inverted_bottleneck.in_grad
                        # pw1_out_grad = self.network.blocks[block_idx].mobile_inverted_conv.inverted_bottleneck.out_grad
                      
                        # pw1_in_value = self.network.blocks[block_idx].mobile_inverted_conv.inverted_bottleneck.in_value
                        # pw1_out_value = self.network.blocks[block_idx].mobile_inverted_conv.inverted_bottleneck.out_value

                        # dw_w = self.network.blocks[block_idx].mobile_inverted_conv.depth_conv.conv.weight.data
                        # pw2_w = self.network.blocks[block_idx].mobile_inverted_conv.point_linear.conv.weight.data
                        # dw_out_grad = self.network.blocks[block_idx].mobile_inverted_conv.depth_conv.out_grad
                        # pw2_out_grad = self.network.blocks[block_idx].mobile_inverted_conv.point_linear.out_grad

                        # torch.save((pw1_w_grad, pw1_in_grad, pw1_out_grad, pw1_in_value, pw1_out_value, dw_w, dw_out_grad, pw2_w, pw2_out_grad), f'./temp/debug_data_{block_idx}.pt')

                else:
                    ##### Test #####
                    if args.debug:
                        """
                            Monte Carlo test
                        """
                        # output = self.net(images)
                        # FO_loss = self.train_criterion(output, labels)
                        # FO_loss.backward()  

                        # ##### Save FO gradient
                        # try:
                        #     block_idx = args.ZO_Estim.param_perturb_block_idx_list[-1]
                        # except:
                        #     try:
                        #         block_idx = args.ZO_Estim.actv_perturb_block_idx_list[-1]
                        #     except:
                        #         block_idx = -1
                        
                        # from ZO_Estim.ZO_utils import SplitedLayer
                        # splited_layer = SplitedLayer(idx=block_idx, name=f'blocks.{block_idx}.adapter_mlp', layer=self.network.blocks[block_idx].mobile_inverted_conv.inverted_bottleneck.conv)

                        # # FO_grad = splited_layer.layer.out_grad[0].data
                        # FO_adapter_up_grad_w = splited_layer.layer.weight.grad.data
                        pass

                    ##### ZO gradient Estimation #####
                    obj_fn_type = self.ZO_Estim.obj_fn_type
                    kwargs = {}
                    if obj_fn_type == 'classifier_layerwise':
                        kwargs = {'get_iterable_block_name': self.ZO_Estim.get_iterable_block_name, "pre_block_forward": self.ZO_Estim.pre_block_forward, "post_block_forward": self.ZO_Estim.post_block_forward}
                    with torch.no_grad():
                        output = self.net(images)
                        loss = self.train_criterion(output, labels)
                        obj_fn = build_obj_fn(obj_fn_type, data=images, target=labels, model=self.network, criterion=self.train_criterion, **kwargs)
                        self.ZO_Estim.update_obj_fn(obj_fn)
                        output, loss = self.ZO_Estim.estimate_grad()
                    
                    if args.debug:
                        """
                            Monte Carlo test
                        """
                        # # ZO_grad = splited_layer.layer.out_grad[0].data
                        # ZO_adapter_up_grad_w = splited_layer.layer.weight.grad.data

                        # # print('\n Grad output')
                        # # print('cos sim grad_output', F.cosine_similarity(FO_grad.view(-1), ZO_grad.view(-1), dim=0))
                        # # print('FO_grad:', torch.linalg.norm(FO_grad))
                        # # print('ZO_grad:', torch.linalg.norm(ZO_grad))

                        # print(f'weight cos sim {F.cosine_similarity(FO_adapter_up_grad_w.view(-1), ZO_adapter_up_grad_w.view(-1), dim=0)}')
                        # print(f'FO_weight_grad norm: {torch.linalg.norm(FO_adapter_up_grad_w)}')
                        # print(f'ZO_weight_grad norm: {torch.linalg.norm(ZO_adapter_up_grad_w)}')
                        # print(f'ZO/FO:  {torch.linalg.norm(ZO_adapter_up_grad_w)/torch.linalg.norm(FO_adapter_up_grad_w)}')
                        pass

                if args.debug:
                    """
                        Gradient Norm
                    """
                    log_dict = {}
                    for block_idx, block in enumerate(self.network.blocks):
                        if args.net == 'proxyless_mobile':
                            if isinstance(block.conv, MBConvLayer):
                                layer=block.conv
                            elif isinstance(block.conv, LiteResidualModule):
                                layer=block.conv.main_branch
                            elif isinstance(block.conv, ZeroLayer):
                                continue
                        elif 'mcunet' in args.net:
                            layer = block.mobile_inverted_conv
                        
                        for layer_name in layer_name_list:
                            conv_layer = getattr(layer, layer_name)
                            if conv_layer is None:
                                continue
                            else:
                                # ZO_scale = 1
                                ZO_scale = math.sqrt(self.ZO_Estim.n_sample / (self.ZO_Estim.n_sample + conv_layer.conv.weight.numel() + 1))
                                log_dict.update({
                                    # f"block_{block_idx}/{layer_name}/grad_w_norm": ZO_scale*torch.norm(conv_layer.conv.weight.grad),
                                    # f"block_{block_idx}/{layer_name}/grad_w_norm": ZO_scale*torch.norm(conv_layer.conv.weight.grad) / math.sqrt(conv_layer.conv.weight.numel()),
                                    # f"block_{block_idx}/{layer_name}/grad_w_norm": ZO_scale*torch.norm(conv_layer.conv.weight.grad) / torch.norm(conv_layer.conv.weight),
                                    f"block_{block_idx}/{layer_name}/grad_w_norm": ZO_scale*torch.norm(conv_layer.conv.weight.grad) / torch.norm(conv_layer.conv.weight) / math.sqrt(conv_layer.conv.weight.numel()),
                                    # f"block_{block_idx}/{layer_name}/grad_a_norm": torch.norm(conv_layer.out_grad),
                                })
                    for key, value in log_dict.items():
                        print(f'{value}')
                    
                    print('done')
                    ##### Remove hook #####
                    for hook_handle in hook_handle_list:
                        hook_handle.remove()
                
                # do SGD step
                if self.run_config.grad_accumulation_steps > 1:
                    # The gradients are computed for each mini-batch by calling loss.backward(). 
                    # This adds the gradients to the existing values instead of replacing them.
                    if (i + 1) % self.run_config.grad_accumulation_steps == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                else:
                    self.optimizer.step()
                    self.optimizer.zero_grad()  # or self.net.zero_grad()

                # measure accuracy and record loss
                losses.update(loss.item(), images.size(0))
                self.update_metric(metric_dict, output, target)

                t.set_postfix(
                    {
                        "loss": losses.avg,
                        **self.get_metric_vals(metric_dict, return_dict=True),
                        "img_size": images.size(2),
                        "lr": new_lr,
                        "loss_type": loss_type,
                        "data_time": data_time.avg,
                    }
                )
                t.update(1)
                end = time.time()
        return losses.avg, self.get_metric_vals(metric_dict)

    def train(self, args, warmup_epoch=0, warmup_lr=0):
        for epoch in range(self.start_epoch, self.run_config.n_epochs + warmup_epoch):
            train_loss, (train_top1, train_top5) = self.train_one_epoch(
                args, epoch, warmup_epoch, warmup_lr
            )
            train_log = "Train [{0}/{1}]\tloss {2:.3f}\t{4} {3:.3f}".format(
                    epoch + 1 - warmup_epoch,
                    self.run_config.n_epochs,
                    np.mean(train_loss),
                    np.mean(train_top1),
                    self.get_metric_names()[0],
                )
            self.write_log(train_log, prefix="train", should_print=False)

            train_log_dict = {
                          f"train/epoch": epoch + 1 - warmup_epoch,
                          f"train/loss": np.mean(train_loss),
                          f"train/top1": np.mean(train_top1),
                      }

            wandb.log(train_log_dict)

            if (epoch + 1) % self.run_config.validation_frequency == 0:
                img_size, val_loss, val_acc, val_acc5 = self.validate_all_resolution(
                    epoch=epoch, is_test=False
                )

                is_best = np.mean(val_acc) > self.best_acc
                self.best_acc = max(self.best_acc, np.mean(val_acc))
                val_log = "Valid [{0}/{1}]\tloss {2:.3f}\t{5} {3:.3f} ({4:.3f})".format(
                    epoch + 1 - warmup_epoch,
                    self.run_config.n_epochs,
                    np.mean(val_loss),
                    np.mean(val_acc),
                    self.best_acc,
                    self.get_metric_names()[0],
                )
                val_log += "\t{2} {0:.3f}\tTrain {1} {top1:.3f}\tloss {train_loss:.3f}\t".format(
                    np.mean(val_acc5),
                    *self.get_metric_names(),
                    top1=train_top1,
                    train_loss=train_loss
                )
                for i_s, v_a in zip(img_size, val_acc):
                    val_log += "(%d, %.3f), " % (i_s, v_a)
                self.write_log(val_log, prefix="valid", should_print=False)

                val_log_dict = {
                          f"val/epoch": epoch + 1 - warmup_epoch,
                          f"val/loss": np.mean(val_loss),
                          f"val/top1": np.mean(val_acc),
                          f"val/best_top1": self.best_acc,
                      }
                wandb.log(val_log_dict)
            else:
                is_best = False

            self.save_model(
                {
                    "epoch": epoch,
                    "best_acc": self.best_acc,
                    "optimizer": self.optimizer.state_dict(),
                    "state_dict": self.network.state_dict(),
                },
                is_best=is_best,
            )
        
        best_path = os.path.join(self.save_path, "model_best.pth.tar")
        checkpoint = torch.load(best_path, map_location="cpu")
        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
        self.network.load_state_dict(checkpoint)
        test_loss, (test_top1, test_top5) = self.validate(epoch, is_test=True, net=self.network)
        test_log = "Early stop best on best valid model. \tTest\tloss {0:.3f}\t{2} {1:.3f}".format(
            np.mean(test_loss), np.mean(test_top1), self.get_metric_names()[0]
        )
        self.write_log(test_log, prefix="valid", should_print=False)

    def reset_running_statistics(
        self, net=None, subset_size=2000, subset_batch_size=200, data_loader=None
    ):
        from ofa.imagenet_classification.elastic_nn.utils import set_running_statistics

        if net is None:
            net = self.network
        if data_loader is None:
            data_loader = self.run_config.random_sub_train_loader(
                subset_size, subset_batch_size
            )
        set_running_statistics(net, data_loader)
