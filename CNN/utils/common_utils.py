import torch
import torch.nn as nn
from torch.nn.utils.fusion import fuse_conv_bn_eval

from tqdm import tqdm
from sklearn.cluster import KMeans

__all__ = [
    'module_require_grad', 'set_module_grad_status', 'enable_bn_update', 'disable_bn_update', 'enable_bias_update',
    'weight_quantization', 'fuse_bn', 'fuse_bn_add_gn', 'replace_bn_with_insn', 'replace_conv2d_with_my_conv2d'
]


def module_require_grad(module):
    return module.parameters().__next__().requires_grad


def set_module_grad_status(module, flag=False):
    if isinstance(module, list):
        for m in module:
            set_module_grad_status(m, flag)
    else:
        for p in module.parameters():
            p.requires_grad = flag


def enable_bn_update(model):
    for m in model.modules():
        if type(m) in [nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d] and m.weight is not None:
            set_module_grad_status(m, True)


def disable_bn_update(model):
    for m in model.modules():
        if type(m) in [nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d] and m.weight is not None:
            set_module_grad_status(m, False)


def enable_bias_update(model):
    for m in model.modules():
        for name, param in m.named_parameters():
            if name == 'bias':
                param.requires_grad = True


def k_means_cpu(weight, n_clusters, init='k-means++', max_iter=50):
    # flatten the weight for computing k-means
    org_shape = weight.shape
    weight = weight.reshape(-1, 1)  # single feature
    if n_clusters > weight.size:
        n_clusters = weight.size

    k_means = KMeans(n_clusters=n_clusters, init=init, n_init=1, max_iter=max_iter)
    k_means.fit(weight)

    centroids = k_means.cluster_centers_
    labels = k_means.labels_
    labels = labels.reshape(org_shape)
    return torch.from_numpy(centroids).view(1, -1), torch.from_numpy(labels).int()


def reconstruct_weight_from_k_means_result(centroids, labels):
    weight = torch.zeros_like(labels).float()
    for i, c in enumerate(centroids.cpu().numpy().squeeze()):
        weight[labels == i] = c.item()
    return weight


def quantization(layer, bits=8, max_iter=50):
    w = layer.weight.data
    centroids, labels = k_means_cpu(w.cpu().numpy(), 2 ** bits, max_iter=max_iter)
    w_q = reconstruct_weight_from_k_means_result(centroids, labels)
    layer.weight.data = w_q.float()


def weight_quantization(model, bits=8, max_iter=50):
    if bits is None:
        return
    to_quantize_modules = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            if not m.weight.requires_grad:
                to_quantize_modules.append(m)

    with tqdm(total=len(to_quantize_modules),
              desc='%d-bits quantization start' % bits) as t:
        for m in to_quantize_modules:
            quantization(m, bits, max_iter)
            t.update()

### Fuse BN add GN

def min_divisible_value(n1, v1):
    """make sure v1 is divisible by n1, otherwise decrease v1"""
    if v1 >= n1:
        return n1
    while n1 % v1 != 0:
        v1 -= 1
    return v1

##### fuse BN, and delete BN
def fuse_bn(model):
    for m in model.modules():
        to_replace_dict = {}
        to_pop_name_list = []
        conv_name = None
        conv_sub_m = None
        for name, sub_m in m.named_children():
            if isinstance(sub_m, nn.Conv2d):
                conv_name = name
                conv_sub_m = sub_m
            elif isinstance(sub_m, nn.BatchNorm2d) and conv_sub_m is not None:

                conv_sub_m.eval()
                sub_m.eval()
                fused_conv = fuse_conv_bn_eval(conv_sub_m, sub_m)
                fused_conv.train()
                to_replace_dict[conv_name] = fused_conv
                to_pop_name_list.append(name)

        for to_pop_name in to_pop_name_list:
            m._modules.pop(to_pop_name)
        m._modules.update(to_replace_dict)

##### fuse BN, and add GN
def fuse_bn_add_gn(model, gn_channel_per_group=None):
    for m in model.modules():
        to_replace_dict = {}
        conv_name = None
        conv_sub_m = None
        for name, sub_m in m.named_children():
            if isinstance(sub_m, nn.Conv2d):
                conv_name = name
                conv_sub_m = sub_m
            elif isinstance(sub_m, nn.BatchNorm2d) and conv_sub_m is not None:

                conv_sub_m.eval()
                sub_m.eval()
                fused_conv = fuse_conv_bn_eval(conv_sub_m, sub_m)
                fused_conv.train()
                to_replace_dict[conv_name] = fused_conv
                
                num_groups = sub_m.num_features // min_divisible_value(
                    sub_m.num_features, gn_channel_per_group
                )
                gn_m = nn.GroupNorm(
                    num_groups=num_groups,
                    num_channels=sub_m.num_features,
                    eps=sub_m.eps,
                    affine=True,
                )

                # don't load weight. just initialize
                # gn_m.weight.data.copy_(sub_m.weight.data)
                # gn_m.bias.data.copy_(sub_m.bias.data)

                # load requires_grad
                gn_m.weight.requires_grad = sub_m.weight.requires_grad
                gn_m.bias.requires_grad = sub_m.bias.requires_grad

                to_replace_dict[name] = gn_m
        m._modules.update(to_replace_dict)

##### replace BN with InstanceNorm
def replace_bn_with_insn(model):
    for m in model.modules():
        to_replace_dict = {}
        for name, sub_m in m.named_children():
            if isinstance(sub_m, nn.BatchNorm2d):
                insn_m = nn.InstanceNorm2d(
                    num_features=sub_m.num_features,
                    eps=sub_m.eps,
                    affine=True,
                    track_running_stats=True,
                )

                # load weight
                insn_m.weight.data.copy_(sub_m.weight.data)
                insn_m.bias.data.copy_(sub_m.bias.data)
                # load running statistics
                insn_m.running_mean.copy_(sub_m.running_mean)
                insn_m.running_var.copy_(sub_m.running_var)
                # load requires_grad
                insn_m.weight.requires_grad = sub_m.weight.requires_grad
                insn_m.bias.requires_grad = sub_m.bias.requires_grad

                to_replace_dict[name] = insn_m
        m._modules.update(to_replace_dict)

from ofa.utils import MyConv2d
def replace_conv2d_with_my_conv2d(net, ws_eps=None):
    if ws_eps is None:
        return

    for m in net.modules():
        to_update_dict = {}
        for name, sub_module in m.named_children():
            if isinstance(sub_module, nn.Conv2d):
                to_update_dict[name] = sub_module
        for name, sub_module in to_update_dict.items():
            if sub_module.bias is not None:
                bias = True
            else:
                bias = False
            m._modules[name] = MyConv2d(
                sub_module.in_channels,
                sub_module.out_channels,
                sub_module.kernel_size,
                sub_module.stride,
                sub_module.padding,
                sub_module.dilation,
                sub_module.groups,
                bias,
            )
            # load weight
            m._modules[name].load_state_dict(sub_module.state_dict())
            # load requires_grad
            m._modules[name].weight.requires_grad = sub_module.weight.requires_grad
            if sub_module.bias is not None:
                m._modules[name].bias.requires_grad = sub_module.bias.requires_grad
    # set ws_eps
    for m in net.modules():
        if isinstance(m, MyConv2d):
            m.WS_EPS = ws_eps