# import torch
# from adaptformer import Adapter

# vit_opt_able_layers_dict = {
#     'Adapter': Adapter,
# }

# def vit_get_iterable_block_name():
#     return 'blocks'

# def vit_pre_block_forward(model, x):
#     x = model.patch_embed(x)
#     cls_token = model.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
#     if model.dist_token is None:
#         x = torch.cat((cls_token, x), dim=1)
#     else:
#         x = torch.cat((cls_token, model.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
#     x = model.pos_drop(x + model.pos_embed)
#     return x

# def vit_post_block_forward(model, x):
#     x = model.norm(x)
#     if model.dist_token is None:
#         x = model.pre_logits(x[:, 0])
#     else:
#         x = x[:, 0], x[:, 1]
    
#     if model.head_dist is not None:
#         x, x_dist = model.head(x[0]), model.head_dist(x[1])  # x must be a tuple
#         if model.training and not torch.jit.is_scripting():
#             # during inference, return the average of both classifier predictions
#             return x, x_dist
#         else:
#             return (x + x_dist) / 2
#     else:
#         x = model.head(x)
#     return x

import torch

opt_able_layers_dict = {
    'Conv2d': torch.nn.Conv2d,
}

def get_iterable_block_name():
    return 'blocks'

def pre_block_forward(model, x):
    x = model.first_conv(x)
    return x

def post_block_forward(model, x):
    if model.feature_mix_layer is not None:
        x = model.feature_mix_layer(x)
    x = x.mean(3).mean(2)
    x = model.classifier(x)
    return x