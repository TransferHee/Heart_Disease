import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Swish(nn.Module):
    def __init__(self, name=None):
        super().__init__()
        self.name = name
        
    def forward(self,x):
        return x * torch.sigmoid(x)
    
class Conv2dSamepadding(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True, name=None):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation, groups=groups, bias=bias)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2
        self.name = name
        
    def forward(self,x):
        input_h, input_w = x.size()[2:]
        kernel_h, kernel_w = self.weight.size()[2:]
        stride_h, stride_w = self.stride
        output_h, output_w = math.ceil(input_h / stride_h), math.ceil(input_w / stride_w)
        pad_h = max((output_h-1) * self.stride[0] + (kernel_h-1) * self.dilation[0]+1 - input_h, 0)
        pad_w = max((output_w-1) * self.stride[1] + (kernel_w-1) * self.dilation[1]+1 - input_w, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    
class BatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, name=None):
        super().__init__(num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)
        self.name = name
        
def drop_connect(inputs, drop_connect_rate, training):
    if not training:
        return inputs
    batch_size = inputs.shape[0]
    keep_prob = 1.0 - drop_connect_rate
    random_tensor = keep_prob
    random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device)
    binary_tensor = torch.floor(random_tensor)
    output = inputs / keep_prob * binary_tensor
    return output

class MBConvBlock(nn.Module):
    def __init__(self, block_args, global_params, idx):
        super().__init__()
        
        block_name = 'blocks_' + str(idx) + '_'
        
        self.block_args = block_args
        self.batch_norm_momentum = 1 - global_params.batch_norm_momentum
        self.batch_norm_epsilon = global_params.batch_norm_epsilon
        self.has_se = (self.block_args.se_ratio is not None) and (0 < self.block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip
        
        self.swish = Swish(block_name + '_swish')
        
        # Expansion phase
        in_channels = self.block_args.input_filters
        out_channels = self.block_args.input_filters * self.block_args.expand_ratio
        if self.block_args.expand_ratio != 1:
            self._expand_conv = Conv2dSamepadding(in_channels=in_channels,
                                                 out_channels=out_channels,
                                                 kernel_size=1,
                                                 bias=False,
                                                 name=block_name + 'expansion_conv')
            self._bn0 = BatchNorm2d(num_features=out_channels,
                                   momentum=self.batch_norm_momentum,
                                   eps=self.batch_norm_epsilon,
                                   name=block_name + 'expansion_batch_norm')
        
        # Depth-wise convolution phase
        kernel_size = self.block_args.kernel_size
        strides = self.block_args.strides
        self._depthwise_conv = Conv2dSamepadding(in_channels=out_channels,
                                                out_channels=out_channels,
                                                groups=out_channels,
                                                kernel_size=kernel_size,
                                                stride=strides,
                                                bias=False,
                                                name=block_name + 'depthwise_conv')
        self._bn1 = BatchNorm2d(num_features=out_channels,
                               momentum=self.batch_norm_momentum,
                               eps=self.batch_norm_epsilon,
                               name=block_name + 'depthwise_batch_norm')
        
        # Squeeze and Excitation
        if self.has_se:
            num_squeezed_channels = max(1, int(self.block_args.input_filters * self.block_args.se_ratio))
            self._se_reduce = Conv2dSamepadding(in_channels=out_channels,
                                                out_channels=num_squeezed_channels,
                                                kernel_size=1,
                                                name=block_name + 'se_reduce')
            self._se_expand = Conv2dSamepadding(in_channels=num_squeezed_channels,
                                                out_channels=out_channels,
                                                kernel_size=1,
                                                name=block_name + 'se_expand')
            
        # output phase
        final_output_channels = self.block_args.output_filters
        self._project_conv = Conv2dSamepadding(in_channels=out_channels,
                                                out_channels=final_output_channels,
                                                kernel_size=1,
                                                bias=False,
                                                name=block_name + 'output_conv')
        self._bn2 = BatchNorm2d(num_features=final_output_channels,
                               momentum=self.batch_norm_momentum,
                               eps=self.batch_norm_epsilon,
                               name=block_name + 'output_batch_norm')
        
    def forward(self, x, drop_connect_rate=None):
        identity = x
        if self.block_args.expand_ratio != 1:
            x = self._expand_conv(x)
            x = self._bn0(x)
            x = self.swish(x)
        
        x = self._depthwise_conv(x)
        x = self._bn1(x)
        x = self.swish(x)
        
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x,1)
            x_squeezed = self._se_expand(self.swish(self._se_reduce(x_squeezed)))
            x = torch.sigmoid(x_squeezed) * x
        
        x = self._bn2(self._project_conv(x))
        
        input_filters, output_filters = self.block_args.input_filters, self.block_args.output_filters
        if self.id_skip and self.block_args.strides == 1 and input_filters == output_filters:
            if drop_connect_rate:
                x = drop_connect(x, drop_connect_rate=drop_connect_rate, training=self.training)
            x = x + identity
        return x
    
def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

def up_conv(in_channels, out_channels):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

def custom_head(in_channels, out_channels):
    return nn.Sequential(
        nn.Dropout(),
        nn.Linear(in_channels, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(),
        nn.Linear(512, out_channels),
    )

import re
from collections import namedtuple

GlobalParams = namedtuple('GlobalParams', ['batch_norm_momentum','batch_norm_epsilon', 'dropout_rate','num_classes',
                                           'width_coefficient','depth_coefficient', 'depth_divisor', 'min_depth',
                                           'drop_connect_rate'])
GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)

BlockArgs = namedtuple('BlockArgs', ['kernel_size', 'num_repeat', 'input_filters', 'output_filters', 'expand_ratio',
                                    'id_skip', 'strides', 'se_ratio'])
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)

def round_filters(filters, global_params):
    multiplier = global_params.width_coefficient
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    if not multiplier:
        return filters
    
    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return int(new_filters)

def round_repeats(repeats, global_params):
    multiplier = global_params.depth_coefficient
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))

def get_efficientnet_params(model_name, override_params=None):
    params_dict = {'efficientnet-b0': (1.0, 1.0, 224, 0.2)}
    if model_name not in params_dict.keys():
        raise KeyError
    
    width_coefficient, depth_coefficient, _, dropout_rate = params_dict[model_name]
    
    blocks_args = [
        'r1_k3_s11_e1_i32_o16_se0.25', 'r2_k3_s22_e6_i16_o24_se0.25',
        'r2_k5_s22_e6_i24_o40_se0.25', 'r3_k3_s22_e6_i40_o80_se0.25',
        'r3_k5_s11_e6_i80_o112_se0.25', 'r4_k5_s22_e6_i112_o192_se0.25',
        'r1_k3_s11_e6_i192_o320_se0.25',
    ]
    global_params = GlobalParams(
    batch_norm_momentum=0.99,
    batch_norm_epsilon=1e-3,
    dropout_rate=dropout_rate,
    drop_connect_rate=0.2,
    num_classes=1000,
    width_coefficient=width_coefficient,
    depth_coefficient=depth_coefficient,
    depth_divisor=8,
    min_depth=None)
    
    if override_params:
        global_params = global_params._replace(**override_params)
        
    decoder = BlockDecoder()
    return decoder.decode(blocks_args), global_params

class BlockDecoder(object):
    
    @staticmethod
    def _decode_block_string(block_string):
        
        assert isinstance(block_string, str)
        ops = block_string.split('_')
        options = {}
        for op in ops:
            splits = re.split(r'(\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value
        if 's' not in options or len(options['s']) != 2:
            raise ValueError('s가 없어?')
            
        return BlockArgs(
            kernel_size = int(options['k']),
            num_repeat = int(options['r']),
            input_filters = int(options['i']),
            output_filters = int(options['o']),
            expand_ratio = int(options['e']),
            id_skip = ('noskip' not in block_string),
            se_ratio = float(options['se']) if 'se' in options else None,
            strides = [int(options['s'][0]), int(options['s'][1])]
        )
    
    def decode(self, string_list):
        
        assert isinstance(string_list, list)
        blocks_args = []
        for block_string in string_list:
            blocks_args.append(self._decode_block_string(block_string))
        return blocks_args
    
class EfficientNet(nn.Module):
    
    def __init__(self, block_args_list, global_params):
        super().__init__()
        
        self.block_args_list = block_args_list
        self.global_params = global_params
        
        batch_norm_momentum = 1-self.global_params.batch_norm_momentum
        batch_norm_epsilon = self.global_params.batch_norm_epsilon
        
        in_channels = 3
        out_channels = round_filters(32, self.global_params)
        self._conv_stem = Conv2dSamepadding(in_channels,
                                           out_channels,
                                           kernel_size=3,
                                           stride=2,
                                           bias=False,
                                           name='stem_conv')
        self._bn0 = BatchNorm2d(num_features=out_channels,
                               momentum=batch_norm_momentum,
                               eps=batch_norm_epsilon,
                               name='stem_batch_norm')
        self._swish = Swish(name='swish')
        
        idx = 0
        self._blocks = nn.ModuleList([])
        for block_args in self.block_args_list:
            
            block_args = block_args._replace(
                input_filters = round_filters(block_args.input_filters, self.global_params),
                output_filters = round_filters(block_args.output_filters, self.global_params),
                num_repeat = round_repeats(block_args.num_repeat, self.global_params)
            )
            
            self._blocks.append(MBConvBlock(block_args, self.global_params, idx=idx))
            idx += 1
            
            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters=block_args.output_filters, strides=1)
                
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(block_args, self.global_params, idx=idx))
                idx += 1
        
        # Head
        in_channels = block_args.output_filters
        out_channels = round_filters(1280, self.global_params)
        self._conv_head = Conv2dSamepadding(in_channels,
                                           out_channels,
                                           kernel_size=1,
                                           bias=False,
                                           name='head_conv')
        self._bn1 = BatchNorm2d(num_features=out_channels,
                              momentum=batch_norm_momentum,
                              eps=batch_norm_epsilon,
                              name='head_batch_norm')
        
        # Final linear layer
        self.dropout_rate = self.global_params.dropout_rate
        self._fc = nn.Linear(out_channels, self.global_params.num_classes)
        
    def forward(self, x):
        # Stem
        x = self._conv_stem(x)
        x = self._bn0(x)
        x = self._swish(x)
        
        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self.global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= idx / len(self._blocks)
            x = block(x, drop_connect_rate)
        
        # Head
        x = self._conv_head(x)
        x = self._bn1(x)
        x = self._swish(x)
        
        # Pooling and Dropout
        x = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        if self.dropout_rate > 0:
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            
        # FC layer
        x = self._fc(x)
        return x
    
    @classmethod
    def from_name(cls, model_name, *, n_classes=1000, pretrained=False):
        return _get_model_by_name(model_name, classes=n_classes, pretrained=pretrained)
    
    @classmethod
    def encoder(cls, model_name, *, pretrained=False):
        model = cls.from_name(model_name, pretrained=pretrained)
        
        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                
                self.name = model_name
                
                self.global_params = model.global_params
                
                self.stem_conv = model._conv_stem
                self.stem_batch_norm = model._bn0
                self.stem_swish = Swish(name='stem_swish')
                self.blocks = model._blocks
                self.head_conv = model._conv_head
                self.head_batch_norm = model._bn1
                self.head_swish = Swish(name='head_swish')
                
            def forward(self,x):
                # stem
                x = self.stem_conv(x)
                x = self.stem_batch_norm(x)
                x = self.stem_swish(x)
                
                # blocks
                for idx, block in enumerate(self.blocks):
                    drop_connect_rate = self.global_params.drop_connect_rate
                    if drop_connect_rate:
                        drop_connect_rate *= idx / len(self.blocks)
                    x = block(x, drop_connect_rate)
                
                # head
                x = self.head_conv(x)
                x = self.head_batch_norm(x)
                x = self.head_swish(x)
                return x
        
        return Encoder()
    # classmethod custom_head 생략
    
def _get_model_by_name(model_name, classes=1000, pretrained=False):
    block_args_list, global_params = get_efficientnet_params(model_name, override_params={'num_classes': classes})
    model = EfficientNet(block_args_list, global_params)
    try:
        if pretrained:
            print('프리트레인?')
    except KeyError as e:
        print(f'Note: Currently model {e} blr blr')
    return model

from collections import OrderedDict

def get_blocks_to_be_concat(model, x):
    shapes = set()
    blocks = OrderedDict()
    hooks = []
    count = 0
    
    def register_hook(module):
        
        def hook(module, input, output):
            try:
                nonlocal count
                if module.name == f'blocks_{count}_output_batch_norm':
                    count +=1
                    shape = output.size()[-2:]
                    if shape not in shapes:
                        shapes.add(shape)
                        blocks[module.name] = output
                elif module.name == 'head_swish':
                    blocks.popitem()
                    blocks[module.name] = output
            except AttributeError:
                pass
        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))
    model.apply(register_hook)
    
    model(x)
    
    for h in hooks:
        h.remove()
    return blocks


class EfficientUnet(nn.Module):
    def __init__(self, encoder, out_channels=2, concat_input=True):
        super().__init__()
        
        self.encoder = encoder
        self.concat_input = concat_input
        
        self.up_conv1 = up_conv(self.n_channels, 512)
        self.double_conv1 = double_conv(self.size[0], 512)
        self.up_conv2 = up_conv(512, 256)
        self.double_conv2 = double_conv(self.size[1], 256)
        self.up_conv3 = up_conv(256, 128)
        self.double_conv3 = double_conv(self.size[2], 128)
        self.up_conv4 = up_conv(128, 64)
        self.double_conv4 = double_conv(self.size[3], 64)
        
        if self.concat_input:
            self.up_conv_input = up_conv(64, 32)
            self.double_conv_input = double_conv(self.size[4], 32)
        
        self.final_conv = nn.Conv2d(self.size[5], out_channels, kernel_size=1)
        
    @property
    def n_channels(self):
        # only for efficientnet-b0 version
        return 1280
    
    @property
    def size(self):
        # only for efficientnet-b0 version
        return [592, 296, 152, 80, 35, 32]
    
    def forward(self,x):
        input_ = x
        
        blocks = get_blocks_to_be_concat(self.encoder, x)
        _, x = blocks.popitem()
        
        x = self.up_conv1(x)
        x = torch.cat([x, blocks.popitem()[1]], dim=1)
        x = self.double_conv1(x)
        
        x = self.up_conv2(x)
        x = torch.cat([x, blocks.popitem()[1]], dim=1)
        x = self.double_conv2(x)
        
        x = self.up_conv3(x)
        x = torch.cat([x, blocks.popitem()[1]], dim=1)
        x = self.double_conv3(x)
        
        x = self.up_conv4(x)
        x = torch.cat([x, blocks.popitem()[1]], dim=1)
        x = self.double_conv4(x)
        
        if self.concat_input:
            x = self.up_conv_input(x)
            x = torch.cat([x, input_], dim=1)
            x = self.double_conv_input(x)
            
        x = self.final_conv(x)
        
        return x
    
def get_efficientunet_b0(out_channels=2, concat_input=True, pretrained=False):
    encoder = EfficientNet.encoder('efficientnet-b0', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input)
    return model
