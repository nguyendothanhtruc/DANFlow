import torch.nn as nn
import sequence_inn as sqi

from model.attention import ECAAttention, ContextBlock, TripletAttention, ShuffleAttention
from model import FlowStep

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True),
        )

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, num_dw, stride, expand_ratio, attention=None):
        super(InvertedResidual, self).__init__()
        if stride not in [1, 2]:
            raise ValueError("Stride must be 1 or 2.")

        if attention not in ['eca', 'gcnet', 'triplet', 'shuffle']:
             raise ValueError("Invalid attention type. Supported types are 'eca', 'gcnet', 'triplet', and 'shuffle'.")
        
        self.stride = stride
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers_pw = []
        if expand_ratio != 1:
            layers_pw.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        self.pw = nn.Sequential(*layers_pw)

        layers_dw = []
        for _ in range(num_dw):
          layers_dw.extend([ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim)],)
        self.dw = nn.Sequential(*layers_dw)

        layers_pw_linear = [
          nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
          nn.BatchNorm2d(oup)
        ]
        self.pw_linear = nn.Sequential(*layers_pw_linear)

        attention_stage = self._create_attention_stage(attention, oup)
        self.attention_stage = nn.Sequential(*attention_stage) if attention is not None else None
        self.attention = attention

    def _create_attention_stage(self, attention, oup):
        if attention is None:
            return []
        
        if attention == 'eca':
            return [ECAAttention(kernel_size=3)]
        
        elif attention == 'gcnet':
            return [ContextBlock(inplanes=oup, ratio=0.25, pooling_type='att', fusion_types=('channel_mul', ))]
        
        elif attention == 'triplet':
            return [TripletAttention()]
        
        elif attention == 'shuffle':
            return [ShuffleAttention(channel=oup, reduction=16, G=8)]

    def forward(self, x):
        x = self.pw(x)
        x = x + self.dw(x)
        x = self.pw_linear(x)

        return x if self.attention is None else self.attention_stage(x)


def subnet_conv_func(num_dw, attention=None):
    def subnet_conv(in_channels, out_channels):
        return InvertedResidual(
            inp = in_channels, 
            oup = out_channels, 
            num_dw = num_dw, 
            stride = 1, 
            expand_ratio = 6, 
            attention = attention, 
        )
    
    return subnet_conv


def build_flowstep(input_chw, num_flowstep, num_dw, double_subnets, clamp=2.0, attention=None):
    nodes = sqi.SequenceINN(*input_chw)

    for _ in range(num_flowstep):
      nodes.append(
          FlowStep,
          subnet_constructor = subnet_conv_func(num_dw=num_dw, attention=attention),
          affine_clamping = clamp,
          permute_soft = False,
          double_subnets = double_subnets,
      )

    return nodes
