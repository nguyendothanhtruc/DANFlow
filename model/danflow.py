import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import sequence_inn as sqi
from .attention import ECAAttention, ContextBlock, TripletAttention, ShuffleAttention
from .flowstep import FlowStep
from . import constants as const


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True),
        )

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, num_dw, stride, expand_ratio, attention, ecaa_kernel_size):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers_pw = []
        if expand_ratio != 1:
            # pw
            layers_pw.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        self.pw = nn.Sequential(*layers_pw)

        # dw
        layers_dw = []
        for i in range(num_dw):
          layers_dw.extend([ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim)],)
        self.dw = nn.Sequential(*layers_dw)

        layers_pw_linear = [
          # pw-linear
          nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
          nn.BatchNorm2d(oup)
        ]
        self.pw_linear = nn.Sequential(*layers_pw_linear)

        if attention is not None:
          attention_stage=[]
          if attention=="eca":
            attention_stage.append(ECAAttention(kernel_size=ecaa_kernel_size))
          elif attention=="gcnet":
            attention_stage.append(ContextBlock(inplanes=oup, ratio=0.25))
          self.attention_stage = nn.Sequential(*attention_stage)
        
        self.attention = attention

    def forward(self, x):
        x = self.pw(x)
        x = x + self.dw(x)
        x = self.pw_linear(x)

        if self.attention is None:
          return x      
        else:
          return self.attention_stage(x)


def subnet_conv_func(num_dw, attention=None):
    def subnet_conv(in_channels, out_channels, ecaa_kernel_size):
        return InvertedResidual(
            inp = in_channels, 
            oup = out_channels, 
            num_dw = num_dw, 
            stride = 1, 
            expand_ratio = 6, 
            attention = attention, 
            ecaa_kernel_size=ecaa_kernel_size
        )
    
    return subnet_conv


def build_flowstep(input_chw, flow_steps, num_dw, double_subnets, clamp=2.0, attention=None):
    nodes = sqi.SequenceINN(*input_chw)

    print("*"*10, " CONFIGS: ","*"*10)
    print("flow_steps:", flow_steps)
    print("double_subnets:", double_subnets)
    print("num_dw:", num_dw)
    print("attention:", attention)

    for _ in range(flow_steps):
      nodes.append(
          FlowStep,
          subnet_constructor = subnet_conv_func(num_dw=num_dw, attention=attention),
          affine_clamping = clamp,
          permute_soft = False,
          double_subnets = double_subnets,
      )

    return nodes


class DANFlow(nn.Module):
    def __init__(
        self,
        input_size,
        backbone_name,
        flow_steps,
        num_dw=1,
        double_subnets=False,
        attention=None,
    ):
        super(DANFlow, self).__init__()
        assert (
            backbone_name in const.SUPPORTED_BACKBONES
        ), "backbone_name must be one of {}".format(const.SUPPORTED_BACKBONES)

        if backbone_name in [const.BACKBONE_CAIT, const.BACKBONE_DEIT]:
            self.feature_extractor = timm.create_model(backbone_name, pretrained=True)
            channels = [768]
            scales = [16]
        else:
            self.feature_extractor = timm.create_model(
                backbone_name,
                pretrained=True,
                features_only=True,
                out_indices=[1, 2, 3],
            )
            channels = self.feature_extractor.feature_info.channels()
            scales = self.feature_extractor.feature_info.reduction()

            # for transformers, use their pretrained norm w/o grad
            # for resnets, self.norms are trainable LayerNorm
            self.norms = nn.ModuleList()
            for in_channels, scale in zip(channels, scales):
                self.norms.append(
                    nn.LayerNorm(
                        [in_channels, int(input_size / scale), int(input_size / scale)],
                        elementwise_affine=True,
                    )
                )

        for param in self.feature_extractor.parameters():
            param.requires_grad = False # block params for not finetuning in feature extractor

        self.nf_flows = nn.ModuleList()
        for in_channels, scale in zip(channels, scales):
            self.nf_flows.append(
                build_flowstep(
                    [in_channels, int(input_size / scale), int(input_size / scale)],
                    flow_steps=flow_steps,
                    num_dw=num_dw,
                    double_subnets=double_subnets,
                    attention=attention
                )
            )

        print(
        "DANFlow Param#: {}".format(
            sum(p.numel() for p in self.nf_flows.parameters() if p.requires_grad)
        )
    )
        self.input_size = input_size
        self.flow_steps = flow_steps

    def forward(self, data):
        self.feature_extractor.eval()
        if isinstance(
            self.feature_extractor, timm.models.vision_transformer.VisionTransformer
        ):
            x = self.feature_extractor.patch_embed(data)
            cls_token = self.feature_extractor.cls_token.expand(x.shape[0], -1, -1)
            if self.feature_extractor.dist_token is None:
                x = torch.cat((cls_token, x), dim=1)
            else:
                x = torch.cat(
                    (
                        cls_token,
                        self.feature_extractor.dist_token.expand(x.shape[0], -1, -1),
                        x,
                    ),
                    dim=1,
                )
            x = self.feature_extractor.pos_drop(x + self.feature_extractor.pos_embed)
            for i in range(4):  #Block Index = 3
                x = self.feature_extractor.blocks[i](x)
            x = self.feature_extractor.norm(x)
            x = x[:, 2:, :]
            N, _, C = x.shape
            x = x.permute(0, 2, 1)
            x = x.reshape(N, C, self.input_size // 16, self.input_size // 16)
            features = [x]
        elif isinstance(self.feature_extractor, timm.models.cait.Cait):
            x = self.feature_extractor.patch_embed(data)
            x = x + self.feature_extractor.pos_embed
            x = self.feature_extractor.pos_drop(x)
            for i in range(41):  # paper Table 6. Block Index = 40
                x = self.feature_extractor.blocks[i](x)
            N, _, C = x.shape
            x = self.feature_extractor.norm(x)
            x = x.permute(0, 2, 1)
            x = x.reshape(N, C, self.input_size // 16, self.input_size // 16)
            features = [x]
        else:
            features = self.feature_extractor(data)
            features = [self.norms[i](feature) for i, feature in enumerate(features)]


        loss = 0
        outputs = []
        is_final = False
        for i, feature in enumerate(features):
            output, log_jac_dets = self.nf_flows[i](feature)
            loss += torch.mean(
                0.5 * torch.sum(output**2, dim=(1, 2, 3)) - log_jac_dets
            )
            outputs.append(output)
        ret = {"loss": loss}

        if not self.training:
            anomaly_map_list = []
            for output in outputs:
              log_prob = -torch.mean(output**2, dim=1, keepdim=True) * 0.5
              prob = torch.exp(log_prob)
              a_map = F.interpolate(
                  -prob,
                  size=[self.input_size, self.input_size],
                  mode="bilinear",
                  align_corners=False,
              )
              anomaly_map_list.append(a_map)
            anomaly_map_list = torch.stack(anomaly_map_list, dim=-1)
            anomaly_map = torch.mean(anomaly_map_list, dim=-1)
            ret["anomaly_map"] = anomaly_map
        return ret