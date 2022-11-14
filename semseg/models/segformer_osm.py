import torch
from torch import Tensor
from torch.nn import functional as F
from semseg.models.base import BaseModel
from semseg.models.heads import SegFormerHead

import math
from torch import nn
from semseg.models.backbones import *
from semseg.models.layers import trunc_normal_


class SegFormerOSM(nn.Module):
    def __init__(self, backbone: str = 'MiT-B0', backbone1: str = 'MiT-B0', num_classes: int = 19) -> None:
        super().__init__()
        b1, v1 = backbone.split('-')
        self.backbone = eval(b1)(3, v1)

        b2, v2 = backbone1.split('-')
        self.backbone1= eval(b2)(1, v2)

        self.decode_head = SegFormerHead(self.backbone.channels, 256 if 'B0' in backbone or 'B1' in backbone else 768, num_classes)
        self.apply(self._init_weights)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        ret1 = self.backbone(x)
        ret2 = self.backbone1(y)
        ret = ret1 + ret2
        ret = []
        for r1, r2 in zip(ret1, ret2):
            ret.append(r1+r2)
        ret = tuple(ret)
        ret= self.decode_head(ret)   # 4x reduction in image size
        ret= F.interpolate(ret, size=x.shape[2:], mode='bilinear', align_corners=False)    # to original image shape
        return ret
    
    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def init_pretrained(self, pretrained: str = None) -> None:
        if pretrained:
            self.backbone.load_state_dict(torch.load(pretrained, map_location='cpu'), strict=False)
    

if __name__ == '__main__':
    model = SegFormerOSM('MiT-B0')
    # model.load_state_dict(torch.load('checkpoints/pretrained/segformer/segformer.b0.ade.pth', map_location='cpu'))
    x = torch.zeros(1, 3, 512, 512)
    y = model(x)
    print(y.shape)