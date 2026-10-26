"""语言条件通道残差诊断器；推理仅消费文本向量和真实BEV，无GT输入。"""
import torch
from torch import nn
from torch.nn import functional as F

class PerceptionAdapter(nn.Module):
    def __init__(self,core,channel_enabled=False):
        super().__init__();self.core=core;self.channel_enabled=channel_enabled
        self.down=nn.Conv2d(256,16,1,bias=False)
        self.text_gate=nn.Sequential(nn.LayerNorm(3072),nn.Linear(3072,16))
        self.up=nn.Conv2d(16,256,1,bias=False)
        nn.init.zeros_(self.up.weight)

    @property
    def alpha(self):return self.core.alpha
    @property
    def flags(self):return self.core.flags
    def set_calibration(self,row):self.core.set_calibration(row)

    def forward_with_details(self,bev,text,enabled=True):
        result=self.core.forward_with_details(bev,text,enabled)
        residual=torch.zeros_like(bev)
        if self.channel_enabled and enabled:
            hidden=F.gelu(self.down(F.group_norm(bev,1)))
            gate=2*torch.sigmoid(self.text_gate(text))[:,:,None,None]
            # Same fixed inherited alpha, no learned gain or alpha sweep.
            residual=self.alpha*result['attention']*torch.tanh(self.up(hidden*gate))
        result['enhanced_bev']=result['enhanced_bev']+residual
        result['channel_residual']=residual
        return result
