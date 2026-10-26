"""第三轮：限定词读出、同象限实例排序、背景难负例；GT仅进入loss。"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from .spatial_v2_model import DirectionReadout, ContrastFusion, balanced_instance_loss, contrast_loss
from .spatial_annotations import project_attention

VARIANTS = {
    'control': dict(selector=False, ranking=False, hard_negative=False, geometry=True, fusion=True),
    'selector': dict(selector=True, ranking=False, hard_negative=False, geometry=True, fusion=True),
    'ranking': dict(selector=True, ranking=True, hard_negative=False, geometry=True, fusion=True),
    'hardneg': dict(selector=True, ranking=False, hard_negative=True, geometry=True, fusion=True),
    'combined': dict(selector=True, ranking=True, hard_negative=True, geometry=True, fusion=True),
    'nogeometry': dict(selector=True, ranking=True, hard_negative=True, geometry=False, fusion=True),
    'nofusion': dict(selector=True, ranking=True, hard_negative=True, geometry=True, fusion=False),
}

class InstanceReadout(DirectionReadout):
    """额外读取nearest/farthest上下文token；无规则距离选择或GT输入。"""
    def __init__(self, base):
        super().__init__(base)
        self.hidden_size = 4 * base.hidden_size
        self.selector_ids = {i for word in ('nearest','farthest') for i in base.tokenizer.encode(word, add_special_tokens=False)}

    def forward(self, texts, device):
        directions = super().forward(texts, device)
        tokens = self.base.tokenize(texts, device)
        with torch.no_grad():
            hidden = self.base.model(**tokens).last_hidden_state
            selected = torch.zeros_like(tokens['attention_mask'], dtype=torch.bool)
            for i in self.selector_ids: selected |= tokens['input_ids'] == i
            selected &= tokens['attention_mask'].bool()
            relation = (hidden * selected[...,None]).sum(1) / selected.sum(1, keepdim=True).clamp_min(1)
            relation = torch.where(selected.any(1,keepdim=True), relation, directions[:,:768])
        return torch.cat((directions,relation),dim=1)

class InstanceFusion(ContrastFusion):
    def __init__(self, initial_state, v2_state, flags):
        super().__init__(initial_state)
        self.load_state_dict(v2_state, strict=True)
        self.flags = dict(flags)
        self.selector_query = nn.Sequential(nn.LayerNorm(768),nn.Linear(768,32))
        nn.init.zeros_(self.selector_query[-1].weight); nn.init.zeros_(self.selector_query[-1].bias)
        self.selector_visual = nn.Conv2d(256,32,1,bias=False)
        self.selector_position = nn.Conv2d(5,32,1,bias=False)
        # Fixed across all ablations. No alpha amplification or hidden gain tuning.
        self.alpha.requires_grad_(False)

    def forward_with_details(self, bev_features, pooled_text, enabled=True):
        d = super().forward_with_details(bev_features, pooled_text[:,:2304], enabled)
        logits = torch.logit(d['attention'].clamp(1e-5,1-1e-5))
        if not self.flags['geometry']: logits = logits - d['geometry_logits']
        if self.flags['selector']:
            keys = self.selector_visual(bev_features)
            if self.flags['geometry']: keys = keys + self.selector_position(self.position_features)
            keys = F.normalize(keys,dim=1)
            q = self.selector_query(pooled_text[:,2304:])
            logits = logits + (keys * q[:,:,None,None]).sum(1,keepdim=True)
        a = logits.sigmoid()
        enhanced = bev_features + self.alpha * bev_features * a if self.flags['fusion'] else bev_features
        d.update(attention=a, logits=logits, enhanced_bev=enhanced)
        return d

def hard_negative_loss(logits, gt, target):
    """真实GT背景topK，K=目标像素数×4；不以整个背景平均稀释难例。"""
    negatives = F.softplus(logits[gt==0])
    if not negatives.numel(): return logits.sum()*0
    k = min(negatives.numel(), max(32,4*int(target.sum())))
    return negatives.topk(k).values.mean()

def instance_ranking_loss(logits, target, distractors, margin=1.):
    """逐个同象限真实实例比较，消除重叠目标像素；不把union当成一个实例。"""
    pos = logits[target].mean()
    terms = [F.softplus(margin-pos+logits[m & ~target].mean()) for m in distractors if (m & ~target).any()]
    return torch.stack(terms).mean() if terms else logits.sum()*0

def losses(fusion, bev, embedding, wrong_embedding, cf_embedding, grid, gt, target, cls, distractors):
    correct = fusion.forward_with_details(bev,embedding)
    value = project_attention(correct['logits'],grid)[0,0]
    wrong = fusion.forward_with_details(bev,wrong_embedding)
    wrong_value = project_attention(wrong['logits'],grid)[0,0]
    balanced = balanced_instance_loss(value,gt,target,cls)
    contrast,gap = contrast_loss(value,wrong_value,target)
    hard = hard_negative_loss(value,gt,target)
    ranking = instance_ranking_loss(value,target,distractors)
    counterfactual = value.sum()*0
    if cf_embedding is not None:
        cf = fusion.forward_with_details(bev,cf_embedding)
        cf_value = project_attention(cf['logits'],grid)[0,0]
        counterfactual,_ = contrast_loss(value,cf_value,target)
    total=balanced + .5*contrast
    if fusion.flags['hard_negative']: total=total+.5*hard
    if fusion.flags['ranking']: total=total+.5*ranking+.5*counterfactual
    return total, dict(balanced=balanced,direction_contrast=contrast,hard_negative=hard,instance_ranking=ranking,
        instance_counterfactual=counterfactual,direction_logit_gap=gap),correct
