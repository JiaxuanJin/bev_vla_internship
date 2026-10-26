"""验证第三轮优化方向与重叠/空实例边界，不运行实验或读取留出集。"""
import torch
from scripts.stage2.spatial_v3_model import hard_negative_loss, instance_ranking_loss


def test_ranking_pushes_selected_instance_up_and_distractor_down():
    logits=torch.zeros((2,3),requires_grad=True)
    target=torch.tensor([[True,False,False],[False,False,False]])
    distractor=torch.tensor([[False,True,False],[False,False,False]])
    loss=instance_ranking_loss(logits,target,[distractor]);loss.backward()
    assert logits.grad[target].item()<0
    assert logits.grad[distractor].item()>0
    assert logits.grad[1].abs().sum()==0


def test_ranking_excludes_shared_pixels_and_handles_no_distinct_distractor():
    logits=torch.zeros((1,3),requires_grad=True)
    target=torch.tensor([[True,False,False]])
    shared=torch.tensor([[True,True,False]])
    instance_ranking_loss(logits,target,[shared]).backward()
    assert logits.grad[0,0]<0 and logits.grad[0,1]>0
    assert instance_ranking_loss(logits,target,[target]).item()==0


def test_hard_negative_uses_highest_background_errors_without_target_penalty():
    logits=torch.arange(100,dtype=torch.float32).reshape(10,10).requires_grad_()
    gt=torch.zeros_like(logits,dtype=torch.long)
    target=torch.zeros_like(logits,dtype=torch.bool);target[9,9]=True;gt[target]=1
    hard_negative_loss(logits,gt,target).backward()
    assert logits.grad[target].sum()==0
    assert torch.count_nonzero(logits.grad)==32
    assert logits.grad[0,0]==0 and logits.grad[9,8]>0
