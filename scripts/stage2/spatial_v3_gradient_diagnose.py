"""训练侧事后梯度冲突诊断；无参数更新，不读取最终集数值。"""
from pathlib import Path
import sys,json
import numpy as np
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.stage2.spatial_v3_data import OUT,write
from scripts.stage2.spatial_v3_experiment import setup,head_forward,actual_losses
from scripts.stage2.spatial_v3_model import VARIANTS
from scripts.stage2.spatial_v3_objects import native_box_mask,relevant_query_weights

def flattened_grad(value,parameters):
    gs=torch.autograd.grad(value,parameters,retain_graph=True,allow_unused=True)
    return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1).detach() for g,p in zip(gs,parameters)])

def main():
    model,wrapped,pipeline,dataset,manifest,features=setup()
    protocol=json.loads((OUT/'objects_train/protocol.json').read_text())
    row=next(r for r in protocol['annotations'] if r['split']=='train' and r['relevant_gt_indices'])
    with torch.no_grad():embedding=pipeline.encoder([row['text']],'cuda')
    result={}
    states=json.loads((OUT/'objects_train/summary.json').read_text())
    for branch,record in states.items():
        saved=torch.load(record['checkpoint'],map_location='cpu',weights_only=False)
        pipeline.fusion.load_state_dict(saved['state_dict']);pipeline.fusion.flags=dict(saved['flags'])
        pipeline.fusion.set_calibration(row);bev=features[row['index']]
        d=pipeline.fusion.forward_with_details(bev,embedding);pred=head_forward(model,d['enhanced_bev'])
        ann=dataset.get_ann_info(row['index']);trace=[]
        global_det=actual_losses(model,pred,ann)
        with relevant_query_weights(model.pts_bbox_head,row,trace):weighted_det=actual_losses(model,pred,ann)
        with np.load(OUT/'annotations'/row['frame_file']) as f:gt=torch.tensor(f['gt'],device='cuda',dtype=torch.long)
        seg=F.cross_entropy(pred['seg_preds'],gt[None])
        target=native_box_mask(ann['gt_bboxes_3d'].tensor.numpy(),row['relevant_gt_indices'],'cuda')
        logits=d['logits'][0,0];positive=F.softplus(-logits[target]).mean()
        negatives=F.softplus(logits[~target]);k=min(len(negatives),max(64,4*int(target.sum())))
        attention=.5*positive+.5*negatives.topk(k).values.mean()
        parameters=[p for p in pipeline.fusion.parameters() if p.requires_grad]
        grads={name:flattened_grad(value,parameters) for name,value in (('global_detection',global_det),('relevant_weighted_detection',weighted_det),('segmentation',seg),('attention',attention))}
        result[branch]=dict(parameter_gradient_l2={name:float(v.norm()) for name,v in grads.items()},
            gradient_cosines={name:float(F.cosine_similarity(v[None],grads['attention'][None])) for name,v in grads.items() if name!='attention'},
            attention_to_weighted_detection_ratio=float(.2*grads['attention'].norm()/grads['relevant_weighted_detection'].norm()),
            relative_bev_l2=float(((d['enhanced_bev']-bev).norm()/bev.norm()).detach()),
            mean_target_attention=float(d['attention'][0,0][target].mean().detach()),
            real_target_native_pixels=int(target.sum()),query_weight_trace=trace)
    write(OUT/'train_gradient_conflict.json',dict(instruction_id=row['instruction_id'],split='train',results=result,
        limitation='one fixed first positive training instruction; local gradient evidence, not global causality; no optimizer.step or final metrics used'))
    print('V3_TRAIN_GRADIENT_DIAGNOSIS',json.dumps(result),flush=True)

if __name__=='__main__':main()
