"""第三阶段真实CUDA梯度与感知响应诊断；训练/开发数据限定。"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import sys
import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.stage2.stage3_metric_review import OUT,V3,sha,write
from scripts.stage2.spatial_v3_experiment import setup,encode,labels,head_forward,actual_losses
from scripts.stage2.spatial_v3_model import VARIANTS,losses
from scripts.stage2.spatial_v3_range_correction import filter_ann
from scripts.stage2.micro_training import seed_all
from scripts.stage2.stage3_adapter import PerceptionAdapter

def prepare():
    model,wrapped,pipeline,dataset,manifest,features=setup()
    v2={k:v.detach().cpu().clone() for k,v in pipeline.fusion.state_dict().items()}
    chosen=json.loads((V3/'train/selection.json').read_text())['selected']
    assert sha(chosen['checkpoint'])==chosen['sha256']
    saved=torch.load(chosen['checkpoint'],map_location='cpu',weights_only=False)
    pipeline.fusion.load_state_dict(saved['state_dict']);pipeline.fusion.flags=dict(saved['flags'])
    seed_all(20261029);pipeline.fusion=PerceptionAdapter(pipeline.fusion).cuda().eval()
    rows=[r for r in manifest['annotations'] if r['split'] in ('train','dev')]
    text=encode(pipeline,rows)
    return model,wrapped,pipeline,dataset,rows,features,text,v2

def filtered_detection(model,pred,ann):
    filtered,stats=filter_ann(ann,model.pts_bbox_head.pc_range)
    return actual_losses(model,pred,filtered),stats

def flat_grad(value,parameters):
    gs=torch.autograd.grad(value,parameters,retain_graph=True,allow_unused=True)
    return torch.cat([(g.detach() if g is not None else torch.zeros_like(p)).reshape(-1) for g,p in zip(gs,parameters)])

def gradient_summary(terms,params,weights):
    grads={k:flat_grad(v,params) for k,v in terms.items()}
    names=list(grads)
    result=dict(unweighted_l2={k:float(v.norm()) for k,v in grads.items()},
        weighted_l2={k:float(v.norm())*weights[k] for k,v in grads.items()},weights=weights,
        cosine={f'{a}__{b}':float(F.cosine_similarity(grads[a][None],grads[b][None])) if grads[a].norm()>0 and grads[b].norm()>0 else None for i,a in enumerate(names) for b in names[i+1:]})
    perception=grads['detection']+grads['segmentation']
    result['perception_l2']=float(perception.norm())
    result['suggested_attention_weight']=float(perception.norm()/grads['attention'].norm()) if grads['attention'].norm()>0 else None
    return result

def verify_filters(dataset,frames,model):
    from mmdet3d.datasets.pipelines import ObjectRangeFilter,ObjectNameFilter
    proof=[]
    for row in frames:
        if row['split'] not in ('train','dev'):continue
        ann=dataset.get_ann_info(row['index']);corrected,stats=filter_ann(ann,model.pts_bbox_head.pc_range)
        author=ObjectNameFilter(list(model.CLASSES))(ObjectRangeFilter(model.pts_bbox_head.pc_range)(dict(gt_bboxes_3d=ann['gt_bboxes_3d'].clone(),gt_labels_3d=ann['gt_labels_3d'].copy())))
        np.testing.assert_array_equal(corrected['gt_bboxes_3d'].tensor.numpy(),author['gt_bboxes_3d'].tensor.numpy())
        np.testing.assert_array_equal(corrected['gt_labels_3d'],author['gt_labels_3d'])
        proof.append(dict(index=row['index'],sample_token=row['sample_token'],split=row['split'],**stats))
    return dict(status='PASS',n_frames=len(proof),frames=proof,definition='exact author ObjectRangeFilter+ObjectNameFilter+limit_yaw on actual dataset annotations')

def diagnose():
    directory=OUT/'gpu_diagnosis';directory.mkdir(exist_ok=False)
    model,wrapped,pipeline,dataset,rows,features,text,v2=prepare()
    manifest=json.loads((V3/'annotations/manifest.json').read_text())
    write(directory/'gt_filter_equivalence.json',verify_filters(dataset,manifest['frames'],model))
    # Fixed metadata-only probe list: first paired training instruction per scene,
    # first paired development instruction per frame; fallback to first unique.
    probes=[]
    for split,key in (('train','scene_token'),('dev','sample_token')):
        values=sorted({r[key] for r in rows if r['split']==split})
        for v in values:
            candidates=[r for r in rows if r['split']==split and r[key]==v]
            probes.append(next((r for r in candidates if r['counterfactual_text']),candidates[0]))
    write(directory/'probe_protocol.json',dict(annotations=probes,selection='metadata-only first paired per train scene / dev frame; no metric-based selection',
        branches=['no_fusion','v2_frozen','v3_frozen','channel_zero_init'],optimizer_updates=0,alpha=float(pipeline.fusion.alpha)))
    original={k:v.detach().cpu().clone() for k,v in pipeline.fusion.state_dict().items()}
    records=[]
    params=[p for p in pipeline.fusion.parameters() if p.requires_grad]
    for row in probes:
        gt,target,grid,region,distractors,other=labels(row);bev=features[row['index']]
        with np.load(OUT/'labels'/(row['instruction_id']+'.npz')) as f:roi=torch.tensor(f['target_neighborhood'],device='cuda')
        with torch.no_grad():baseline=head_forward(model,bev)
        for branch in ('no_fusion','v2_frozen','v3_frozen','channel_zero_init'):
            pipeline.fusion.load_state_dict(original);pipeline.fusion.channel_enabled=branch=='channel_zero_init'
            pipeline.fusion.core.flags=dict(VARIANTS['combined'])
            if branch=='v2_frozen':pipeline.fusion.core.load_state_dict(v2);pipeline.fusion.core.flags=dict(VARIANTS['control'])
            if branch=='no_fusion':pipeline.fusion.core.flags['fusion']=False
            pipeline.fusion.set_calibration(row)
            # LR direction contrast uses actual wrong-direction text, not instance CF.
            from scripts.stage2.spatial_v2_experiment import swap_direction
            attention,parts,d=losses(pipeline.fusion,bev,text[row['text']],text[swap_direction(row['text'])],text.get(row['counterfactual_text']),grid,gt,target,row['target_class'],distractors)
            enhanced=d['enhanced_bev']
            if not enhanced.requires_grad:enhanced=enhanced.detach().requires_grad_(True)
            pred=head_forward(model,enhanced)
            detection,stats=filtered_detection(model,pred,dataset.get_ann_info(row['index']))
            segmentation=F.cross_entropy(pred['seg_preds'],gt[None])
            local_ce=F.cross_entropy(pred['seg_preds'],gt[None],reduction='none')[0][roi].mean()
            terms=dict(detection=detection,segmentation=segmentation,attention=attention,target_neighborhood_ce=local_ce)
            grad=gradient_summary(terms,params,dict(detection=1.,segmentation=1.,attention=.2,target_neighborhood_ce=0.))
            correct=pred['seg_preds'].detach()
            cf_stats=None
            if row['counterfactual_text']:
                with torch.no_grad():
                    cd=pipeline.fusion.forward_with_details(bev,text[row['counterfactual_text']]);cp=head_forward(model,cd['enhanced_bev'])
                    cf_stats=dict(target_neighborhood_logit_rms_delta=float((correct-cp['seg_preds'])[:,:,roi].square().mean().sqrt()),
                        target_neighborhood_changed_pixels=int(((correct.argmax(1)[0]!=cp['seg_preds'].argmax(1)[0])&roi).sum()))
            record=dict(instruction_id=row['instruction_id'],split=row['split'],branch=branch,
                losses={k:float(v.detach()) for k,v in terms.items()},gradient=grad,filtered_gt=stats,
                alpha=float(pipeline.fusion.alpha),relative_bev_l2=float(((d['enhanced_bev']-bev).norm()/bev.norm()).detach()),
                seg_logits_rms_delta_vs_no_fusion=float((correct-baseline['seg_preds']).square().mean().sqrt()),
                detection_logits_rms_delta_vs_no_fusion=float((pred['all_cls_scores']-baseline['all_cls_scores']).square().mean().sqrt().detach()),
                changed_pixels_vs_no_fusion=int((correct.argmax(1)!=baseline['seg_preds'].argmax(1)).sum()),
                target_neighborhood_changed_pixels_vs_no_fusion=int(((correct.argmax(1)[0]!=baseline['seg_preds'].argmax(1)[0])&roi).sum()),
                counterfactual=cf_stats,optimizer_updates=0,parameter_delta_l2=0.,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
            records.append(record);print('STAGE3_GRADIENT',row['split'],branch,grad['unweighted_l2'],flush=True)
            del pred,terms,d,attention,enhanced,correct,parts,detection,segmentation,local_ce
        # baseline is a detached graph and can be released after this frame.
    train=[r for r in records if r['split']=='train' and r['branch']=='v3_frozen']
    coefficient=min(.2,float(np.median([r['gradient']['suggested_attention_weight'] for r in train])))
    write(directory/'diagnosis.json',dict(status='PASS',records=records,n_train_probes=len(train),n_dev_probes=len(probes)-len(train),
        optimizer_updates=0,proposed_attention_weight=coefficient,
        proposal='fixed median train ||g_detection+g_segmentation||/||g_attention|| capped0.2; no dev/final used for numerical weight choice',
        evidence='compare native channel residual zero-init task gradients vs scalar fusion; exact head paths, fixed inherited alpha'))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.parse_args();diagnose()

if __name__=='__main__':main()
