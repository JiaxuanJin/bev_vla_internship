"""第三轮等预算真实CUDA实验：冻结作者特征/双Head，训练语言空间适配器。"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.stage2.spatial_v3_data import OUT,V2,sha,write
from scripts.stage2.spatial_v3_model import InstanceReadout,InstanceFusion,VARIANTS,losses
from scripts.stage2.spatial_v2_experiment import V1,swap_direction
from scripts.stage2.spatial_experiment import initialize,INITIAL
from scripts.stage2.spatial_annotations import project_attention
from scripts.stage2.micro_training import seed_all,parameter_stats
from scripts.stage2.multitask_training import aggregate_detection_losses
from scripts.stage2.bev_coordinates import BEVGrid
from scripts.stage2.runtime_memory import release_memory
from scripts.stage2.spatial_report import confusion,summary

SELECTED=json.loads((V2/'train/selection.json').read_text())['selected']['checkpoint']
THRESHOLDS=(.1,.2,.3,.5,.7,.9)

def setup():
    model,wrapped,pipeline,dataset=initialize('train')
    pipeline.encoder=InstanceReadout(pipeline.encoder).cuda().eval()
    initial=torch.load(INITIAL,map_location='cpu',weights_only=False)['state_dict']
    v2=torch.load(SELECTED,map_location='cpu',weights_only=False)['state_dict']
    seed_all(20261028)
    pipeline.fusion=InstanceFusion(initial,v2,VARIANTS['control']).cuda().eval()
    manifest=json.loads((OUT/'annotations/manifest.json').read_text())
    cache=json.loads((V2/'cache/cache.json').read_text())['frames']
    features={}
    for r in cache:
        with np.load(V2/'cache'/r['cache_file']) as f: features[r['index']]=torch.tensor(f['bev'],device='cuda',dtype=torch.float32)
    return model,wrapped,pipeline,dataset,manifest,features

def encode(pipeline,rows):
    texts=sorted({r['text'] for r in rows}|{swap_direction(r['text']) for r in rows}|{r['counterfactual_text'] for r in rows if r['counterfactual_text']})
    values={}
    for i in range(0,len(texts),16):
        with torch.no_grad(): e=pipeline.encoder(texts[i:i+16],'cuda')
        values.update({t:v[None] for t,v in zip(texts[i:i+16],e)})
    return values

def labels(row):
    with np.load(OUT/'annotations'/row['frame_file']) as f:
        gt=torch.tensor(f['gt'],device='cuda',dtype=torch.long)
        masks=torch.tensor(f['masks'],device='cuda',dtype=torch.bool)
        xy=torch.tensor(f['ego_xy'],device='cuda')
        grid=torch.tensor(f['sampling_grid'],device='cuda')
        classes=torch.tensor(f['classes'],device='cuda')
    region=((xy[...,0]>0) if row['fb']=='front' else (xy[...,0]<0))&((xy[...,1]>0) if row['lr']=='left' else (xy[...,1]<0))
    target=masks[row['selected_index']] & region
    distractors=[masks[k]&region for k in row['same_quadrant_indices']]
    other=(masks[classes==row['target_class']].any(0)&~masks[row['selected_index']])
    return gt,target,grid,region,distractors,other

def head_forward(model,bev):
    """真实冻结BEV作为缓存边界；解码器/分割Head均真实CUDA算子和计算图。

    Shape-only mlvl_feats从不参与数值计算：get_bev_features临时返回已认证真实BEV。
    不把这个阶段记为图像主干训练或新的图像前向。
    """
    transformer=model.pts_bbox_head.transformer
    original=transformer.get_bev_features
    transformer.get_bev_features=lambda *a,**k: BEVGrid().to_sequence(bev)
    try:
        return model.pts_bbox_head([bev.new_empty((1,6,256,1,1))],[{}])
    finally:
        transformer.get_bev_features=original

def decode(model,pred):
    from mmdet3d.core.bbox import LiDARInstance3DBoxes
    boxes,scores,labels_=model.pts_bbox_head.get_bboxes(pred,[dict(box_type_3d=LiDARInstance3DBoxes)],rescale=False)[0]
    return dict(boxes=boxes.tensor.detach().cpu().numpy(),scores=scores.detach().cpu().numpy(),labels=labels_.detach().cpu().numpy())

def actual_losses(model,pred,ann):
    box=ann['gt_bboxes_3d'].to('cuda')
    cls=torch.tensor(ann['gt_labels_3d'],device='cuda',dtype=torch.long)
    author=model.pts_bbox_head.loss([box],[cls],pred)
    detection=aggregate_detection_losses({k:v for k,v in author.items() if 'seg' not in k})
    return detection

def norm_grad(loss, tensors):
    if not loss.requires_grad: return [0. for _ in tensors]
    grads=torch.autograd.grad(loss,tensors,retain_graph=True,allow_unused=True)
    return [float(g.detach().norm()) if g is not None else 0. for g in grads]

def attention_score(pipeline,rows,features,text):
    count={str(t):dict(tp=0,union=0,pred=0,background_fp=0,same_quadrant_fp=0,gt=0) for t in THRESHOLDS}
    gap=[];cf_gap=[];density=[];mass=[];perturb=[]
    with torch.no_grad():
        for r in rows:
            gt,target,grid,region,distractors,other=labels(r)
            pipeline.fusion.set_calibration(r); bev=features[r['index']]
            d=pipeline.fusion.forward_with_details(bev,text[r['text']])
            a=project_attention(d['attention'],grid)[0,0]
            b=pipeline.fusion.forward_with_details(bev,text[swap_direction(r['text'])])
            wrong=project_attention(b['attention'],grid)[0,0]
            gap.append(float((a[target]-wrong[target]).mean()))
            if r['counterfactual_text']:
                cf=pipeline.fusion.forward_with_details(bev,text[r['counterfactual_text']])
                ca=project_attention(cf['attention'],grid)[0,0]
                cf_gap.append(float((a[target]-ca[target]).mean()))
            if distractors:
                density.append(float(a[target].mean()-torch.stack([a[m].mean() for m in distractors]).max()))
            mass.append(float(a[other].sum()/a.sum().clamp_min(1e-9)))
            perturb.append(float((d['enhanced_bev']-bev).norm()/bev.norm()))
            same=other&region
            for t in THRESHOLDS:
                p=a>=t;c=count[str(t)]
                c['tp']+=int((p&target).sum());c['union']+=int((p|target).sum());c['pred']+=int(p.sum())
                c['gt']+=int(target.sum());c['background_fp']+=int((p&(gt==0)).sum());c['same_quadrant_fp']+=int((p&same).sum())
    for c in count.values(): c['iou']=c['tp']/c['union'] if c['union'] else None
    return dict(thresholds=count,n=len(rows),direction_probability_gap=float(np.mean(gap)),direction_wins=sum(g>0 for g in gap),
        instance_counterfactual_gap=float(np.mean(cf_gap)) if cf_gap else None,instance_counterfactual_wins=sum(g>0 for g in cf_gap),instance_counterfactual_count=len(cf_gap),
        target_minus_hardest_same_quadrant=float(np.mean(density)) if density else None,
        same_quadrant_wins=sum(g>0 for g in density),same_quadrant_count=len(density),
        other_instance_mass=float(np.mean(mass)),relative_feature_l2=float(np.mean(perturb)))

def diagnose():
    model,wrapped,pipeline,dataset,manifest,features=setup()
    rows=[r for r in manifest['annotations'] if r['split'] in ('train','dev')]
    text=encode(pipeline,rows)
    result={}
    for split in ('train','dev'):
        result[split]=attention_score(pipeline,[r for r in rows if r['split']==split],features,text)
    paired=next(r for r in rows if r['counterfactual_text'])
    ea=text[paired['text']];eb=text[paired['counterfactual_text']]
    result['qualifier_embedding_cosine']={name:float(F.cosine_similarity(ea[:,a:b],eb[:,a:b])) for name,a,b in (('pooled',0,768),('directions',768,2304),('selector',2304,3072))}
    r=next(r for r in rows if r['split']=='dev');gt,target,grid,region,distractors,other=labels(r)
    bev=features[r['index']]; pipeline.fusion.set_calibration(r)
    with torch.no_grad(): original=head_forward(model,bev)
    d=pipeline.fusion.forward_with_details(bev,text[r['text']]);pred=head_forward(model,d['enhanced_bev'])
    detection=actual_losses(model,pred,dataset.get_ann_info(r['index']))
    segmentation=F.cross_entropy(pred['seg_preds'],gt[None])
    parameters=[p for p in pipeline.fusion.parameters() if p.requires_grad]
    result['head_gradient_diagnosis']=dict(instruction_id=r['instruction_id'],detection_loss=float(detection.detach()),segmentation_loss=float(segmentation.detach()),
        gradient_tensors=['native_attention','fused_BEV']+[n for n,p in pipeline.fusion.named_parameters() if p.requires_grad],
        detection_gradient_l2=norm_grad(detection,[d['attention'],d['enhanced_bev']]+parameters),
        segmentation_gradient_l2=norm_grad(segmentation,[d['attention'],d['enhanced_bev']]+parameters),
        relative_bev_l2=float(((d['enhanced_bev']-bev).norm()/bev.norm()).detach()),alpha=float(pipeline.fusion.alpha),
        segmentation_logit_rms_delta=float((pred['seg_preds']-original['seg_preds']).square().mean().sqrt().detach()),
        segmentation_argmax_changed_pixels=int((pred['seg_preds'].argmax(1)!=original['seg_preds'].argmax(1)).sum()),
        detection_score_logit_rms_delta=float((pred['all_cls_scores']-original['all_cls_scores']).square().mean().sqrt().detach()),
        note='attention multiplicatively rescales existing feature channels; no new semantic channels. Heads and backbone frozen. Global GT losses are instruction-independent; nonzero gradient alone does not establish relevant task improvement.')
    write(OUT/'diagnosis.json',result)
    print('V3_DIAGNOSIS',json.dumps(result),flush=True)

def train():
    directory=OUT/'train';directory.mkdir(exist_ok=False)
    model,wrapped,pipeline,dataset,manifest,features=setup()
    train_rows=[r for r in manifest['annotations'] if r['split']=='train']
    dev=[r for r in manifest['annotations'] if r['split']=='dev']
    text=encode(pipeline,train_rows+dev)
    initial={k:v.detach().cpu().clone() for k,v in pipeline.fusion.state_dict().items()}
    warmup=160;joint=16;steps=warmup+joint
    schedule=np.random.RandomState(20261028).randint(0,len(train_rows),steps).tolist()
    protocol=dict(seed=20261028,warmup=warmup,joint=joint,variants=VARIANTS,schedule=[train_rows[n]['instruction_id'] for n in schedule],
        initialization=SELECTED,initialization_sha256=sha(SELECTED),alpha_fixed=float(pipeline.fusion.alpha),
        warmup_lr=.0005,joint_lr=.0003,loss_weights=dict(attention=.2,hard_negative=.5,ranking=.5,counterfactual=.5),
        selection='one terminal checkpoint per variant; maximize dev instance IoU at fixed0.5 among selector/ranking/hardneg/combined; threshold selected on dev only, always report0.5 too',
        stages='160 cached-real-BEV attention updates +16 cached-real-BEV frozen native dual-head three-loss updates; no image backbone backward',
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),ROOT/'scripts/stage2/spatial_v3_model.py',ROOT/'scripts/stage2/spatial_v3_data.py')})
    write(directory/'protocol.json',protocol)
    results={}
    for name,flags in VARIANTS.items():
        seed_all(20261028);pipeline.fusion.load_state_dict(initial);pipeline.fusion.flags=dict(flags);pipeline.fusion.eval()
        params=[p for p in pipeline.fusion.parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW(params,lr=.0005,weight_decay=.01)
        torch.cuda.reset_peak_memory_stats();started=time.monotonic()
        variant=directory/name;variant.mkdir()
        with (variant/'training.jsonl').open('x') as log:
            for step,n in enumerate(schedule,1):
                r=train_rows[n];gt,target,grid,region,distractors,other=labels(r)
                bev=features[r['index']];pipeline.fusion.set_calibration(r);optimizer.zero_grad(set_to_none=True)
                attention,parts,d=losses(pipeline.fusion,bev,text[r['text']],text[swap_direction(r['text'])],text.get(r['counterfactual_text']),grid,gt,target,r['target_class'],distractors)
                detection=attention*0;segmentation=attention*0;gradients=None
                if step>warmup:
                    for group in optimizer.param_groups: group['lr']=.0003
                    pred=head_forward(model,d['enhanced_bev'])
                    detection=actual_losses(model,pred,dataset.get_ann_info(r['index']))
                    segmentation=F.cross_entropy(pred['seg_preds'],gt[None])
                    total=detection+segmentation+.2*attention
                    if step==warmup+1:
                        gradients={k:norm_grad(v,params) for k,v in (('detection',detection),('segmentation',segmentation),('attention',attention))}
                else: total=attention
                total.backward();gn=float(torch.nn.utils.clip_grad_norm_(params,5.,error_if_nonfinite=True))
                before={k:p.detach().cpu().clone() for k,p in pipeline.fusion.named_parameters()}
                optimizer.step();delta=parameter_stats(pipeline.fusion,before)
                record=dict(step=step,stage='cached_real_BEV_attention' if step<=warmup else 'cached_real_BEV_native_dual_head',
                    instruction_id=r['instruction_id'],frame_index=r['index'],optimizer_step_executed=True,
                    total_loss=float(total.detach()),attention_loss=float(attention.detach()),detection_loss=float(detection.detach()) if step>warmup else None,
                    segmentation_loss=float(segmentation.detach()) if step>warmup else None,gradient_l2=gn,task_parameter_gradients=gradients,
                    loss_parts={k:float(v.detach()) for k,v in parts.items()},parameter_updates=delta,
                    alpha=float(pipeline.fusion.alpha),peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
                assert any(v['changed_elements'] for v in delta.values())
                log.write(json.dumps(record)+'\n');log.flush()
                del total,attention,parts,d,before
                if step>warmup:del pred
                optimizer.zero_grad(set_to_none=True)
                if step%40==0: print('V3_UPDATE',name,step,record['attention_loss'],flush=True)
        checkpoint=variant/'fusion_step176.pt'
        with checkpoint.open('xb') as f: torch.save(dict(state_dict=pipeline.fusion.state_dict(),optimizer=optimizer.state_dict(),steps=steps,flags=flags,protocol_sha256=sha(directory/'protocol.json')),f)
        score=attention_score(pipeline,dev,features,text)
        results[name]=dict(checkpoint=str(checkpoint),sha256=sha(checkpoint),steps=steps,warmup_steps=warmup,joint_steps=joint,dev=score,
            seconds=time.monotonic()-started,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
        write(variant/'summary.json',results[name]);print('V3_DEV',name,json.dumps(score),flush=True)
        release_memory()
    eligible=('selector','ranking','hardneg','combined')
    chosen=max(eligible,key=lambda n:results[n]['dev']['thresholds']['0.5']['iou'])
    threshold=max(THRESHOLDS,key=lambda t:results[chosen]['dev']['thresholds'][str(t)]['iou'])
    write(directory/'selection.json',dict(variants=results,selected_variant=chosen,selected=results[chosen],selected_threshold=threshold,
        selection_scope='training/dev only; all checkpoints sealed before new final features or predictions are computed'))
    print('V3_SELECTED',chosen,threshold,flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--phase',choices=('diagnose','train'),required=True)
    args=parser.parse_args()
    if args.phase=='diagnose':diagnose()
    else:train()

if __name__=='__main__':main()
