"""独立车辆/行人语言训练；真实Hungarian正例加权，非官方AP@2m_xy。"""
from __future__ import annotations
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.stage2.spatial_v3_data import OUT,sha,write
from scripts.stage2.spatial_v3_experiment import setup,head_forward,actual_losses,norm_grad
from scripts.stage2.spatial_v3_model import VARIANTS
from scripts.stage2.spatial_detection import parse_object_instruction,ego_centers,in_region
from scripts.stage2.bev_coordinates import BEVGrid
from scripts.stage2.micro_training import parameter_stats,seed_all

def object_rows(frames,dataset,final=False):
    from pyquaternion import Quaternion
    raw=json.loads((OUT/'object_records.json').read_text())
    rows=[]
    for f in frames:
        if (f['split']=='final') != final: continue
        ann=dataset.get_ann_info(f['dataset_index'] if final else f['index'])
        boxes=ann['gt_bboxes_3d'].tensor.numpy();labels=ann['gt_labels_3d'];ego=ego_centers(boxes,f)
        world=ego@Quaternion(f['ego2global_rotation']).rotation_matrix.T+np.asarray(f['ego2global_translation'])
        records=raw[f['sample_token']]
        dd=np.linalg.norm(world[:,None]-np.asarray([r['translation'] for r in records])[None],axis=-1)
        if len(boxes):
            nearest=dd.argmin(1);error=float(dd.min(1).max())
            if error>.001 or len(set(nearest))!=len(nearest):raise ValueError('unreliable real object correspondence')
        else: nearest=[];error=0.
        for name in ('cars','pedestrians'):
            for fb,lr in (('front','left'),('front','right'),('rear','left'),('rear','right')):
                text=f'Find {name} on the {fb} {lr}';prompt=parse_object_instruction(text)
                relevant=(labels==prompt['class_id'])&in_region(ego,prompt)
                irrelevant=(labels==prompt['class_id'])&~in_region(ego,prompt)
                rows.append({**f,'instruction_id':f"{f['sample_token']}_{name}_{fb}_{lr}",'text':text,'prompt':prompt,
                    'gt_annotation_ids':[records[i]['token'] for i in nearest],
                    'gt_instance_ids':[records[i]['instance_token'] for i in nearest],
                    'relevant_gt_indices':np.flatnonzero(relevant).tolist(),'irrelevant_gt_indices':np.flatnonzero(irrelevant).tolist(),
                    'geometry_max_error_m':error})
    return rows

def native_box_mask(boxes,selected,device):
    """真实3D box的LiDAR地面footprint；小于BEV单元时保留包含中心的单元。"""
    xy=BEVGrid().lidar_centers(device=device,dtype=torch.float32)[...,:2]
    target=torch.zeros((150,150),device=device,dtype=torch.bool)
    for k in selected:
        b=torch.as_tensor(boxes[k],device=device)
        d=xy-b[:2];c=b[6].cos();s=b[6].sin()
        u=d[...,0]*c+d[...,1]*s;v=-d[...,0]*s+d[...,1]*c
        mask=(u.abs()<=b[3]/2)&(v.abs()<=b[4]/2)
        if not mask.any() and (b[:2].abs()<51.2).all():
            mask.reshape(-1)[d.square().sum(-1).argmin()]=True
        target|=mask
    return target

@contextmanager
def relevant_query_weights(head,row,trace):
    """作者Hungarian分配保持不变；仅匹配到真实相关GT的分类/回归权重×3。

    所有无关GT仍保留权重1，所有负query保留权重1，不把无关对象当背景。
    """
    from pyquaternion import Quaternion
    original=head._get_target_single
    rotation=None;shift=None
    def weighted(*args,**kwargs):
        nonlocal rotation,shift
        result=list(original(*args,**kwargs))
        labels,lw,bt,bw,pos,neg=result
        if rotation is None:
            rotation=torch.tensor(Quaternion(row['lidar2ego_rotation']).rotation_matrix,device=bt.device,dtype=bt.dtype)
            shift=bt.new_tensor(row['lidar2ego_translation'])
        center=bt[pos,:3]@rotation.T+shift
        p=row['prompt'];inside=(center[:,0]>0) if p['front_back']=='front' else (center[:,0]<0)
        inside &= (center[:,1]>0) if p['left_right']=='left' else (center[:,1]<0)
        selected=pos[inside&(labels[pos]==p['class_id'])]
        lw=lw.clone();bw=bw.clone();lw[selected]*=3.;bw[selected]*=3.
        trace.append(dict(positive_queries=int(len(pos)),relevant_positive_queries=int(len(selected)),negative_queries=int(len(neg))))
        result[1]=lw;result[3]=bw
        return tuple(result)
    head._get_target_single=weighted
    try:yield
    finally:del head._get_target_single

def train():
    directory=OUT/'objects_train';directory.mkdir(exist_ok=False)
    model,wrapped,pipeline,dataset,manifest,features=setup()
    frames=[f for f in manifest['frames'] if f['split'] in ('train','dev')]
    rows=object_rows(frames,dataset)
    write(directory/'protocol.json',dict(annotations=rows,steps_per_branch=64,branches=['global_control','object_supervised'],
        initialization='same frozen V2step220; fresh AdamW; fixed alpha',lr=.0003,seed=20261028,
        loss='control: global native detection + global segmentation; supervised: native detection with relevant matched positive query weights3 and all other weights1 +global segmentation +0.2 native real-box Attention BCE/hardneg',
        selector='plural car/pedestrian set in fixedEgo quadrant; annotation+instance token correspondences <=1mm; no map-to-object mapping',
        evaluation='noninterpolated AP@2m_xy, full native top300, class and fixed quadrant partitions, all confidence scores; NOT nuScenes official AP',
        no_dev_selection='fixed terminal step64; no early stopping or final-set selection'))
    train_rows=[r for r in rows if r['split']=='train']
    texts=sorted({r['text'] for r in rows});encoded={}
    for n in range(0,len(texts),16):
        with torch.no_grad():ee=pipeline.encoder(texts[n:n+16],'cuda')
        encoded.update({t:e[None] for t,e in zip(texts[n:n+16],ee)})
    initial={k:v.detach().cpu().clone() for k,v in pipeline.fusion.state_dict().items()}
    pipeline.fusion.flags=dict(VARIANTS['selector'])
    schedule=np.random.RandomState(20261028).randint(0,len(train_rows),64).tolist()
    summaries={}
    for branch in ('global_control','object_supervised'):
        seed_all(20261028);pipeline.fusion.load_state_dict(initial);pipeline.fusion.eval()
        parameters=[p for p in pipeline.fusion.parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW(parameters,lr=.0003,weight_decay=.01)
        torch.cuda.reset_peak_memory_stats();traces=[]
        branchdir=directory/branch;branchdir.mkdir()
        with (branchdir/'training.jsonl').open('x') as log:
            for step,n in enumerate(schedule,1):
                r=train_rows[n];ann=dataset.get_ann_info(r['index']);bev=features[r['index']]
                pipeline.fusion.set_calibration(r);optimizer.zero_grad(set_to_none=True)
                d=pipeline.fusion.forward_with_details(bev,encoded[r['text']]);pred=head_forward(model,d['enhanced_bev'])
                trace=[]
                if branch=='object_supervised':
                    with relevant_query_weights(model.pts_bbox_head,r,trace): detection=actual_losses(model,pred,ann)
                else:detection=actual_losses(model,pred,ann)
                with np.load(OUT/'annotations'/r['frame_file']) as f:gt=torch.tensor(f['gt'],device='cuda',dtype=torch.long)
                segmentation=F.cross_entropy(pred['seg_preds'],gt[None])
                target=native_box_mask(ann['gt_bboxes_3d'].tensor.numpy(),r['relevant_gt_indices'],'cuda')
                logits=d['logits'][0,0]
                positive=F.softplus(-logits[target]).mean() if target.any() else logits.sum()*0
                negative=F.softplus(logits[~target]);k=min(len(negative),max(64,4*int(target.sum())))
                attention=.5*positive+.5*negative.topk(k).values.mean()
                total=detection+segmentation+(.2*attention if branch=='object_supervised' else 0.)
                task_grads={name:norm_grad(value,parameters) for name,value in (('detection',detection),('segmentation',segmentation),('attention',attention))} if step==1 else None
                total.backward();gn=float(torch.nn.utils.clip_grad_norm_(parameters,5.,error_if_nonfinite=True))
                before={k:p.detach().cpu().clone() for k,p in pipeline.fusion.named_parameters()}
                optimizer.step();delta=parameter_stats(pipeline.fusion,before)
                assert any(v['changed_elements'] for v in delta.values())
                record=dict(step=step,instruction_id=r['instruction_id'],optimizer_step_executed=True,total_loss=float(total.detach()),
                    detection_loss=float(detection.detach()),segmentation_loss=float(segmentation.detach()),attention_loss=float(attention.detach()),
                    attention_loss_in_objective=branch=='object_supervised',task_parameter_gradients=task_grads,gradient_l2=gn,
                    relevant_gt_count=len(r['relevant_gt_indices']),native_target_pixels=int(target.sum()),query_weight_trace=trace,
                    parameter_updates=delta,alpha=float(pipeline.fusion.alpha),peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
                log.write(json.dumps(record)+'\n');log.flush();traces.extend(trace)
                if step%16==0:print('V3_OBJECT_UPDATE',branch,step,record['detection_loss'],flush=True)
        checkpoint=branchdir/'fusion_step064.pt'
        with checkpoint.open('xb') as f:torch.save(dict(state_dict=pipeline.fusion.state_dict(),optimizer=optimizer.state_dict(),steps=64,flags=pipeline.fusion.flags,protocol_sha256=sha(directory/'protocol.json')),f)
        summaries[branch]=dict(checkpoint=str(checkpoint),sha256=sha(checkpoint),steps=64,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
            weighted_positive_assignments=sum(t['relevant_positive_queries'] for t in traces),actual_object_supervised_steps=64 if branch=='object_supervised' else 0)
        write(branchdir/'summary.json',summaries[branch])
    write(directory/'summary.json',summaries)

if __name__=='__main__':train()
