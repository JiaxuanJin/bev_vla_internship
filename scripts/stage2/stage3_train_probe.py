"""由训练侧梯度证据驱动的等预算32步GPU优化；无最终集调用。"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.stage2.stage3_metric_review import OUT,V3,write,sha,aggregate_predictions
from scripts.stage2.stage3_bottleneck import prepare,filtered_detection,gradient_summary
from scripts.stage2.spatial_v3_experiment import labels,head_forward
from scripts.stage2.spatial_v3_model import losses,VARIANTS
from scripts.stage2.spatial_v2_experiment import swap_direction
from scripts.stage2.micro_training import seed_all,parameter_stats
from scripts.stage2.spatial_annotations import project_attention

VARIANTS_PROBE=('joint_control','gradient_balanced','local_balanced')

def train():
    directory=OUT/'optimization';directory.mkdir(exist_ok=False)
    diagnosis=json.loads((OUT/'gpu_diagnosis/diagnosis.json').read_text())
    weight=diagnosis['proposed_attention_weight']
    model,wrapped,pipeline,dataset,rows,features,text,v2=prepare()
    train_rows=[r for r in rows if r['split']=='train'];dev=[r for r in rows if r['split']=='dev']
    order=np.random.RandomState(20261029).randint(0,len(train_rows),32).tolist()
    protocol=dict(seed=20261029,steps_per_branch=32,branches=VARIANTS_PROBE,
        schedule=[train_rows[n]['instruction_id'] for n in order],initial_checkpoint=json.loads((V3/'train/selection.json').read_text())['selected'],
        attention_weights={'joint_control':.2,'gradient_balanced':weight,'local_balanced':weight},
        target_neighborhood_ce_weights={'joint_control':0.,'gradient_balanced':0.,'local_balanced':1.},
        common=dict(learning_rate=.0003,weight_decay=.01,gradient_clip=5.,alpha_fixed=float(pipeline.fusion.alpha),optimizer='fresh AdamW',backbone_head_bert_frozen=True,channel_residual_enabled=False),
        rationale='train-gradient evidence only: median perception/attention norm; fixed2m neighborhood has stronger semantic gradient. Single-variable comparisons, no blind step extension.',
        selection='terminal step32 only; highest pooled dev target_neighborhood_semantic_IoU among all three; always report irrelevant/objectAP tradeoffs; final/generalization BLOCKED',
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),ROOT/'scripts/stage2/stage3_bottleneck.py',ROOT/'scripts/stage2/stage3_adapter.py',ROOT/'scripts/stage2/stage3_metric_review.py')})
    write(directory/'protocol.json',protocol)
    initial={k:v.detach().cpu().clone() for k,v in pipeline.fusion.state_dict().items()}
    parameters=[p for p in pipeline.fusion.parameters() if p.requires_grad]
    results={}
    for name in VARIANTS_PROBE:
        pipeline.fusion.load_state_dict(initial);pipeline.fusion.core.flags=dict(VARIANTS['combined']);pipeline.fusion.channel_enabled=False;pipeline.fusion.eval();seed_all(20261029)
        optimizer=torch.optim.AdamW(parameters,lr=.0003,weight_decay=.01)
        folder=directory/name;folder.mkdir();checkpoints=[];torch.cuda.reset_peak_memory_stats()
        with (folder/'training.jsonl').open('x') as log:
            for step,n in enumerate(order,1):
                r=train_rows[n];gt,target,grid,region,distractors,other=labels(r);bev=features[r['index']]
                with np.load(OUT/'labels'/(r['instruction_id']+'.npz')) as f:roi=torch.tensor(f['target_neighborhood'],device='cuda')
                pipeline.fusion.set_calibration(r);optimizer.zero_grad(set_to_none=True)
                attention,parts,d=losses(pipeline.fusion,bev,text[r['text']],text[swap_direction(r['text'])],text.get(r['counterfactual_text']),grid,gt,target,r['target_class'],distractors)
                pred=head_forward(model,d['enhanced_bev']);detection,stats=filtered_detection(model,pred,dataset.get_ann_info(r['index']))
                per_pixel=F.cross_entropy(pred['seg_preds'],gt[None],reduction='none')[0]
                segmentation=per_pixel.mean();local=per_pixel[roi].mean()
                terms=dict(detection=detection,segmentation=segmentation,attention=attention,target_neighborhood_ce=local)
                weights=dict(detection=1.,segmentation=1.,attention=protocol['attention_weights'][name],target_neighborhood_ce=protocol['target_neighborhood_ce_weights'][name])
                grad=gradient_summary(terms,parameters,weights)
                total=sum(terms[k]*weights[k] for k in terms)
                total.backward();gn=float(torch.nn.utils.clip_grad_norm_(parameters,5.,error_if_nonfinite=True))
                before={k:p.detach().cpu().clone() for k,p in pipeline.fusion.named_parameters()};optimizer.step()
                delta=parameter_stats(pipeline.fusion,before)
                assert any(v['changed_elements'] for v in delta.values())
                record=dict(step=step,instruction_id=r['instruction_id'],sample_token=r['sample_token'],frame_index=r['index'],split='train',
                    optimizer_step_executed=True,total_loss=float(total.detach()),losses={k:float(v.detach()) for k,v in terms.items()},
                    task_gradients=grad,clip_gradient_l2=gn,parameter_updates=delta,filtered_gt=stats,
                    relative_bev_l2=float(((d['enhanced_bev']-bev).norm()/bev.norm()).detach()),alpha=float(pipeline.fusion.alpha),
                    peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
                log.write(json.dumps(record)+'\n');log.flush()
                if step in (1,32):
                    checkpoint=folder/f'fusion_step{step:03d}.pt'
                    with checkpoint.open('xb') as f:torch.save(dict(state_dict=pipeline.fusion.state_dict(),optimizer=optimizer.state_dict(),steps=step,channel_enabled=False,flags=pipeline.fusion.flags,protocol_sha256=sha(directory/'protocol.json')),f)
                    checkpoints.append(dict(checkpoint=str(checkpoint),sha256=sha(checkpoint),step=step))
                if step%8==0:print('STAGE3_REAL_OPTIMIZER',name,step,record['losses'],flush=True)
                del total,terms,pred,d,attention,parts,before,detection,segmentation,local,per_pixel
                optimizer.zero_grad(set_to_none=True)
        results[name]=dict(checkpoint=checkpoints[-1]['checkpoint'],sha256=checkpoints[-1]['sha256'],checkpoints=checkpoints,optimizer_updates=32,
            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
    write(directory/'checkpoints.json',results)
    # Evaluate all terminal branches identically, no checkpoint/radius/threshold sweep.
    pred_dir=OUT/'dev_new_predictions';pred_dir.mkdir()
    meta=json.loads((V3/'dev_map/predictions.json').read_text());records=[]
    loaded={name:torch.load(r['checkpoint'],map_location='cpu',weights_only=False)['state_dict'] for name,r in results.items()}
    for n,r in enumerate(dev):
        source=next(x for x in meta['annotations'] if x['instruction_id']==r['instruction_id'])
        with np.load(V3/'dev_map'/source['prediction_file']) as f:payload={k:f[k] for k in ('gt','target','region','original','v2','v2_attention','combined','combined_attention')}
        gt,target,grid,region,distractors,other=labels(r);bev=features[r['index']];pipeline.fusion.set_calibration(r)
        with torch.no_grad():
            for name,state in loaded.items():
                pipeline.fusion.load_state_dict(state);d=pipeline.fusion.forward_with_details(bev,text[r['text']]);pred=head_forward(model,d['enhanced_bev'])
                payload[name]=pred['seg_preds'].argmax(1)[0].cpu().numpy();payload[name+'_attention']=project_attention(d['attention'],grid)[0,0].cpu().numpy()
        filename=f'instruction_{n:03d}.npz';np.savez_compressed(pred_dir/filename,**payload);records.append({**r,'prediction_file':filename})
    write(pred_dir/'predictions.json',dict(annotations=records,checkpoints=results,scope='dev only; terminal32; no final predictions'))
    metrics=aggregate_predictions(dict(annotations=records),pred_dir,OUT/'dev_new_metrics',('original','v2','combined',*VARIANTS_PROBE))
    selected=max(VARIANTS_PROBE,key=lambda n:metrics[n]['target_neighborhood_semantic_IoU']['iou'])
    write(directory/'selection.json',dict(selected_branch=selected,selected=results[selected],all_branches=results,
        n_optimizer_updates=96,selection_rule=protocol['selection'],final_status='BLOCKED: no fresh independent heldout opened',
        dev_metrics={name:metrics[name] for name in VARIANTS_PROBE}))
    print('STAGE3_DEV_SELECTED',selected,flush=True)
    for name in ('combined',*VARIANTS_PROBE):
        m=metrics[name];print('STAGE3_DEV_METRICS',name,m['target_neighborhood_semantic_IoU']['iou'],m['instruction_irrelevant_semantic_mIoU']['foreground_miou'],m['counterfactual'],flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.parse_args();train()
