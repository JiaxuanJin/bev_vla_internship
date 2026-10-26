"""第三轮GPU数值评价；新留出场景仅在终点权重封存后开启一次。"""
from __future__ import annotations
import argparse
import collections
import copy
import csv
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.stage2.spatial_v3_data import OUT,V2,sha,write
from scripts.stage2.spatial_v3_experiment import setup,encode,labels,head_forward,decode,attention_score,SELECTED
from scripts.stage2.spatial_v3_model import VARIANTS
from scripts.stage2.spatial_v3_objects import object_rows
from scripts.stage2.spatial_v2_experiment import V1,swap_direction
from scripts.stage2.bev_fusion import LanguageBEVFusion
from scripts.stage2.spatial_annotations import AUTHOR,project_attention
from scripts.stage2.spatial_report import confusion,summary
from scripts.stage2.spatial_detection import ego_centers,in_region
from scripts.stage2.micro_training import get_batch,forward_payload
from scripts.stage2.bev_coordinates import BEVGrid
from scripts.stage2.runtime_memory import release_memory

def segmentation(head,bev):
    return head.seg_decoder(head.feat_cropper(torch.flip(bev.transpose(2,3),[2])))

def fresh_cache(model,wrapped,manifest):
    """真实六相机前向，未注入语言；只在选择封存后执行。"""
    selection=json.loads((OUT/'train/selection.json').read_text())
    objects=json.loads((OUT/'objects_train/summary.json').read_text())
    write(OUT/'final_opened.json',dict(selection=selection,objects=objects,policy='single final opening; no further model selection/updates'))
    from mmcv import Config
    from mmdet3d.datasets import build_dataset
    from scripts.stage2.low_memory_dataset import bounded_metadata
    cfg=Config.fromfile(str(AUTHOR/'projects/configs/bevformer/bevformer_small_seg_det.py'))
    dc=copy.deepcopy(cfg.data.test);dc.ann_file=str(OUT/'dataset_infos.pkl');dc.test_mode=True;dc.pop('samples_per_gpu',None)
    with bounded_metadata({f['sample_token'] for f in manifest['frames']}):dataset=build_dataset(dc)
    directory=OUT/'final_cache';directory.mkdir()
    features={};records=[]
    for r in manifest['frames']:
        if r['split']!='final':continue
        captured={};original=model.pts_bbox_head.transformer.get_bev_features
        def capture(*a,**kw):
            value=original(*a,**kw);captured['bev']=BEVGrid().to_map(value).detach();return value
        model.pts_bbox_head.transformer.get_bev_features=capture
        data=get_batch(dataset,r['dataset_index'],20261028+r['index'])
        from scripts.stage2.joint_runtime import _first_tensor
        with np.load(OUT/'annotations'/r['frame_file']) as f: expected=f['gt']
        np.testing.assert_array_equal(_first_tensor(data['semantic_indices']).reshape(200,400).cpu().numpy(),expected)
        model.prev_frame_info={'prev_bev':None,'scene_token':None,'prev_pos':0,'prev_angle':0}
        try:
            with torch.no_grad():result=wrapped(return_loss=False,rescale=True,**forward_payload(data,inference=True))[0]['pts_bbox']
        finally:del model.pts_bbox_head.transformer.get_bev_features
        features[r['index']]=captured['bev']
        with torch.no_grad():decoded=decode(model,head_forward(model,captured['bev']))
        errors={k:float(np.max(np.abs(decoded[k]-result[key].tensor.cpu().numpy() if k=='boxes' else decoded[k]-result[key].cpu().numpy()))) for k,key in (('boxes','boxes_3d'),('scores','scores_3d'),('labels','labels_3d'))}
        if max(errors.values())>1e-4:raise ValueError(f'cached head differs from real GPU full image: {errors}')
        name=f"bev_{r['index']}.npz";np.savez_compressed(directory/name,bev=captured['bev'].cpu().numpy())
        records.append(dict(frame=r,cache_file=name,sha256=sha(directory/name),full_image_vs_cached_head_max_errors=errors,author_pipeline_gt_exact=True))
        print('V3_FRESH_IMAGE_FORWARD',r['sample_token'],errors,flush=True)
    write(directory/'cache.json',dict(frames=records,source='actual original full image CUDA forward; FP32 cache'))
    return dataset,features

def weights(pipeline,mode):
    chosen=json.loads((OUT/'train/selection.json').read_text())
    states={'v2':dict(state={k:v.detach().cpu().clone() for k,v in pipeline.fusion.state_dict().items()},flags=VARIANTS['control'],sha256=sha(SELECTED))}
    for name,r in chosen['variants'].items():
        if mode=='final' and name!=chosen['selected_variant']:continue
        assert sha(r['checkpoint'])==r['sha256']
        saved=torch.load(r['checkpoint'],map_location='cpu',weights_only=False)
        states[name]=dict(state=saved['state_dict'],flags=saved['flags'],sha256=r['sha256'])
    return states,chosen

def plot(row,data,path,ckpt):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,5,figsize=(21,8),constrained_layout=True)
    panels=[('Map GT',data['gt'],'tab10',0,3),('Target / ROI',data['region'].astype(int)+data['target'].astype(int),'viridis',0,2),
        ('Other same-class instances',data['other'],'gray_r',0,1),('Round2 Attention',data['v2_attention'],'magma',0,1),
        ('Round3 Attention',data['v3_attention'],'magma',0,1),('Original segmentation',data['original'],'tab10',0,3),
        ('Round2 segmentation',data['v2'],'tab10',0,3),('Round3 segmentation',data['v3'],'tab10',0,3),
        ('Same-direction other-instance text',data['cf_attention'],'magma',0,1),
        ('R3 minus R2 segmentation error',((data['v3']!=data['gt']).astype(int)-(data['v2']!=data['gt']).astype(int))*data['region'],'coolwarm',-1,1)]
    for ax,(title,value,cmap,lo,hi) in zip(axes.flat,panels):
        if title=='Same-direction other-instance text' and not row['counterfactual_text']:
            ax.text(.5,.5,'N/A: no unambiguous instance pair',ha='center',va='center',transform=ax.transAxes)
            ax.set_axis_off();continue
        ax.imshow(value,origin='lower',extent=(-30,30,-15,15),cmap=cmap,vmin=lo,vmax=hi,interpolation='nearest');ax.set_title(title,fontsize=10)
    fig.suptitle(f"{row['split'].upper()} sample={row['sample_token']} | {row['text']}\nR2={sha(SELECTED)[:12]} R3={ckpt[:12]} | same-direction counterfactual={row['counterfactual_text'] or 'N/A'}",fontsize=10)
    fig.savefig(path,dpi=130);plt.close(fig)

def maps(model,pipeline,manifest,features,mode):
    directory=OUT/(mode+'_map');directory.mkdir(exist_ok=False)
    rows=[r for r in manifest['annotations'] if r['split']==mode];text=encode(pipeline,rows)
    states,chosen=weights(pipeline,mode);selected=chosen['selected_variant'];native=pipeline.fusion
    old=LanguageBEVFusion(768).cuda().eval();old.load_state_dict(torch.load(V1,map_location='cpu',weights_only=False)['state_dict'])
    systems={'original':None,'v1':old,**states}
    acc={};records=[];all_payload={r['instruction_id']:{} for r in rows}
    for name,state in systems.items():
        if isinstance(state,dict):native.load_state_dict(state['state']);native.flags=dict(state['flags'])
        cm=np.zeros((4,4),dtype=np.int64);im=cm.copy();outside_fp=0;bg=0
        for n,r in enumerate(rows):
            gt,target,grid,region,distractors,other=labels(r);bev=features[r['index']];native.set_calibration(r)
            with torch.no_grad():
                if name=='original':enhanced=bev;a=None
                elif name=='v1':
                    d=old.forward_with_details(bev,text[r['text']][:,:768]);enhanced=d['enhanced_bev'];a=project_attention(d['attention'],grid)[0,0]
                else:
                    d=native.forward_with_details(bev,text[r['text']]);enhanced=d['enhanced_bev'];a=project_attention(d['attention'],grid)[0,0]
                logits=segmentation(model.pts_bbox_head,enhanced)
                if n==0 and name=='original':
                    full=head_forward(model,enhanced)['seg_preds'];assert torch.equal(logits,full)
                pred=logits.argmax(1)[0].cpu().numpy();truth=gt.cpu().numpy();roi=region.cpu().numpy()
                cm+=confusion(truth,pred,roi);im+=confusion(truth,pred,~roi)
                outside_fp+=int(((pred!=0)&(truth==0)&~roi).sum());bg+=int(((truth==0)&~roi).sum())
                payload=all_payload[r['instruction_id']];payload[name]=pred
                if a is not None:payload[name+'_attention']=a.cpu().numpy()
                if name==selected:
                    if r['counterfactual_text']:
                        cf=native.forward_with_details(bev,text[r['counterfactual_text']]);payload['cf_attention']=project_attention(cf['attention'],grid)[0,0].cpu().numpy()
                    else:payload['cf_attention']=np.zeros((200,400),dtype=np.float32)
                payload.update(gt=truth,target=target.cpu().numpy(),region=roi,other=other.cpu().numpy())
        acc[name]=dict(relevant_segmentation=summary(cm),irrelevant_segmentation=summary(im),outside_roi_foreground_fp=outside_fp,outside_roi_background=bg)
        if name not in ('original','v1'):
            acc[name]['attention']=attention_score(pipeline,rows,features,text)
        print('V3_MAP_EVAL',mode,name,acc[name]['relevant_segmentation']['foreground_miou'],flush=True)
    for n,r in enumerate(rows):
        payload=all_payload[r['instruction_id']];payload['v3']=payload[selected];payload['v3_attention']=payload[selected+'_attention']
        file=f'instruction_{n:03d}.npz';np.savez_compressed(directory/file,**payload);records.append({**r,'prediction_file':file})
        if mode=='final' or n<4:plot(r,payload,directory/f'comparison_{n:03d}.png',states[selected]['sha256'])
    write(directory/'metrics.json',dict(split=mode,n_instructions=len(rows),n_frames=len({r['index'] for r in rows}),metrics=acc,
        selected_variant=selected,selected_threshold=chosen['selected_threshold'],selection_used_this_split=mode=='dev',
        protocol='pooled confusion counts then foreground class mean IoU; relevant ROI=fixed Ego quadrant, irrelevant=complement, original unchanged GT200x400. Instance attention union counted over whole GT grid; pair-repeated quadrants instruction weighted, not independent samples.'))
    write(directory/'predictions.json',dict(annotations=records,checkpoints={k:v['sha256'] for k,v in states.items()},v1_sha256=sha(V1)))
    with (directory/'metrics.csv').open('x') as f:
        writer=csv.writer(f);writer.writerow(['branch','relevant_iou','irrelevant_iou','attention_iou_0.5','background_fp_0.5','same_quadrant_fp_0.5','instance_counterfactual_gap'])
        for name,r in acc.items():
            a=r.get('attention',{});c=a.get('thresholds',{}).get('0.5',{})
            writer.writerow([name,r['relevant_segmentation']['foreground_miou'],r['irrelevant_segmentation']['foreground_miou'],c.get('iou'),c.get('background_fp'),c.get('same_quadrant_fp'),a.get('instance_counterfactual_gap')])
    return states,chosen

def ap_metrics(rows,directory,branches):
    pools=collections.defaultdict(list);counts=collections.Counter();matches=[]
    for r in rows:
        with np.load(directory/r['prediction_file']) as f:
            gt=f['gt_boxes'];cls=r['prompt']['class_id']
            for branch in branches:
                boxes,scores,labels_=(f[branch+s] for s in ('_boxes','_scores','_labels'))
                regions=in_region(ego_centers(boxes,r),r['prompt'])
                for region,member in (('relevant',True),('irrelevant',False)):
                    ids=r[region+'_gt_indices'];key=(branch,cls,region);counts[key]+=len(ids);used=set()
                    for k in np.argsort(-scores,kind='stable'):
                        if labels_[k]!=cls or bool(regions[k])!=member:continue
                        possible=[i for i in ids if i not in used];chosen=None;distance=None
                        if possible:
                            dd=np.linalg.norm(gt[possible,:2]-boxes[k,:2],axis=1);j=int(dd.argmin())
                            if dd[j]<=2.:chosen=possible[j];distance=float(dd[j]);used.add(chosen)
                        pools[key].append((float(scores[k]),chosen is not None))
                        matches.append(dict(instruction_id=r['instruction_id'],branch=branch,region=region,score=float(scores[k]),tp=chosen is not None,
                            matched_annotation=r['gt_annotation_ids'][chosen] if chosen is not None else None,distance_xy_m=distance))
    metrics={}
    for b in branches:
        metrics[b]={}
        for region in ('relevant','irrelevant'):
            classes={}
            for c in (0,8):
                key=(b,c,region);preds=sorted(pools[key],key=lambda x:-x[0]);tp=0;area=0.;curve=[]
                for n,(s,hit) in enumerate(preds,1):
                    tp+=int(hit)
                    if hit:area+=tp/n
                    curve.append(dict(score=s,precision=tp/n,recall=tp/counts[key] if counts[key] else None))
                classes[str(c)]=dict(ap=area/counts[key] if counts[key] else None,gt=counts[key],tp=tp,fp=len(preds)-tp,fn=counts[key]-tp,precision_recall=curve)
            values=[v['ap'] for v in classes.values() if v['ap'] is not None]
            metrics[b][region]=dict(macro_ap=float(np.mean(values)) if values else None,classes=classes)
    return metrics,matches

def objects(model,pipeline,dataset,manifest,features,mode):
    directory=OUT/(mode+'_objects');directory.mkdir(exist_ok=False)
    frames=[r for r in manifest['frames'] if r['split']==mode];rows=object_rows(frames,dataset,final=mode=='final')
    texts=sorted({r['text'] for r in rows}|{swap_direction(r['text']) for r in rows});encoded={}
    for i in range(0,len(texts),16):
        with torch.no_grad():ee=pipeline.encoder(texts[i:i+16],'cuda')
        encoded.update({t:e[None] for t,e in zip(texts[i:i+16],ee)})
    states,chosen=weights(pipeline,mode);selected=chosen['selected_variant'];native=pipeline.fusion
    for name,record in json.loads((OUT/'objects_train/summary.json').read_text()).items():
        saved=torch.load(record['checkpoint'],map_location='cpu',weights_only=False)
        states[name]=dict(state=saved['state_dict'],flags=saved['flags'],sha256=record['sha256'])
    branches=['original','v1','v2',selected,'global_control','object_supervised','object_incorrect']
    old=LanguageBEVFusion(768).cuda().eval();old.load_state_dict(torch.load(V1,map_location='cpu',weights_only=False)['state_dict'])
    predictions=[]
    for n,r in enumerate(rows):
        bev=features[r['index']];native.set_calibration(r)
        ann=dataset.get_ann_info(r['dataset_index'] if mode=='final' else r['index'])
        payload=dict(gt_boxes=ann['gt_bboxes_3d'].tensor.numpy(),gt_labels=ann['gt_labels_3d'])
        for branch in branches:
            with torch.no_grad():
                if branch=='original':enhanced=bev
                elif branch=='v1':enhanced=old.forward_with_details(bev,encoded[r['text']][:,:768])['enhanced_bev']
                else:
                    state=states['object_supervised' if branch=='object_incorrect' else branch]
                    native.load_state_dict(state['state']);native.flags=dict(state['flags'])
                    text=swap_direction(r['text']) if branch=='object_incorrect' else r['text']
                    enhanced=native.forward_with_details(bev,encoded[text])['enhanced_bev']
                decoded=decode(model,head_forward(model,enhanced))
                payload.update({branch+'_'+k:v for k,v in decoded.items()})
        file=f'objects_{n:03d}.npz';np.savez_compressed(directory/file,**payload);predictions.append({**r,'prediction_file':file})
        if n%16==0:print('V3_OBJECT_EVAL',mode,n,len(rows),flush=True)
    metrics,matches=ap_metrics(predictions,directory,branches)
    write(directory/'protocol.json',dict(annotations=predictions,checkpoints={k:v['sha256'] for k,v in states.items()},
        definition='noninterpolated AP=sum precision at each TP / GT count. Per prompt/frame greedy same-class unmatchedGT XYdistance<=2m; predictions+GT partitioned by fixedEgo quadrant before matching. All decodedtop300 fullscore ranking; macro car/pedestrian withGT. EmptyGT prompts retained asFP. NOT nuScenes official AP.'))
    write(directory/'metrics.json',dict(metrics=metrics,n_instructions=len(rows),n_frames=len(frames),n_empty_prompts=sum(not r['relevant_gt_indices'] for r in rows)))
    write(directory/'matches.json',matches)
    with (directory/'metrics.csv').open('x') as f:
        w=csv.writer(f);w.writerow(['branch','relevant_ap_2m_xy','irrelevant_ap_2m_xy'])
        for b,m in metrics.items():w.writerow([b,m['relevant']['macro_ap'],m['irrelevant']['macro_ap']])

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--split',choices=('dev','final'),required=True);a=parser.parse_args()
    model,wrapped,pipeline,dataset,manifest,features=setup()
    if a.split=='final':dataset,features=fresh_cache(model,wrapped,manifest)
    maps(model,pipeline,manifest,features,a.split)
    # Reset before object weights(): maps mutates the module.
    from scripts.stage2.spatial_experiment import INITIAL
    from scripts.stage2.spatial_v3_model import InstanceFusion
    initial=torch.load(INITIAL,map_location='cpu',weights_only=False)['state_dict'];v2=torch.load(SELECTED,map_location='cpu',weights_only=False)['state_dict']
    from scripts.stage2.micro_training import seed_all
    seed_all(20261028);pipeline.fusion=InstanceFusion(initial,v2,VARIANTS['control']).cuda().eval()
    objects(model,pipeline,dataset,manifest,features,a.split)

if __name__=='__main__':main()
