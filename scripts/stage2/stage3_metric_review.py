"""第三阶段：物理实例邻域语义指标与同象限反事实；只读历史输入。"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
from scipy.ndimage import distance_transform_edt

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.stage2.spatial_report import confusion,summary

V3=ROOT/'outputs/stage2_spatial_v3_20261026'
V2=ROOT/'outputs/stage2_spatial_v2_20261026'
OUT=Path(os.environ.get('BEV_STAGE3_OUTPUT',str(ROOT/'outputs/stage3_metric_review_20261026'))).resolve()
BRANCHES=('original','v1','v2','control','selector','ranking','hardneg','combined','nogeometry','nofusion')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)

def ratio(a,b):return float(a/b) if b else None

def neighborhood(mask,radius_m=2.,pixel_m=.15):
    """原始GT像素中心的欧氏距离，米制半径，不依赖预测。"""
    if mask.dtype!=bool or mask.ndim!=2:raise ValueError('expected 2D bool instance mask')
    if not mask.any():raise ValueError('empty instance: NOT_EVALUABLE, never fabricate a neighborhood')
    return distance_transform_edt(~mask,sampling=(pixel_m,pixel_m))<=radius_m

def binary_counts(gt,pred,roi):
    return dict(intersection=int((gt&pred&roi).sum()),union=int(((gt|pred)&roi).sum()),
        gt_pixels=int((gt&roi).sum()),predicted_pixels=int((pred&roi).sum()),roi_pixels=int(roi.sum()))

def semantic_scores(counts):
    return {**dict(counts),'iou':ratio(counts['intersection'],counts['union']),
        'recall':ratio(counts['intersection'],counts['gt_pixels'])}

def preregister():
    OUT.mkdir(parents=True,exist_ok=True)
    # This file is written before predictions are opened by the new evaluator.
    write(OUT/'protocol.json',dict(status='PASS',radius_m=2.,gt_pixel_size_m=.15,attention_threshold=.5,
        definition='Euclidean buffer of visible selected real map record intersected with instruction Ego quadrant, clipped only at original200x400 window. Binary semantic GT is ALL true pixels of target class inside neighborhood, including other correct same-class instances.',
        target_pixel_recall='semantic target-class prediction on selected real-instance pixels, not attention recall',
        distractor='each other same-class real record visible in same Ego quadrant, using identical2m buffer; shared selected pixels excluded only for Attention wrong-instance accounting, never relabeled semantic background',
        pooling='sum intersections/unions across instruction-frame pairs then divide; class1/2 separate; distractors additionally repeat per real-record pair; publish independent frame and pair counts',
        selection='radius2m chosen by physical definition before new prediction metrics; no threshold/radius sweep or heldout tuning',
        tuning_allowed_splits=['six training scenes','scene-0061 development'],
        forbidden_selection_scenes=['scene-0553','scene-0001','old round1 validation'],
        final_policy='frozen-array consistency checks only; no new final predictions or selection',
        source_sha256=sha(Path(__file__))))
    historical=[]
    for name in ('stage2_spatial_20261026','stage2_spatial_v2_20261026','stage2_spatial_v3_20261026'):
        for p in sorted((ROOT/'outputs'/name).rglob('*')):
            if p.is_file() and not p.is_symlink():historical.append(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)))
    write(OUT/'frozen_history.json',dict(files=historical,policy='all three earlier rounds immutable; verify again at completion'))
    source=json.loads((V3/'annotations/manifest.json').read_text())
    directory=OUT/'labels';directory.mkdir()
    rows=[]
    for row in source['annotations']:
        if row['split'] not in ('train','dev'):continue
        with np.load(V3/'annotations'/row['frame_file']) as f:
            masks=f['masks'];gt=f['gt'];xy=f['ego_xy'];grid=f['sampling_grid']
        region=((xy[...,0]>0) if row['fb']=='front' else (xy[...,0]<0))&((xy[...,1]>0) if row['lr']=='left' else (xy[...,1]<0))
        full=masks[row['selected_index']];target=full&region
        distractors=[masks[k]&region for k in row['same_quadrant_indices']]
        neighbors=[neighborhood(m) for m in distractors]
        target_roi=neighborhood(target)
        others=np.logical_or.reduce(distractors) & ~full if distractors else np.zeros_like(target)
        filename=row['instruction_id']+'.npz'
        np.savez_compressed(directory/filename,gt=gt,target=target,region=region,target_neighborhood=target_roi,
            distractor_neighborhoods=np.array(neighbors,dtype=bool).reshape(-1,200,400),
            wrong_instance=others,sampling_grid=grid)
        boundary=target_roi[0].any() or target_roi[-1].any() or target_roi[:,0].any() or target_roi[:,-1].any()
        rows.append({**row,'metric_label_file':filename,'metric_label_sha256':sha(directory/filename),
            'target_ego_centroid_m':xy[target].mean(0).tolist(),'target_lidar_xy_centroid_m':(grid[0][target]*51.2).mean(0).tolist(),
            'roi_radius_m':2.,'roi_pixels':int(target_roi.sum()),'roi_area_m2':float(target_roi.sum()*.15**2),
            'window_boundary_touched':bool(boundary),'distractor_instance_ids':[row['instance_ids'][k] for k in row['same_quadrant_indices']]})
    write(directory/'manifest.json',dict(annotations=rows,counts=dict(Counter(r['split'] for r in rows)),
        source_manifest_sha256=sha(V3/'annotations/manifest.json'),protocol_sha256=sha(OUT/'protocol.json')))
    print('PREREGISTERED_2M_NEIGHBORHOODS',dict(Counter(r['split'] for r in rows)),flush=True)

def aggregate_predictions(metadata,directory,output,branches):
    output.mkdir(parents=True,exist_ok=False)
    labels_by_id={r['instruction_id']:r for r in json.loads((OUT/'labels/manifest.json').read_text())['annotations']}
    rows=metadata['annotations'];lookup={(r['sample_token'],r['text']):r for r in rows}
    pools={};records=[];counterfactual=[];missing=[]
    for b in branches:
        pools[b]=dict(target=Counter(),distractor=Counter(),by_class=defaultdict(lambda:dict(target=Counter(),distractor=Counter())),
            target_recall=Counter(),attention=Counter(),quadrant=np.zeros((4,4),dtype=np.int64),irrelevant=np.zeros((4,4),dtype=np.int64),
            outside_target_neighborhood=Counter(),outside_quadrant=Counter())
    for r in rows:
        label_row=labels_by_id[r['instruction_id']]
        with np.load(OUT/'labels'/label_row['metric_label_file']) as f:lab={k:f[k] for k in f.files}
        with np.load(directory/r['prediction_file']) as f:data={k:f[k] for k in f.files}
        gt=lab['gt'];target=lab['target'];roi=lab['target_neighborhood'];region=lab['region'];cls=r['target_class'];truth=gt==cls
        np.testing.assert_array_equal(data['gt'],gt)
        partner=lookup.get((r['sample_token'],r.get('counterfactual_text')))
        cf=None
        if partner:
            if partner['target_class']!=cls or partner['fb']!=r['fb'] or partner['lr']!=r['lr'] or partner['instance_id']==r['instance_id']:raise ValueError('invalid instance counterfactual')
            with np.load(directory/partner['prediction_file']) as f:cf={k:f[k] for k in f.files}
        elif r.get('counterfactual_text'):missing.append(r['instruction_id'])
        for b in branches:
            p=data[b];pred=p==cls;acc=pools[b]
            tc=binary_counts(truth,pred,roi);acc['target'].update(tc);acc['by_class'][str(cls)]['target'].update(tc)
            tr=dict(intersection=int((pred&target).sum()),gt_pixels=int(target.sum()));acc['target_recall'].update(tr)
            acc['quadrant']+=confusion(gt,p,region);acc['irrelevant']+=confusion(gt,p,~region)
            acc['outside_target_neighborhood'].update(fp=int((pred&~truth&~roi).sum()),negative_pixels=int((~truth&~roi).sum()))
            acc['outside_quadrant'].update(fp=int(((p!=0)&(gt==0)&~region).sum()),background_pixels=int(((gt==0)&~region).sum()))
            item=dict(instruction_id=r['instruction_id'],sample_token=r['sample_token'],instance_id=r['instance_id'],target_class=cls,branch=b,
                prediction_file=str(directory/r['prediction_file']),prediction_sha256=sha(directory/r['prediction_file']),
                metric_label_file=str(OUT/'labels'/label_row['metric_label_file']),**tc,target_instance_semantic_tp=tr['intersection'],target_instance_gt_pixels=tr['gt_pixels'],
                n_same_quadrant_distractors=len(lab['distractor_neighborhoods']))
            for i,neighbor in enumerate(lab['distractor_neighborhoods']):
                dc=binary_counts(truth,pred,neighbor);acc['distractor'].update(dc);acc['by_class'][str(cls)]['distractor'].update(dc)
                records.append(dict(item,record_type='distractor_neighborhood',evaluated_instance_id=label_row['distractor_instance_ids'][i],**{f'distractor_{k}':v for k,v in dc.items()}))
            if b+'_attention' in data:
                a=data[b+'_attention'];active=a>=.5;wrong=lab['wrong_instance'];bg=gt==0
                ac=dict(tp=int((active&target).sum()),union=int((active|target).sum()),gt=int(target.sum()),pred=int(active.sum()),
                    background_fp=int((active&bg).sum()),background_pixels=int(bg.sum()),wrong_instance_fp=int((active&wrong).sum()),wrong_instance_pixels=int(wrong.sum()),
                    wrong_instance_attention_sum=float(a[wrong].sum()),target_attention_sum=float(a[target].sum()),attention_sum=float(a.sum()))
                acc['attention'].update(ac);item.update({f'attention_{k}':v for k,v in ac.items()})
            records.append(dict(item,record_type='target_neighborhood',evaluated_instance_id=r['instance_id']))
            if cf is not None:
                cp=cf[b];cc=binary_counts(truth,cp==cls,roi)
                cfrow=dict(instruction_id=r['instruction_id'],counterfactual_id=partner['instruction_id'],sample_token=r['sample_token'],branch=b,target_class=cls,
                    correct_iou=ratio(tc['intersection'],tc['union']),counterfactual_iou=ratio(cc['intersection'],cc['union']),
                    correct_intersection=tc['intersection'],correct_union=tc['union'],counterfactual_intersection=cc['intersection'],counterfactual_union=cc['union'],
                    roi_pixels=int(roi.sum()),semantic_changed_pixels=int(((p!=cp)&roi).sum()),
                    correct_target_semantic_tp=int((pred&target).sum()),counterfactual_target_semantic_tp=int(((cp==cls)&target).sum()),target_gt_pixels=int(target.sum()),
                    correct_target_attention=float(data[b+'_attention'][target].mean()) if b+'_attention' in data else None,
                    counterfactual_target_attention=float(cf[b+'_attention'][target].mean()) if b+'_attention' in cf else None)
                counterfactual.append(cfrow)
    metrics={}
    for b,acc in pools.items():
        a=acc['attention'];paired=[r for r in counterfactual if r['branch']==b]
        af=None if not a else {**dict(a),'target_attention_IoU':ratio(a['tp'],a['union']),'target_attention_precision':ratio(a['tp'],a['pred']),
            'target_attention_recall':ratio(a['tp'],a['gt']),'same_quadrant_wrong_instance_mass':ratio(a['wrong_instance_attention_sum'],a['attention_sum']),
            'background_attention_FP_rate':ratio(a['background_fp'],a['background_pixels'])}
        classes={c:{k:semantic_scores(v) if v else dict(status='NOT_EVALUABLE',reason='no distractor neighborhood') for k,v in d.items()} for c,d in acc['by_class'].items()}
        metrics[b]=dict(instruction_quadrant_semantic_mIoU=summary(acc['quadrant']),instruction_irrelevant_semantic_mIoU=summary(acc['irrelevant']),
            target_neighborhood_semantic_IoU=semantic_scores(acc['target']),target_pixel_recall={**dict(acc['target_recall']),'recall':ratio(acc['target_recall']['intersection'],acc['target_recall']['gt_pixels'])},
            distractor_neighborhood_semantic_IoU=semantic_scores(acc['distractor']) if acc['distractor'] else dict(status='NOT_EVALUABLE'),
            per_target_class=classes,attention=af,attention_status='PASS' if af else 'NOT_EVALUABLE: no attention branch',
            outside_target_neighborhood=dict(acc['outside_target_neighborhood']),outside_quadrant=dict(acc['outside_quadrant']),
            counterfactual=dict(directed_instruction_pairs=len(paired),unique_unordered_pairs=len(paired)//2,
                semantic_changed_pixels=sum(r['semantic_changed_pixels'] for r in paired),roi_pixels=sum(r['roi_pixels'] for r in paired),
                any_semantic_change_count=sum(r['semantic_changed_pixels']>0 for r in paired),
                mean_target_neighborhood_iou_delta=float(np.mean([r['correct_iou']-r['counterfactual_iou'] for r in paired])) if paired else None,
                pooled_correct_iou=ratio(sum(r['correct_intersection'] for r in paired),sum(r['correct_union'] for r in paired)),
                pooled_counterfactual_iou=ratio(sum(r['counterfactual_intersection'] for r in paired),sum(r['counterfactual_union'] for r in paired)),
                semantic_iou_wins=sum(r['correct_iou']>r['counterfactual_iou'] for r in paired),semantic_iou_ties=sum(r['correct_iou']==r['counterfactual_iou'] for r in paired),
                mean_target_attention_gap=float(np.mean([r['correct_target_attention']-r['counterfactual_target_attention'] for r in paired])) if af and paired else None,
                attention_wins=sum(r['correct_target_attention']>r['counterfactual_target_attention'] for r in paired) if af else None))
    write(output/'metrics.json',dict(status='PASS',metrics=metrics,n_instructions=len(rows),n_frames=len({r['sample_token'] for r in rows}),
        n_scenes=len({r['scene_token'] for r in rows}),n_distractor_record_evaluations=sum(len(labels_by_id[r['instruction_id']]['same_quadrant_indices']) for r in rows),
        not_evaluable_counterfactual_instructions=[r['instruction_id'] for r in rows if not r.get('counterfactual_text')],missing_counterfactuals=missing,
        protocol_sha256=sha(OUT/'protocol.json'),weighting='instruction repeated; distractor additionally per-record repeated; class1/2 pooled union IoU, no per-frame IoU averaging',
        map_detection_ap=dict(status='NOT_EVALUABLE',reason='map record has no corresponding object3D box')))
    for name,items in [('per_instruction',records),('counterfactual',counterfactual)]:
        with (output/(name+'.csv')).open('x') as f:
            writer=csv.DictWriter(f,fieldnames=sorted({k for r in items for k in r}));writer.writeheader();writer.writerows(items)
    with (output/'summary.csv').open('x') as f:
        w=csv.writer(f);w.writerow(['branch','quadrant_semantic_miou','target_neighborhood_semantic_iou','target_pixel_recall','distractor_neighborhood_semantic_iou','irrelevant_semantic_miou','attention_iou','cf_semantic_changed_pixels','cf_attention_gap'])
        for b,m in metrics.items():w.writerow([b,m['instruction_quadrant_semantic_mIoU']['foreground_miou'],m['target_neighborhood_semantic_IoU']['iou'],m['target_pixel_recall']['recall'],m['distractor_neighborhood_semantic_IoU'].get('iou'),m['instruction_irrelevant_semantic_mIoU']['foreground_miou'],(m['attention'] or {}).get('target_attention_IoU'),m['counterfactual']['semantic_changed_pixels'],m['counterfactual']['mean_target_attention_gap']])
    return metrics

def frozen_audit():
    directory=OUT/'frozen_array_audit';directory.mkdir()
    report=json.loads((V3/'report/metrics.json').read_text());meta=json.loads((V3/'final_map/predictions.json').read_text());branches=('original','v1','v2','combined')
    pools={b:dict(q=np.zeros((4,4),dtype=np.int64),outside=np.zeros((4,4),dtype=np.int64),a=Counter()) for b in branches}
    for row in meta['annotations']:
        with np.load(V3/'final_map'/row['prediction_file']) as f:
            for b in branches:
                x=pools[b];gt=f['gt'];roi=f['region'];x['q']+=confusion(gt,f[b],roi);x['outside']+=confusion(gt,f[b],~roi)
                if b!='original':
                    a=f[b+'_attention'];target=f['target'];active=a>=.5
                    x['a'].update(tp=int((active&target).sum()),union=int((active|target).sum()),gt=int(target.sum()),pred=int(active.sum()),background_fp=int((active&(gt==0)).sum()))
    checks={}
    for b,pool in pools.items():
        measured=summary(pool['q'])['foreground_miou'];expected=report['final_map_metrics']['metrics'][b]['relevant_segmentation']['foreground_miou']
        assert measured==expected
        for k,v in pool['a'].items():assert v==report['final_attention_raw_counts'][b][k]
        checks[b]=dict(quadrant_semantic_miou=measured,irrelevant_semantic_miou=summary(pool['outside'])['foreground_miou'],attention_counts=dict(pool['a']))
    from scripts.stage2.spatial_v3_evaluate import ap_metrics
    obj=json.loads((V3/'final_objects/protocol.json').read_text());br=tuple(report['final_object_metrics']['metrics'])
    values,_=ap_metrics(obj['annotations'],V3/'final_objects',br)
    for b in br:
        for region in ('relevant','irrelevant'):assert values[b][region]['macro_ap']==report['final_object_metrics']['metrics'][b][region]['macro_ap']
    write(directory/'audit.json',dict(status='PASS',map_from_raw_npz=checks,object_from_raw_npz=values,n_object_instructions=len(obj['annotations']),
        empty_object_prompts=sum(not r['relevant_gt_indices'] for r in obj['annotations']),
        policy='frozen final arrays reaggregated only; zero final model forward/selection/optimization',
        source_hashes={str(p.relative_to(ROOT)):sha(p) for p in (V3/'report/metrics.json',V3/'report/comparison.csv',V3/'report/final_pixel_counts.csv',V3/'final_objects/protocol.json')}))
    print('FROZEN_RAW_ARRAY_AUDIT',len(meta['annotations']),len(obj['annotations']),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=('protocol','frozen','dev'),required=True);a=p.parse_args()
    if a.phase=='protocol':preregister()
    elif a.phase=='frozen':frozen_audit()
    else:
        meta=json.loads((V3/'dev_map/predictions.json').read_text())
        m=aggregate_predictions(meta,V3/'dev_map',OUT/'dev_frozen_metrics',BRANCHES)
        for b in ('original','v2','combined'):
            print('REAL_NEIGHBORHOOD',b,m[b]['target_neighborhood_semantic_IoU']['iou'],m[b]['counterfactual'],flush=True)

if __name__=='__main__':main()
