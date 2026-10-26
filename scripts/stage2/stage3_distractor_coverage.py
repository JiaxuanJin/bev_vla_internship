"""补核全部可见同象限同类记录，包括历史候选门槛8像素以下的微小记录。"""
import json
import csv
from collections import Counter,defaultdict
import numpy as np
from scripts.stage2.stage3_metric_review import OUT,V3,write,neighborhood,binary_counts,semantic_scores,ratio

def main():
    folder=OUT/'complete_distractor_audit';folder.mkdir(exist_ok=False)
    labels=json.loads((OUT/'labels/manifest.json').read_text())['annotations'];coverage=[];pool=defaultdict(Counter);records=[]
    meta=json.loads((OUT/'dev_new_predictions/predictions.json').read_text());pred_by_id={r['instruction_id']:r for r in meta['annotations']}
    branches=('original','v2','combined','joint_control','gradient_balanced','local_balanced')
    for row in labels:
        with np.load(V3/'annotations'/row['frame_file']) as f:masks=f['masks'];classes=f['classes'];xy=f['ego_xy']
        with np.load(OUT/'labels'/row['metric_label_file']) as f:gt=f['gt'];region=f['region']
        ids=[k for k,c in enumerate(classes) if c==row['target_class'] and (masks[k]&region).any()]
        others=[k for k in ids if k!=row['selected_index']];small=[k for k in others if k not in row['same_quadrant_indices']]
        distances={k:float(np.linalg.norm(xy[masks[k]&region],axis=-1).min()) for k in ids}
        expected=min(ids,key=lambda k:distances[k]) if row['qualifier']=='nearest' else max(ids,key=lambda k:distances[k]) if row['qualifier']=='farthest' else None
        coverage.append(dict(instruction_id=row['instruction_id'],split=row['split'],additional_small_distractors=len(small),
            additional_small_pixels=sum(int((masks[k]&region).sum()) for k in small),
            qualifier=row['qualifier'],eligible_protocol_target_id=row['instance_id'],
            unrestricted_all_visible_target_id=row['instance_ids'][expected] if expected is not None else None,
            eligible_target_matches_all_visible_order=expected==row['selected_index'] if expected is not None else None,
            all_visible_distractor_instance_ids=[row['instance_ids'][k] for k in others]))
        if row['split']!='dev':continue
        predrow=pred_by_id[row['instruction_id']]
        with np.load(OUT/'dev_new_predictions'/predrow['prediction_file']) as f:data={k:f[k] for k in f.files}
        wrong=np.logical_or.reduce([masks[k]&region for k in others]) & ~masks[row['selected_index']] if others else np.zeros_like(region)
        for b in branches:
            truth=gt==row['target_class'];prediction=data[b]==row['target_class']
            for k in others:
                roi=neighborhood(masks[k]&region);counts=binary_counts(truth,prediction,roi)
                pool[(b,'semantic')].update(counts)
                records.append(dict(instruction_id=row['instruction_id'],branch=b,target_class=row['target_class'],
                    distractor_instance_id=row['instance_ids'][k],visible_pixels=int((masks[k]&region).sum()),**counts))
            if b+'_attention' in data:
                a=data[b+'_attention'];pool[(b,'attention')].update(wrong_sum=float(a[wrong].sum()),attention_sum=float(a.sum()),
                    wrong_fp=int(((a>=.5)&wrong).sum()),wrong_pixels=int(wrong.sum()))
    result={b:dict(distractor_neighborhood_semantic_IoU=semantic_scores(pool[(b,'semantic')]),
        same_quadrant_wrong_instance_mass=ratio(pool[(b,'attention')]['wrong_sum'],pool[(b,'attention')]['attention_sum']),
        wrong_instance_attention_counts=dict(pool[(b,'attention')])) for b in branches}
    write(folder/'metrics.json',dict(status='PASS',metrics=result,coverage=coverage,
        additional_small_records_by_split={s:sum(r['additional_small_distractors'] for r in coverage if r['split']==s) for s in ('train','dev')},
        definition='ALL visible real records (>=1pixel) for distractor analysis; same fixed2m radius; unchanged targets and no new model selection',
        qualifier_scope='frozen nearest/farthest candidate eligibility >=8pixels. Ordering vs all tiny visible records is separately audited; do not generalize unrestricted nearest semantics when mismatched.'))
    with (folder/'per_distractor.csv').open('x') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    print('COMPLETE_DISTRACTOR_COVERAGE', {s:sum(r['additional_small_distractors'] for r in coverage if r['split']==s) for s in ('train','dev')},flush=True)

if __name__=='__main__':main()
