"""几何完整性补核：不把旧>=8像素候选语义等同于所有可见实例的nearest/farthest。"""
import json
import numpy as np
from scripts.stage2.stage3_metric_review import OUT,V3,write,aggregate_predictions

def main():
    folder=OUT/'strict_qualifier_audit';folder.mkdir(exist_ok=False)
    annotations=json.loads((OUT/'labels/manifest.json').read_text())['annotations'];audit=[];invalid=set()
    for row in annotations:
        with np.load(V3/'annotations'/row['frame_file']) as f:masks=f['masks'];classes=f['classes'];xy=f['ego_xy']
        with np.load(OUT/'labels'/row['metric_label_file']) as f:region=f['region']
        distances=sorted((float(np.linalg.norm(xy[masks[k]&region],axis=-1).min()),k) for k,c in enumerate(classes) if c==row['target_class'] and (masks[k]&region).any())
        reason=None;gap=None
        if row['qualifier']=='unique':
            if len(distances)!=1:reason='multiple real visible records including <8pixel instance'
        else:
            ordered=distances if row['qualifier']=='nearest' else list(reversed(distances))
            if ordered[0][1]!=row['selected_index']:reason='selected eligible record is not unrestricted visible-record extreme'
            if len(ordered)>1:
                gap=abs(ordered[1][0]-ordered[0][0])
                if gap<1.:reason='all-visible order margin below frozen1m ambiguity rule'
        if reason:invalid.add(row['instruction_id'])
        audit.append(dict(instruction_id=row['instruction_id'],sample_token=row['sample_token'],split=row['split'],
            status='NOT_EVALUABLE' if reason else 'PASS',reason=reason,all_visible_order_gap_m=gap,n_visible_records=len(distances),
            frozen_eligible_target_id=row['instance_id'],all_visible_records=[dict(distance_m=d,instance_id=row['instance_ids'][k]) for d,k in distances]))
    metadata=json.loads((OUT/'dev_new_predictions/predictions.json').read_text());lookup={(r['sample_token'],r['text']):r for r in metadata['annotations']};strict=[];cf_invalid=[]
    for row in metadata['annotations']:
        if row['instruction_id'] in invalid:continue
        r=dict(row);partner=lookup.get((r['sample_token'],r.get('counterfactual_text')))
        if partner and partner['instruction_id'] in invalid:cf_invalid.append(r['instruction_id']);r['counterfactual_text']=None
        strict.append(r)
    result=aggregate_predictions(dict(annotations=strict),OUT/'dev_new_predictions',folder/'dev_metrics',('original','v2','combined','joint_control','gradient_balanced','local_balanced'))
    write(folder/'audit.json',dict(status='PASS',annotations=audit,invalid_counts={s:sum(r['split']==s and r['status']=='NOT_EVALUABLE' for r in audit) for s in ('train','dev')},
        invalid_counterfactual_partners=cf_invalid,strict_dev_instructions=len(strict),
        strict_dev_directed_counterfactuals=result['joint_control']['counterfactual']['directed_instruction_pairs'],
        policy='geometry-only completeness audit; original65/52 frozen protocol and selection preserved. Strict subset shown additionally, no new training/selection or heldout changes.'))
    print('STRICT_QUALIFIER_AUDIT',len(strict),result['joint_control']['counterfactual']['directed_instruction_pairs'],flush=True)

if __name__=='__main__':main()
