"""训练侧发现的GT过滤遗漏修正；独立新输出，只评价开发集，不重开最终集。"""
from pathlib import Path
import sys,json,math
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.stage2.spatial_v3_data import OUT,sha,write
from scripts.stage2 import spatial_v3_objects as obj
from scripts.stage2 import spatial_v3_evaluate as ev
from scripts.stage2.spatial_v3_experiment import setup,actual_losses

def filter_ann(ann,pc_range):
    boxes=ann['gt_bboxes_3d'].clone()
    region=boxes.in_range_bev([pc_range[0],pc_range[1],pc_range[3],pc_range[4]])
    labels=np.asarray(ann['gt_labels_3d'])
    names=(labels>=0)&(labels<10)
    keep=region.cpu().numpy()&names
    boxes=boxes[keep];boxes.limit_yaw(offset=.5,period=2*math.pi)
    return {**ann,'gt_bboxes_3d':boxes,'gt_labels_3d':labels[keep]},dict(raw=len(labels),outside_bev=int((~region).sum()),unknown_class=int((~names).sum()),retained=int(keep.sum()))

def corrected_losses(model,pred,ann):
    filtered,_=filter_ann(ann,model.pts_bbox_head.pc_range)
    return actual_losses(model,pred,filtered)

def main():
    out=OUT/'range_filter_correction';out.mkdir(exist_ok=False)
    for name in ('annotations','object_records.json','train'):(out/name).symlink_to(OUT/name,target_is_directory=name!='object_records.json')
    write(out/'correction_protocol.json',dict(trigger='training-only data audit: raw get_ann_info bypassed author ObjectRangeFilter/ObjectNameFilter',
        source_sha256=sha(Path(__file__)),base_object_source_sha256=sha(ROOT/'scripts/stage2/spatial_v3_objects.py'),
        action='restore exact author LiDAR in_range_bev, valid class0..9, limit_yaw; both object branches same64steps/init/schedule. Filter native attention target to same center range.',
        final_evaluation='BLOCKED for corrected models: scene0001 already opened, never reused; no new final set opened',
        no_selection='all terminal step64 results reported on dev only; no best-checkpoint selection'))
    obj.OUT=out;obj.actual_losses=corrected_losses
    original_mask=obj.native_box_mask
    def filtered_mask(boxes,selected,device):
        keep=[k for k in selected if -51.2<boxes[k,0]<51.2 and -51.2<boxes[k,1]<51.2]
        return original_mask(boxes,keep,device)
    obj.native_box_mask=filtered_mask
    obj.train()
    model,wrapped,pipeline,dataset,manifest,features=setup()
    profile=[]
    for frame in manifest['frames']:
        if frame['split']=='final':continue
        _,stats=filter_ann(dataset.get_ann_info(frame['index']),model.pts_bbox_head.pc_range)
        profile.append(dict(index=frame['index'],split=frame['split'],**stats))
    write(out/'training_gt_filter_audit.json',dict(frames=profile,totals={k:sum(r[k] for r in profile) for k in ('raw','outside_bev','unknown_class','retained')}))
    ev.OUT=out
    ev.objects(model,pipeline,dataset,manifest,features,'dev')
    original=json.loads((OUT/'dev_objects/metrics.json').read_text())['metrics']
    corrected=json.loads((out/'dev_objects/metrics.json').read_text())['metrics']
    result={}
    for branch in ('global_control','object_supervised'):
        a=original[branch];b=corrected[branch]
        result[branch]=dict(before=a,corrected=b,
            relevant_change_pp=(b['relevant']['macro_ap']-a['relevant']['macro_ap'])*100,
            irrelevant_change_pp=(b['irrelevant']['macro_ap']-a['irrelevant']['macro_ap'])*100)
    write(out/'comparison.json',dict(metrics=result,total_additional_updates=128,final_status='BLOCKED: corrected models have no new independent final evaluation',
        conclusion='dev-only input-correctness repair; does not retroactively replace original third-round final numbers'))
    print('V3_CORRECTED_DEV',json.dumps({b:(v['corrected']['relevant']['macro_ap'],v['relevant_change_pp']) for b,v in result.items()}),flush=True)

if __name__=='__main__':main()
