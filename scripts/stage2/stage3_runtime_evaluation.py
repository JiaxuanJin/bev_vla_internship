"""真实CUDA开发对象AP、Hungarian匹配核验及缓存/六相机延迟。"""
from __future__ import annotations
import copy
import csv
import datetime
import json
from pathlib import Path
import pickle
import subprocess
import sys
import time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.stage2.stage3_metric_review import OUT,V3,sha,write
from scripts.stage2.stage3_bottleneck import prepare,filtered_detection
from scripts.stage2.spatial_v3_experiment import head_forward,decode
from scripts.stage2.spatial_v3_range_correction import filter_ann
from scripts.stage2.spatial_v3_evaluate import ap_metrics
from scripts.stage2.micro_training import get_batch,forward_payload
from scripts.stage2.spatial_annotations import AUTHOR
from scripts.stage2.runtime_memory import release_memory

def object_evaluation(model,pipeline,dataset,features):
    folder=OUT/'dev_object_evaluation';folder.mkdir()
    meta=json.loads((V3/'range_filter_correction/dev_objects/protocol.json').read_text())
    checkpoints=json.loads((OUT/'optimization/checkpoints.json').read_text())
    states={b:torch.load(r['checkpoint'],map_location='cpu',weights_only=False)['state_dict'] for b,r in checkpoints.items()}
    texts=sorted({r['text'] for r in meta['annotations']});encoded={}
    for i in range(0,len(texts),16):
        with torch.no_grad():e=pipeline.encoder(texts[i:i+16],'cuda')
        encoded.update({t:v[None] for t,v in zip(texts[i:i+16],e)})
    records=[]
    for n,row in enumerate(meta['annotations']):
        source=V3/'range_filter_correction/dev_objects'/row['prediction_file']
        with np.load(source) as f:
            payload={k:f[k] for k in ('gt_boxes','gt_labels','original_boxes','original_scores','original_labels','v2_boxes','v2_scores','v2_labels','combined_boxes','combined_scores','combined_labels')}
        raw=dataset.get_ann_info(row['index'])
        np.testing.assert_array_equal(raw['gt_bboxes_3d'].tensor.numpy(),payload['gt_boxes']);np.testing.assert_array_equal(raw['gt_labels_3d'],payload['gt_labels'])
        pipeline.fusion.set_calibration(row);bev=features[row['index']]
        with torch.no_grad():
            for b,state in states.items():
                pipeline.fusion.load_state_dict(state);d=pipeline.fusion.forward_with_details(bev,encoded[row['text']]);prediction=decode(model,head_forward(model,d['enhanced_bev']))
                payload.update({b+'_'+k:v for k,v in prediction.items()})
        name=f'objects_{n:03d}.npz';np.savez_compressed(folder/name,**payload);records.append({**row,'prediction_file':name,'source_frozen_gt_sha256':sha(source)})
        if n%16==0:print('STAGE3_DEV_OBJECT',n,len(meta['annotations']),flush=True)
    branches=('original','v2','combined',*checkpoints)
    metrics,matches=ap_metrics(records,folder,branches)
    write(folder/'metrics.json',dict(status='PASS',metrics=metrics,n_instructions=len(records),n_frames=len({r['index'] for r in records}),
        empty_prompts=sum(not r['relevant_gt_indices'] for r in records),
        definition=meta['definition'],training='map instruction optimization with global filtered detection loss; object AP is transfer, no new dedicated object-language training'))
    write(folder/'predictions.json',dict(annotations=records,checkpoints=checkpoints));write(folder/'matches.json',matches)
    with (folder/'summary.csv').open('x') as f:
        w=csv.writer(f);w.writerow(['branch','relevant_ap_2m_xy','irrelevant_ap_2m_xy'])
        for b,m in metrics.items():w.writerow([b,m['relevant']['macro_ap'],m['irrelevant']['macro_ap']])

def matching_audit(model,pipeline,dataset,rows,features,text):
    selected=json.loads((OUT/'optimization/selection.json').read_text())['selected']
    pipeline.fusion.load_state_dict(torch.load(selected['checkpoint'],map_location='cpu',weights_only=False)['state_dict'])
    records=[]
    for index in sorted({r['index'] for r in rows if r['split']=='dev'}):
        row=next(r for r in rows if r['index']==index and r['split']=='dev')
        ann=dataset.get_ann_info(index);filtered,stats=filter_ann(ann,model.pts_bbox_head.pc_range)
        truth=filtered['gt_bboxes_3d'].gravity_center.numpy();classes=filtered['gt_labels_3d'];trace=[]
        original=model.pts_bbox_head._get_target_single
        def traced(*a,**kw):
            value=original(*a,**kw);labels_,lw,bt,bw,pos,neg=value
            if len(pos):
                centers=bt[pos,:3].detach().cpu().numpy();dist=np.linalg.norm(centers[:,None]-truth[None],axis=-1);ids=dist.argmin(1)
                assert dist.min(1).max()<1e-5 and len(set(ids))==len(ids)
                np.testing.assert_array_equal(labels_[pos].cpu().numpy(),classes[ids])
            else:ids=np.array([],dtype=int)
            assert len(pos)==len(classes)
            trace.append(dict(positive_queries=int(len(pos)),negative_queries=int(len(neg)),unique_gt_matches=len(set(ids)),
                max_match_center_error_m=float(dist.min(1).max()) if len(pos) else None,query_indices=pos.cpu().tolist(),gt_indices=ids.tolist()))
            return value
        pipeline.fusion.set_calibration(row)
        with torch.no_grad():d=pipeline.fusion.forward_with_details(features[index],text[row['text']]);pred=head_forward(model,d['enhanced_bev'])
        model.pts_bbox_head._get_target_single=traced
        try:
            with torch.no_grad():filtered_detection(model,pred,ann)
        finally:del model.pts_bbox_head._get_target_single
        records.append(dict(sample_token=row['sample_token'],index=index,filtered_gt=stats,decoder_layers=trace))
    write(OUT/'gpu_diagnosis/query_matching.json',dict(status='PASS',n_frames=len(records),records=records,
        scope='all8dev frames; actual author Hungarian calls with selected adapter outputs, filtered raw GT, class labels and unique gravity centers verified; no optimizer update'))

def timing(fn,warmup,iterations):
    for _ in range(warmup):
        with torch.inference_mode():fn()
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();values=[]
    for _ in range(iterations):
        torch.cuda.synchronize();start=time.perf_counter()
        with torch.inference_mode():fn()
        torch.cuda.synchronize();values.append((time.perf_counter()-start)*1000.)
    return dict(mean_ms=float(np.mean(values)),median_ms=float(np.median(values)),p95_ms=float(np.percentile(values,95)),
        iterations=iterations,warmup=warmup,raw_ms=values,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)

def benchmark(model,wrapped,pipeline,rows,features,text,v3state):
    folder=OUT/'latency';folder.mkdir()
    row=next(r for r in rows if r['split']=='dev')
    # Only one real frame remains resident; training caches must not inflate inference memory.
    for k in list(features):
        if k!=row['index']:del features[k]
    bev=features[row['index']]
    selection=json.loads((OUT/'optimization/selection.json').read_text());newstate=torch.load(selection['selected']['checkpoint'],map_location='cpu',weights_only=False)['state_dict']
    # Same real test pipeline / shape; no GT or ROI enters network inference.
    from mmcv import Config
    from mmdet3d.datasets import build_dataset
    from scripts.stage2.low_memory_dataset import bounded_metadata
    with (V3/'dataset_infos.pkl').open('rb') as f:ann=pickle.load(f)
    info=next(r for r in ann['infos'] if r['token']==row['sample_token'])
    with (folder/'timing_info.pkl').open('xb') as f:pickle.dump(dict(infos=[info],metadata={'version':'v1.0-trainval'}),f)
    cfg=Config.fromfile(str(AUTHOR/'projects/configs/bevformer/bevformer_small_seg_det.py'))
    dc=copy.deepcopy(cfg.data.test);dc.ann_file=str(folder/'timing_info.pkl');dc.test_mode=True;dc.pop('samples_per_gpu',None)
    with bounded_metadata({row['sample_token']}):dataset=build_dataset(dc)
    data=get_batch(dataset,0,20261029)
    from scripts.stage2.joint_runtime import _first_tensor
    shape=list(_first_tensor(data['img']).shape)
    protocol=dict(batch_size=1,sample_token=row['sample_token'],text=row['text'],image_shape=shape,bev_shape=list(bev.shape),dtype=str(bev.dtype),
        cached=dict(warmup=10,iterations=50,scope='cached genuineBEV + adapter + both native heads; text embedding cached, no backbone/BERT'),
        full=dict(warmup=5,iterations=20,scope='single real6camera inference: tensor scatter/metadata copy + frozen image backbone/BEV encoder + adapter text/BERT(if enabled) + native heads/decode; excludes diskIO, GT generation and fixed-frame calibration setup'),
        synchronization='torch.cuda.synchronize before/after each iteration; perf_counter; inference_mode; no temporal prevBEV',
        memory='resident author model only for Original; adapter/BERT resident additionally for adapter branches; all other training BEV caches released; allocated and reserved peaks separately',
        branch_order=['original','v3_frozen','new_selected'],hardware_gpu=torch.cuda.get_device_name(),torch=torch.__version__,selected_branch=selection['selected_branch'])
    write(folder/'protocol.json',protocol)
    result={}
    for branch,state in (('original',None),('v3_frozen',v3state),('new_selected',newstate)):
        if state is None:pipeline.cpu()
        else:pipeline.cuda();pipeline.fusion.load_state_dict(state);pipeline.fusion.set_calibration(row)
        release_memory()
        embedding=text[row['text']]
        def cached():
            if state is None:enhanced=bev
            else:enhanced=pipeline.fusion.forward_with_details(bev,embedding)['enhanced_bev']
            return head_forward(model,enhanced)
        def full():
            model.prev_frame_info={'prev_bev':None,'scene_token':None,'prev_pos':0,'prev_angle':0}
            payload=forward_payload(data,inference=True)
            if state is None:return wrapped(return_loss=False,rescale=True,**payload)
            with pipeline.inject_shared_bev(model.pts_bbox_head.transformer,[row['text']],strict=False):
                return wrapped(return_loss=False,rescale=True,**payload)
        result[branch]=dict(cached_head=timing(cached,10,50),full_six_camera=timing(full,5,20))
        print('STAGE3_LATENCY',branch,{k:{c:v[c] for c in ('mean_ms','median_ms','p95_ms','peak_allocated_mib')} for k,v in result[branch].items()},flush=True)
    write(folder/'metrics.json',dict(status='PASS',protocol=protocol,metrics=result))
    with (folder/'metrics.csv').open('x') as f:
        w=csv.writer(f);w.writerow(['branch','scope','mean_ms','median_ms','p95_ms','peak_allocated_mib','iterations','warmup'])
        for branch,scopes in result.items():
            for scope,value in scopes.items():w.writerow([branch,scope,*[value[k] for k in ('mean_ms','median_ms','p95_ms','peak_allocated_mib','iterations','warmup')]])

def main():
    model,wrapped,pipeline,dataset,rows,features,text,v2=prepare()
    v3state={k:v.detach().cpu().clone() for k,v in pipeline.fusion.state_dict().items()}
    write(OUT/'hardware.json',dict(recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),torch=torch.__version__,cuda_available=torch.cuda.is_available(),
        gpu=torch.cuda.get_device_name(),compute_capability=torch.cuda.get_device_capability(),python=sys.executable,
        nvidia_smi=subprocess.check_output(['nvidia-smi'],text=True),host_memory=subprocess.check_output(['free','-h'],text=True)))
    object_evaluation(model,pipeline,dataset,features)
    matching_audit(model,pipeline,dataset,rows,features,text)
    benchmark(model,wrapped,pipeline,rows,features,text,v3state)

if __name__=='__main__':main()
