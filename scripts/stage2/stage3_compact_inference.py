"""P0完成后的轻量化：移除未启用的通道诊断包装，逐指令验证输出等价。"""
from __future__ import annotations
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.stage2.stage3_metric_review import OUT,V3,sha,write,aggregate_predictions
from scripts.stage2.stage3_bottleneck import prepare
from scripts.stage2.spatial_v3_experiment import head_forward,decode
from scripts.stage2.spatial_annotations import project_attention
from scripts.stage2.spatial_v3_evaluate import ap_metrics
from scripts.stage2 import stage3_runtime_evaluation as runtime

def main():
    folder=OUT/'compact_inference';folder.mkdir(exist_ok=False)
    model,wrapped,pipeline,dataset,rows,features,text,v2=prepare()
    wrapper=pipeline.fusion
    v3state={k:v.detach().cpu().clone() for k,v in wrapper.core.state_dict().items()}
    selection=json.loads((OUT/'optimization/selection.json').read_text());selected=selection['selected']
    state=torch.load(selected['checkpoint'],map_location='cpu',weights_only=False)
    assert state['channel_enabled'] is False
    wrapper.load_state_dict(state['state_dict']);compact=wrapper.core
    checkpoint=folder/'fusion_compact_step032.pt'
    with checkpoint.open('xb') as f:torch.save(dict(state_dict=compact.state_dict(),flags=compact.flags,steps=32,
        source_checkpoint_sha256=sha(selected['checkpoint']),optimizer_updates_added=0,
        transformation='remove disabled diagnostic channel modules and exact zero-add allocations; actual core weights unchanged'),f)
    removed=sum(v.numel() for k,v in state['state_dict'].items() if not k.startswith('core.'))
    compact_count=sum(p.numel() for p in compact.parameters())
    write(folder/'protocol.json',dict(status='PASS',selected_source=selected,checkpoint=str(checkpoint),sha256=sha(checkpoint),
        compact_parameters=compact_count,removed_unused_parameters=removed,optimizer_updates_added=0,
        equality='all65 map instructions +all64 object instructions on same8devframes; compare full native head logits/box tensors and actual outputs, not merely parameter names',
        purpose='remove disabled diagnostic branch only after P0; no alpha/head/threshold change, no final inputs'))
    proof=[];pred_dir=folder/'dev_map_predictions';pred_dir.mkdir();pred_rows=[]
    meta=json.loads((OUT/'dev_new_predictions/predictions.json').read_text())
    with torch.inference_mode():
        for n,row in enumerate(meta['annotations']):
            bev=features[row['index']];wrapper.set_calibration(row)
            dw=wrapper.forward_with_details(bev,text[row['text']]);dc=compact.forward_with_details(bev,text[row['text']])
            assert torch.equal(dw['enhanced_bev'],dc['enhanced_bev']) and torch.equal(dw['attention'],dc['attention'])
            pw=head_forward(model,dw['enhanced_bev']);pc=head_forward(model,dc['enhanced_bev'])
            for key in ('seg_preds','all_cls_scores','all_bbox_preds'):assert torch.equal(pw[key],pc[key])
            with np.load(OUT/'dev_new_predictions'/row['prediction_file']) as f:payload={k:f[k] for k in f.files}
            payload['joint_control_compact']=pc['seg_preds'].argmax(1)[0].cpu().numpy()
            with np.load(OUT/'labels'/(row['instruction_id']+'.npz')) as f:grid=torch.tensor(f['sampling_grid'],device='cuda')
            payload['joint_control_compact_attention']=project_attention(dc['attention'],grid)[0,0].cpu().numpy()
            np.testing.assert_array_equal(payload['joint_control_compact'],payload['joint_control'])
            name=f'instruction_{n:03d}.npz';np.savez_compressed(pred_dir/name,**payload);pred_rows.append({**row,'prediction_file':name})
            proof.append(dict(instruction_id=row['instruction_id'],task='map',bev_bitwise_equal=True,head_bitwise_equal=True,max_head_error=0.))
    write(pred_dir/'predictions.json',dict(annotations=pred_rows,checkpoint_sha256=sha(checkpoint)))
    aggregate_predictions(dict(annotations=pred_rows),pred_dir,folder/'dev_map_metrics',('original','v2','combined','joint_control','joint_control_compact'))
    object_dir=folder/'dev_object_predictions';object_dir.mkdir();objects=json.loads((OUT/'dev_object_evaluation/predictions.json').read_text());object_rows=[]
    texts=sorted({r['text'] for r in objects['annotations']});encoded={}
    for n in range(0,len(texts),16):
        with torch.no_grad():values=pipeline.encoder(texts[n:n+16],'cuda')
        encoded.update({t:v[None] for t,v in zip(texts[n:n+16],values)})
    with torch.inference_mode():
        for n,row in enumerate(objects['annotations']):
            bev=features[row['index']];wrapper.set_calibration(row)
            dw=wrapper.forward_with_details(bev,encoded[row['text']]);dc=compact.forward_with_details(bev,encoded[row['text']])
            assert torch.equal(dw['enhanced_bev'],dc['enhanced_bev'])
            pw=head_forward(model,dw['enhanced_bev']);pc=head_forward(model,dc['enhanced_bev'])
            for key in ('all_cls_scores','all_bbox_preds','seg_preds'):assert torch.equal(pw[key],pc[key])
            with np.load(OUT/'dev_object_evaluation'/row['prediction_file']) as f:payload={k:f[k] for k in f.files}
            payload.update({'joint_control_compact_'+k:v for k,v in decode(model,pc).items()})
            name=f'objects_{n:03d}.npz';np.savez_compressed(object_dir/name,**payload);object_rows.append({**row,'prediction_file':name})
            proof.append(dict(instruction_id=row['instruction_id'],task='object',bev_bitwise_equal=True,head_bitwise_equal=True,max_head_error=0.))
    values,matches=ap_metrics(object_rows,object_dir,('joint_control','joint_control_compact'))
    assert values['joint_control']==values['joint_control_compact']
    write(object_dir/'metrics.json',dict(status='PASS',metrics=values));write(folder/'equivalence.json',dict(status='PASS',records=proof,n_map=65,n_object=64,maximum_head_error=0.))
    pipeline.fusion=compact
    # Same benchmark implementation, distinct new destination and native state schema.
    opt=folder/'optimization';opt.mkdir()
    write(opt/'selection.json',dict(selected_branch='joint_control_compact',selected=dict(checkpoint=str(checkpoint),sha256=sha(checkpoint))))
    del wrapper,dw,dc,pw,pc
    compact.last_bev=None
    runtime.OUT=folder
    runtime.benchmark(model,wrapped,pipeline,rows,features,text,v3state)
    print('STAGE3_COMPACT',compact_count,removed,'exact outputs on129 instructions',flush=True)

if __name__=='__main__':main()
