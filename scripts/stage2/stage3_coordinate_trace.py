"""只读执行作者实际cropper的坐标ramp，比较Attention监督与SegHead采样位置。"""
import ast
import json
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.stage2.stage3_metric_review import OUT,V3,sha,write
from scripts.stage2.spatial_annotations import AUTHOR
from scripts.stage2.bev_coordinates import BEVGrid

def main():
    head=AUTHOR/'projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_head.py'
    config=AUTHOR/'projects/configs/bevformer/bevformer_small_seg_det.py'
    parsed=ast.parse(head.read_text());nodes=[n for n in parsed.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in ('calculate_birds_eye_view_parameters','BevFeatureSlicer')]
    env=dict(torch=torch,nn=nn,F=F);exec(compile(ast.Module(body=nodes,type_ignores=[]),str(head),'exec'),env)
    grids={}
    for node in ast.parse(config.read_text()).body:
        if isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name) and target.id in ('det_grid_conf','map_grid_conf'):grids[target.id]=ast.literal_eval(node.value)
    cropper=env['BevFeatureSlicer'](grids['det_grid_conf'],grids['map_grid_conf'])
    native=BEVGrid().lidar_centers(dtype=torch.float64)[...,:2]
    ramp=native.permute(2,0,1)[None]
    author=cropper(torch.flip(ramp.transpose(2,3),[2]))[0].permute(1,2,0).numpy()
    # Map-grid float32 centers are taken from the actual author cropper, not a rewrite.
    labels=json.loads((OUT/'labels/manifest.json').read_text())['annotations'];frames={r['sample_token']:r for r in labels}
    records=[]
    for token,row in frames.items():
        with np.load(OUT/'labels'/row['metric_label_file']) as f:grid=f['sampling_grid']
        true=grid[0]*51.2;error=np.linalg.norm(author-true,axis=-1)
        records.append(dict(sample_token=token,split=row['split'],mean_error_m=float(error.mean()),max_error_m=float(error.max()),
            mean_lidar_xy_offset_m=(author-true).mean((0,1)).tolist()))
    write(OUT/'coordinate_trace.json',dict(status='PASS',records=records,n_frames=len(records),
        mean_sampling_offset_m=float(np.mean([r['mean_error_m'] for r in records])),
        source_sha256={str(head):sha(head),str(config):sha(config)},
        method='actual author AST-extracted pure cropper/helper, native metric ramp, exact transpose/flip, compared with calibrated Ego/LiDAR GT centers; no author source or weights edited',
        limitation='structural sampling mismatch only; frozen pretrained head may have adapted. No evidence that changing cropper would improve metrics; no model/data/GT coordinate change made.'))
    print('ACTUAL_HEAD_COORDINATE_OFFSET_METERS',np.mean([r['mean_error_m'] for r in records]),flush=True)

if __name__=='__main__':main()
