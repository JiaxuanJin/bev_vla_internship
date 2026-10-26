"""第三轮独立派生标签；历史目录只读，实例选择只依赖真实几何。"""
from __future__ import annotations
import collections
import hashlib
import json
from pathlib import Path
import pickle
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.stage2.spatial_annotations import AUTHOR, DATA, load_module, metric_grid, tagged_vectors
from scripts.stage2.low_memory_dataset import json_records

OUT = ROOT / 'outputs/stage2_spatial_v3_20261026'
V2 = ROOT / 'outputs/stage2_spatial_v2_20261026'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''): h.update(block)
    return h.hexdigest()

def write(path, value):
    with Path(path).open('x') as f: json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)

def main():
    OUT.mkdir(exist_ok=False)
    frozen = []
    for directory in (ROOT/'outputs/stage2_spatial_20261026', V2):
        for path in sorted(directory.rglob('*')):
            if path.is_file(): frozen.append(dict(path=str(path.relative_to(ROOT)), size=path.stat().st_size, sha256=sha(path)))
    write(OUT/'frozen_history.json', {'files': frozen, 'policy': 'all round1/2 artifacts immutable; verify again at completion'})
    manifest = json.loads((V2/'annotations_nearest/manifest.json').read_text())
    frames = [dict(r) for r in manifest['frames'] if r['split'] in ('train','dev')]
    with (DATA/'nuscenes_infos_temporal_train.pkl').open('rb') as f: mini = pickle.load(f)
    mini_infos = sorted(mini['infos'], key=lambda r:r['timestamp'])
    with (ROOT/'outputs/bevformer_full_infos/bevformer_full_infos_temporal_train.pkl').open('rb') as f: full=pickle.load(f)
    scenes={r['token']:r for r in json.loads((DATA/'v1.0-trainval/scene.json').read_text())}
    logs={r['token']:r for r in json.loads((DATA/'v1.0-trainval/log.json').read_text())}
    mini_scenes={r['token'] for r in json.loads((DATA/'v1.0-mini/scene.json').read_text())}
    # Metadata-only deterministic scene choice, made before any third-round model metrics.
    candidates=sorted({r['scene_token'] for r in full['infos']} - mini_scenes, key=lambda s:scenes[s]['name'])
    final_scene=candidates[0]
    available=sorted([r for r in full['infos'] if r['scene_token']==final_scene],key=lambda r:r['timestamp'])
    final_infos=[available[n] for n in np.linspace(0,len(available)-1,4,dtype=int)]
    infos={r['index']:mini_infos[r['index']] for r in frames}
    for n,info in enumerate(final_infos,1000):
        infos[n]=info
        frames.append(dict(index=n,split='final',scene_token=final_scene,scene_name=scenes[final_scene]['name'],
            sample_token=info['token'], location=logs[scenes[final_scene]['log_token']]['location'],
            **{k:info[k] for k in ('lidar2ego_rotation','lidar2ego_translation','ego2global_rotation','ego2global_translation')}))
    sensor_check=[]
    for info in final_infos:
        paths=[Path(c['data_path']) for c in info['cams'].values()]+[Path(info['lidar_path'])]
        if not all(p.is_file() for p in paths): raise ValueError('new heldout incomplete')
        sensor_check.extend({'path':str(p),'bytes':p.stat().st_size} for p in paths)
    write(OUT/'split_protocol.json', dict(seed=20261028, train_scenes=manifest['split_scenes']['train'],
        dev_scenes=manifest['split_scenes']['dev'], final_scene=final_scene, final_scene_name=scenes[final_scene]['name'],
        final_sample_tokens=[r['token'] for r in final_infos], final_sensor_files=sensor_check,
        forbidden_selection_scenes=[manifest['split_scenes']['final'][0]],
        scope='New scene from official train outside all mini scenes; no round1/2 optimizer or evaluation uses it. Author pretrained checkpoint may have seen official train: adaptation-scene holdout, NOT pretraining-unseen generalization.',
        historical_official_val='All6019 official val frames were evaluated on Oct19; therefore none relabeled as fresh.',
        final_opened=False))
    labels=OUT/'annotations'; labels.mkdir()
    tools=AUTHOR/'projects/mmdet3d_plugin/datasets/utils'
    vector=load_module('v3_vector',tools/'vector_map.py'); raster=load_module('v3_raster',tools/'rasterize.py')
    mapper=vector.VectorizedLocalMap(str(DATA),(30.,60.),(200,400))
    annotations=[]; rejected=[]; dataset_infos=[]
    for frame in frames:
        info=infos[frame['index']]
        frame['dataset_index']=len(dataset_infos); dataset_infos.append(info)
        vectors,tagged=tagged_vectors(mapper,info,frame['location'])
        semantic,*_=raster.preprocess_map(vectors,(30.,60.),(200,400),3,5,36)
        gt=(semantic.numpy()*np.arange(1,4)[:,None,None]).sum(0).astype(np.uint8)
        if frame['split']!='final':
            with np.load(V2/'annotations_nearest'/frame['frame_file']) as old: np.testing.assert_array_equal(gt,old['gt'])
        groups={}
        for v,identity in tagged: groups.setdefault(identity,[]).append(v)
        masks=[]; identities=[]; classes=[]
        for identity,vv in groups.items():
            single,*_=raster.preprocess_map(vv,(30.,60.),(200,400),3,5,36)
            cls=vv[0]['type']+1; mask=single[cls-1].numpy().astype(bool)&(gt==cls)
            if mask.any(): identities.append(identity);classes.append(cls);masks.append(mask)
        masks=np.asarray(masks,dtype=bool)
        grid,ego,_=metric_grid(info)
        frame['frame_file']=f"frame_{frame['index']:04d}.npz"
        np.savez_compressed(labels/frame['frame_file'],gt=gt,sampling_grid=grid.numpy(),masks=masks,classes=classes,ego_xy=ego[...,:2])
        frame['instance_ids']=identities
        for cls,name in ((1,'lane markings'),(2,'crosswalk')):
            for fb,lr in (('front','left'),('front','right'),('rear','left'),('rear','right')):
                region=((ego[...,0]>0) if fb=='front' else (ego[...,0]<0))&((ego[...,1]>0) if lr=='left' else (ego[...,1]<0))
                eligible=[k for k,c in enumerate(classes) if c==cls and (masks[k]&region).sum()>=8]
                ordered=sorted((float(np.linalg.norm(ego[...,:2][masks[k]&region],axis=-1).min()),k) for k in eligible)
                if not ordered: continue
                if len(ordered)==1: selections=[('unique',ordered[0])]
                else:
                    selections=[]
                    if ordered[1][0]-ordered[0][0]>=1.: selections.append(('nearest',ordered[0]))
                    if ordered[-1][0]-ordered[-2][0]>=1.: selections.append(('farthest',ordered[-1]))
                for qualifier,(distance,k) in selections:
                    opposite='farthest' if qualifier=='nearest' else 'nearest'
                    has_pair=(len(selections)==2 and qualifier!='unique')
                    prefix='' if qualifier=='unique' else qualifier+' '
                    annotations.append({**frame,'instruction_id':f"{info['token']}_c{cls}_{fb}_{lr}_{qualifier}",
                        'text':f'Find the {prefix}{name} on the {fb} {lr}', 'target_class':cls,
                        'qualifier':qualifier,'fb':fb,'lr':lr,'selected_index':k,'instance_id':identities[k],
                        'distance_m':distance,'same_quadrant_indices':[i for _,i in ordered if i!=k],
                        'counterfactual_text':f'Find the {opposite} {name} on the {fb} {lr}' if has_pair else None,
                        'target_pixels':int((masks[k]&region).sum()), 'selection_rule':'min visible-raster-point ego radial distance; unique or adjacent-order gap >=1m; farthest means largest minimum distance, not farthest point'})
        print('V3_REAL_LABEL',frame['split'],frame['index'],len(identities),flush=True)
    # The dataset re-sorts by timestamp. Persist exact resulting order.
    dataset_infos=sorted(dataset_infos,key=lambda r:r['timestamp'])
    token_to_index={r['token']:i for i,r in enumerate(dataset_infos)}
    for row in frames+annotations: row['dataset_index']=token_to_index[row['sample_token']]
    with (OUT/'dataset_infos.pkl').open('xb') as f: pickle.dump({'infos':dataset_infos,'metadata':{'version':'v1.0-trainval'}},f)
    # Exact object annotation identities; no fabricated map-to-object correspondence.
    wanted={r['sample_token'] for r in frames}; raw=collections.defaultdict(list)
    for r in json_records(DATA/'v1.0-trainval/sample_annotation.json'):
        if r['sample_token'] in wanted: raw[r['sample_token']].append(r)
    write(OUT/'object_records.json',raw)
    write(labels/'manifest.json',dict(frames=frames,annotations=annotations,counts=dict(collections.Counter(r['split'] for r in annotations)),
        paired_counts=dict(collections.Counter(r['split'] for r in annotations if r['counterfactual_text'])),
        instance_definition='author-exact real map records; crossing two edges grouped per polygon; boundary omitted',
        independent_final_scope='adaptation holdout only; author pretraining overlap; pure unseen-pretraining generalization BLOCKED'))
    print('V3_COUNTS',collections.Counter(r['split'] for r in annotations),flush=True)

if __name__=='__main__': main()
