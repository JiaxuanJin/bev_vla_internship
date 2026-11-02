import numpy as np
import pytest
from pathlib import Path
from eval_metrics import quadrant, DetectionAP
from eval_driver import queries, detection_records
from core import read, selected_rows, SHARED

def test_rear_alias_and_zero_ownership():
    xy=[[-1,-1],[-1,1],[0,0],[1,-1]]
    for lr in ('left','right'):
        assert np.array_equal(quadrant(xy,'rear',lr),quadrant(xy,'back',lr))
    assert sum(quadrant(xy,fb,lr).astype(int) for fb in ('front','rear') for lr in ('left','right')).tolist()==[1]*4

@pytest.mark.parametrize('distance,expected',[(1.999,1),(2.0,0),(2.001,0)])
def test_exact_two_meter_threshold(distance,expected):
    m=DetectionAP(('car',))
    m.add('q',[dict(id='1',xy=[1,1],**{'class':'car'})],[dict(xy=[1+distance,1],score=.9,**{'class':'car'})],'car','front','left')
    r=m.result()['relevant']
    assert r['classes']['car']['2.0']['TP']==expected
    assert r['AP@2m_xy']==r['classes']['car']['2.0']['AP']

def test_real_training_diagnostic_queries_and_coordinates():
    from scripts.stage3.lg_bev_engineering_data import LabelStore
    from pyquaternion import Quaternion
    original=Path(__file__).resolve().parents[3]
    cfg=read(original/'configs/main.json')
    rows=selected_rows(cfg,'diagnostic')
    chosen=[r for i,r in enumerate(rows) if i%32<2]
    assert len(chosen)==8 and all(r['split']=='train' for r in chosen)
    labels=LabelStore(read(SHARED/'runs/formal_gt_20261027_v1/data_report.json'))
    rear_count=0
    for i,row in enumerate(chosen):
        label=labels.get(row['sample_token'])
        for q in queries(label,i):
            assert q['fb'] in ('front','rear')
            rear_count+=q['fb']=='rear'
            assert quadrant([],q['fb'],q['lr']).size==0
            truth,_=detection_records(label,dict(boxes=[],labels=[],scores=[]),q['instruction_id'])
            raw={t:b for t,b in zip(label['eval_annotation_tokens'],label['eval_boxes'].numpy())}
            for b in truth:
                point=raw[b['id']][:3].astype(float).copy()
                point[2]+=raw[b['id']][5]/2
                expected=Quaternion(label['frame']['lidar2ego_rotation']).rotate(point)+np.array(label['frame']['lidar2ego_translation'])
                assert np.allclose(expected[:2],b['xy'],rtol=0,atol=1e-10)
    assert rear_count>0
