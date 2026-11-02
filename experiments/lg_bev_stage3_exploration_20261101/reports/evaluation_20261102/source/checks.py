"""Independent checks using installed SDK and explicit pixel counts."""
import numpy as np
from eval_metrics import DetectionAP, SegmentationIoU, quadrant, segmentation_regions
from core import require


def check_records(records):
    from nuscenes.eval.common.data_classes import EvalBoxes
    from nuscenes.eval.common.utils import center_distance
    from nuscenes.eval.detection.algo import accumulate, calc_ap
    from nuscenes.eval.detection.data_classes import DetectionBox
    checks = 0
    for row in records:
        if row['group']=='a0':
            continue
        for control,prediction in row['predictions'].items():
            if row['task']=='object':
                cls='car' if row['target_class']==0 else 'pedestrian'
                metric=DetectionAP((cls,))
                metric.add(row['query_id'],prediction['gt_boxes'],prediction['pred_boxes'],cls,row['fb'],row['lr'])
                result=metric.result()
                for region in ('relevant','irrelevant'):
                    gt,pred=EvalBoxes(),EvalBoxes()
                    def boxes(items,truth):
                        chosen=[]
                        for b in items:
                            if b['class']!=cls:
                                continue
                            x,y=b['xy']
                            inside=(x>=0 if row['fb']=='front' else x<0) and (y>=0 if row['lr']=='left' else y<0)
                            if inside != (region=='relevant'):
                                continue
                            chosen.append(DetectionBox(sample_token='q',translation=(x,y,0),size=(1,1,1),rotation=(1,0,0,0),velocity=(0,0),detection_name=cls,detection_score=-1.0 if truth else float(b['score']),attribute_name='',num_pts=1))
                        return chosen
                    truths=boxes(prediction['gt_boxes'],True)
                    forecasts=boxes(prediction['pred_boxes'],False)
                    gt.add_boxes('q',truths)
                    pred.add_boxes('q',forecasts)
                    value=result[region]['classes'][cls]['2.0']
                    require(value['GT']==len(truths) and value['TP']+value['FP']==len(forecasts),'Detection raw counts disagree')
                    if truths:
                        native=calc_ap(accumulate(gt,pred,cls,center_distance,2.0),0.1,0.1)
                        require(abs(value['AP']-native)<1e-12,'SDK AP@2m mismatch')
                    else:
                        require(value['AP'] is None and value['FP']==len(forecasts),'Empty GT handling mismatch')
                    checks+=1
            else:
                metric=SegmentationIoU()
                metric.add(row['query_id'],row['gt'],prediction['semantic'],row['target'],row['valid'],row['target_class'])
                regions=segmentation_regions(row['target'],row['valid'])
                require(np.array_equal(regions['relevant']|regions['irrelevant'],row['valid']),'Region union mismatch')
                require(not (regions['relevant']&regions['irrelevant']).any(),'Region overlap')
                for region,mask in regions.items():
                    manual=np.array([[np.count_nonzero(mask&(row['gt']==i)&(prediction['semantic']==j)) for j in range(4)] for i in range(4)])
                    require(np.array_equal(manual,metric.matrices[region,row['target_class']]),'Confusion mismatch')
                    checks+=1
    return checks
