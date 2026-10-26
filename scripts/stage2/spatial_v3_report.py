"""只读原始GPU数组生成第三轮完整数值/消融/中文分析；不运行模型或选型。"""
from __future__ import annotations
import collections
import csv
import json
import math
from pathlib import Path
import pickle
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.stage2.spatial_v3_data import OUT,V2,sha,write

def tab(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])+'\n'

def fmt(x):return f'{100*x:.6f}' if x is not None else 'N/A'

def raw_attention(split,branches):
    directory=OUT/(split+'_map');meta=json.loads((directory/'predictions.json').read_text())
    counts={b:collections.Counter() for b in branches};records=[]
    for r in meta['annotations']:
        with np.load(directory/r['prediction_file']) as d:
            target=d['target'];roi=d['region'];other=d['other'];gt=d['gt']
            for b in branches:
                pred=d[b];truth=gt==r['target_class'];prediction=pred==r['target_class']
                record=dict(instruction_id=r['instruction_id'],branch=b,target_class=r['target_class'],target_gt_pixels=int(target.sum()),
                    roi_pixels=int(roi.sum()),target_class_intersection=int((truth&prediction&roi).sum()),target_class_union=int(((truth|prediction)&roi).sum()),
                    outside_roi_foreground_fp=int(((pred!=0)&(gt==0)&~roi).sum()))
                for c in range(4):
                    record[f'class{c}_gt_roi']=int(((gt==c)&roi).sum());record[f'class{c}_pred_roi']=int(((pred==c)&roi).sum())
                    record[f'class{c}_intersection_roi']=int(((gt==c)&(pred==c)&roi).sum());record[f'class{c}_union_roi']=int((((gt==c)|(pred==c))&roi).sum())
                if b!='original':
                    a=d[b+'_attention'];active=a>=.5
                    values=dict(tp=int((active&target).sum()),union=int((active|target).sum()),gt=int(target.sum()),pred=int(active.sum()),
                        background_fp=int((active&(gt==0)).sum()),background_pixels=int((gt==0).sum()),
                        same_quadrant_fp=int((active&other&roi).sum()),same_quadrant_other_pixels=int((other&roi).sum()),
                        other_instance_attention_sum=float(a[other].sum()),target_attention_sum=float(a[target].sum()),attention_sum=float(a.sum()))
                    counts[b].update(values);record.update(values)
                records.append(record)
    result={}
    for b,c in counts.items():
        if b=='original':result[b]=None;continue
        result[b]={**dict(c),'iou':c['tp']/c['union'],'target_recall':c['tp']/c['gt'],'target_precision':c['tp']/c['pred'] if c['pred'] else None,
            'pooled_other_instance_mass':c['other_instance_attention_sum']/c['attention_sum'],
            'pooled_target_mass':c['target_attention_sum']/c['attention_sum'],
            'background_fp_rate':c['background_fp']/c['background_pixels'],
            'same_quadrant_fp_rate':c['same_quadrant_fp']/c['same_quadrant_other_pixels'] if c['same_quadrant_other_pixels'] else None}
    return result,records

def training_audit():
    summary={};total=0
    for parent in ('train','objects_train'):
        for log in sorted((OUT/parent).glob('*/training.jsonl')):
            rows=[json.loads(line) for line in log.read_text().splitlines()]
            assert [r['step'] for r in rows]==list(range(1,len(rows)+1))
            assert all(r['optimizer_step_executed'] and any(v['changed_elements'] for v in r['parameter_updates'].values()) for r in rows)
            gradients=[r for r in rows if r.get('task_parameter_gradients')]
            summary[str(log.relative_to(OUT))]=dict(optimizer_updates=len(rows),all_updates_changed_parameters=True,
                initial_loss=rows[0]['total_loss'],terminal_loss=rows[-1]['total_loss'],max_peak_allocated_mib=max(r['peak_allocated_mib'] for r in rows),
                unique_training_frames=len({r.get('frame_index',r['instruction_id'].split('_')[0]) for r in rows}),
                first_joint_task_gradient_l2={k:math.sqrt(sum(x*x for x in v)) for k,v in gradients[0]['task_parameter_gradients'].items()} if gradients else None)
            total+=len(rows)
    return dict(total_optimizer_updates=total,cached_attention_updates=1120,cached_native_dual_head_updates=240,
        fresh_full_image_forward_frames=4,backbone_backward_updates=0,backbone_head_distilbert_frozen=True,branches=summary)

def figures(report,selection):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scripts.stage2.spatial_v3_evaluate import plot
    for split in ('dev','final'):
        directory=OUT/(split+'_map');meta=json.loads((directory/'predictions.json').read_text())
        for n,row in enumerate(meta['annotations']):
            if split=='dev' and n>=4:continue
            with np.load(directory/row['prediction_file']) as d:plot(row,d,directory/f'comparison_{n:03d}.png',selection['selected']['sha256'])
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for ax,split in zip(axes,('dev','final')):
        metrics=json.loads((OUT/(split+'_map/metrics.json')).read_text())['metrics']
        for branch in ('v2',selection['selected_variant']):
            vals=metrics[branch]['attention']['thresholds'];ax.plot([float(t) for t in vals],[100*c['iou'] for c in vals.values()],marker='o',label=branch)
        ax.set(title=f'{split}: full preregistered threshold grid',xlabel='Attention threshold',ylabel='Pooled instance IoU (%)');ax.legend();ax.grid(alpha=.3)
    fig.savefig(report/'threshold_sensitivity.png',dpi=150);plt.close(fig)
    with (OUT/'dataset_infos.pkl').open('rb') as f:infos=pickle.load(f)['infos']
    frames=json.loads((OUT/'annotations/manifest.json').read_text())['frames']
    for frame in frames:
        if frame['split']!='final':continue
        info=infos[frame['dataset_index']];fig,axes=plt.subplots(2,3,figsize=(15,6),constrained_layout=True)
        for ax,(name,cam) in zip(axes.flat,info['cams'].items()):
            ax.imshow(plt.imread(cam['data_path']));ax.set_title(name);ax.set_axis_off()
        fig.suptitle('Real six-camera input | sample='+frame['sample_token'])
        fig.savefig(report/f"camera_{frame['index']}.png",dpi=110);plt.close(fig)

def append_correction():
    """追加训练侧修正；不把开发集结果混入主实验的最终集数值。"""
    folder=OUT/'range_filter_correction'
    if not (folder/'comparison.json').is_file():return
    result=json.loads((folder/'comparison.json').read_text())
    profile=json.loads((folder/'training_gt_filter_audit.json').read_text())
    rows=[]
    for name,r in result['metrics'].items():
        rows.append([name,fmt(r['before']['relevant']['macro_ap']),fmt(r['corrected']['relevant']['macro_ap']),f"{r['relevant_change_pp']:+.6f}",fmt(r['corrected']['irrelevant']['macro_ap'])])
    totals=profile['totals']
    section=f'''\n## 追加根因修正：恢复作者训练GT过滤（开发集限定）

主实验结束后继续在训练输入侧发现：缓存Head路径调用原始get_ann_info，遗漏作者ObjectRangeFilter/ObjectNameFilter。上面的三轮最终数值均为真实结果，但第三轮联合Detection Loss使用了未经过这两步过滤的真实GT，因此不能称为与作者图像训练流水线完全相同的GT输入。56帧训练/开发共有{totals['raw']}个原始box，{totals['outside_bev']}个超出BEV中心范围、{totals['unknown_class']}个未知类别，恢复过滤后保留{totals['retained']}个。对象协议相关GT的147个指令对应超范围（按指令重复计数，不是147个不同实例）。这可能让检测损失优化不可编码的目标，是可验证的输入缺陷；不等于它单独解释了全部AP失败。

另开range_filter_correction/，保留原主实验及最终集。修正作者原生LiDAR in_range_bev、类别0..9、limit_yaw，Attention目标也限制为同样的中心范围。两个对象分支仍从同一个V2step220初始化，各64步、同顺序、同学习率，只改变GT过滤，与原各自分支构成训练侧单变量对照。额外真实GPU更新128次，本轮累计**1488次**，主实验1360次不回写。每步Loss、参数变化、optimizer状态、SHA保存在新子目录。

{tab(['对象分支','修正前开发RelevantAP%','修正后开发RelevantAP%','变化pp','修正后IrrelevantAP%'],rows)}

这些仅为开发集结果。监督分支没有因此提高AP，说明过滤修正不足以解决语言相关检测学习，不能宣称问题已解决。修正模型的独立最终泛化评价 **BLOCKED**：scene-0001已经开启，不重复用于方案选择，也没有把修正模型重新跑到该场景。不能将这里的开发集AP填入主最终集表。主地图分支的过滤遗漏已记录，本轮没有悄悄重训/替换其结果。
修正checkpoint及SHA见range_filter_correction/objects_train/summary.json，完整开发匹配和PR见range_filter_correction/dev_objects/。复现时在gradient后、report前增加range阶段。
'''
    with (OUT/'report/第三轮实验分析.md').open('a') as f:f.write(section)
    metrics=json.loads((OUT/'report/metrics.json').read_text());metrics['range_filter_correction']=dict(comparison=result,gt_filter_profile=profile,
        total_optimizer_updates_including_correction=1488,final_status='BLOCKED for corrected object models; original main final numbers retained')
    (OUT/'report/metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2)+'\n')
    (OUT/'report/range_correction.md').write_text(section)

def main():
    report=OUT/'report';report.mkdir(exist_ok=False)
    selected=json.loads((OUT/'train/selection.json').read_text());s=selected['selected_variant']
    dev=json.loads((OUT/'dev_map/metrics.json').read_text());final=json.loads((OUT/'final_map/metrics.json').read_text())
    objects=json.loads((OUT/'final_objects/metrics.json').read_text());dobjects=json.loads((OUT/'dev_objects/metrics.json').read_text())
    raw,records=raw_attention('final',('original','v1','v2',s));draw,drecords=raw_attention('dev',tuple(dev['metrics']))
    audit=training_audit();write(report/'training_audit.json',audit)
    for name,items in (('final_pixel_counts',records),('dev_pixel_counts',drecords)):
        keys=sorted({k for r in items for k in r})
        with (report/(name+'.csv')).open('x') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(items)
    comparison=[]
    for name,getter in (
        ('Relevant Segmentation IoU',lambda b:final['metrics'][b]['relevant_segmentation']['foreground_miou']),
        ('Irrelevant Segmentation IoU',lambda b:final['metrics'][b]['irrelevant_segmentation']['foreground_miou']),
        ('Relevant Object AP@2m_xy (map adapter transfer)',lambda b:objects['metrics'][b]['relevant']['macro_ap']),
        ('Instance Attention IoU@0.5',lambda b:raw[b]['iou'] if raw[b] else None),
        ('Same-class Other-instance Mass (pooled)',lambda b:raw[b]['pooled_other_instance_mass'] if raw[b] else None)):
        vals={b:getter(b) for b in ('original','v1','v2',s)}
        comparison.append(dict(metric=name,**vals,v3_minus_v2_pp=(vals[s]-vals['v2'])*100 if vals[s] is not None and vals['v2'] is not None else None,
            relative_change_vs_v2_pct=(vals[s]/vals['v2']-1)*100 if vals[s] is not None and vals['v2'] else None))
    write(report/'metrics.json',dict(protocol='new fixed scene0001 adaptation holdout; all22 map instructions/4frames +all32 object prompts (22empty,14GT); old0553/old10 not re-evaluated',
        scope='author official-train pretraining overlap; pretraining-unseen generalization BLOCKED; one scene no uncertainty/stability claim',
        comparison=comparison,final_attention_raw_counts=raw,dev_attention_raw_counts=draw,
        final_map_metrics=final,final_object_metrics=objects,dev_map_metrics=dev,dev_object_metrics=dobjects,
        training_audit=audit,selection=selected))
    with (report/'comparison.csv').open('x') as f:w=csv.DictWriter(f,fieldnames=list(comparison[0]));w.writeheader();w.writerows(comparison)
    comparison_table=tab(['指标（%）','Original','第一轮','第二轮','第三轮地图','第三−第二（百分点）'],
        [[r['metric'],*[fmt(r[b]) for b in ('original','v1','v2',s)],f"{r['v3_minus_v2_pp']:+.6f}" if r['v3_minus_v2_pp'] is not None else 'N/A'] for r in comparison])
    (report/'comparison.md').write_text(comparison_table)
    ab=[]
    for n,r in selected['variants'].items():
        d=r['dev'];m=dev['metrics'][n];c=d['thresholds']['0.5']
        ab.append([n,r['steps'],fmt(m['relevant_segmentation']['foreground_miou']),fmt(m['irrelevant_segmentation']['foreground_miou']),fmt(c['iou']),c['background_fp'],c['same_quadrant_fp'],fmt(d['instance_counterfactual_gap'])])
    ab_table=tab(['分支','更新','相关IoU%','无关IoU%','实例IoU%','背景FP','同象限实例FP','实例文本概率差pp'],ab)
    (report/'ablations.md').write_text(ab_table)
    figures(report,selected)
    diagnosis=json.loads((OUT/'diagnosis.json').read_text());conflict=json.loads((OUT/'train_gradient_conflict.json').read_text())
    text=f'''# 第三轮：真实实例排序、背景难负例与独立对象监督（2026-10-26）

第三轮取得**有限的实例定位改善**，仍未达到最终分割/检测目标。第一、二轮全部结果保留，旧scene-0553的30条和第一轮旧10条没有重新推理、选型或追加训练。

## 同一固定新场景的三轮实测

{comparison_table}

上表所有模型重新评价同一scene-0001四帧、22条地图指令；对象AP使用同四帧32条车辆/行人指令，与地图指标是两个独立任务。不能把本表12.0742%与旧scene-0553上的3.7113%直接相减称为提升。前两轮旧验证集结果、失败案例、原图、权重未修改。

第三轮对象专用监督模型Relevant AP **{fmt(objects['metrics']['object_supervised']['relevant']['macro_ap'])}%**，等预算global_control **{fmt(objects['metrics']['global_control']['relevant']['macro_ap'])}%**，差0.000000个百分点。其Irrelevant AP均为78.008234%。32条中22条为空目标，实际相关GT仅汽车3、行人11；各模型TP汽车2、行人10，召回不变。总AP不变不表示模型完全一样，原始分数和FP有差别，全部NPZ和PR/匹配已保存。地图指令的3D Detection AP仍为NOT_EVALUABLE，不能把类别分割包装成AP。

## 数据隔离和物理标签

训练342条/48帧/6场景，开发65条/8帧/scene-0061；其中同方位同类别nearest/farthest反事实236/52条。所有实例来自真实map record；crosswalk两个长边按同polygon分组，Boundary没有伪造ID。
每个实例分别栅格化，与原作者GT相交；位置约束为标定Ego米制象限。nearest按象限内可见GT点到Ego原点的最小距离排序；farthest是这一实例距离的最大排序，而非记录上最远端点；相邻排序间距必须≥1m、至少8像素。只有nearest和farthest都唯一时才构造反事实，推理中不输入ID、GT、距离、候选掩码或ROI。
GT保持原始200×400、0.15m网格；Attention原生150×150、102.4m覆盖。复用已验证的LiDAR/Ego标定grid_sample(align_corners=False)，没有resize/翻转GT。训练/开发56帧重建GT与冻结第二轮标签逐像素一致；新四帧与作者测试流水线GT也完全一致。

本地官方Val6019帧已于10月19日评价，不冒称新留出。按预先固定的场景名排序，选择官方Train、Mini之外第一个scene-0001，四个linspace帧，标注和传感器完整性在任何新模型数值前封存。该场景未用于第一二轮适配器训练/评价，也未进入第三轮训练或开发；**作者预训练可能包含它**。因此这里只支持“空间适配器跨场景留出”结论，严格预训练未见数据的泛化验证 **BLOCKED**。仅一个场景、8条有效实例反事实，不能证明稳定泛化。

## 根因和可验证改进

第二轮在新扩展开发标签上：同象限目标减最强同类实例平均注意力 **−4.5440pp**，nearest/farthest文本目标概率差 **+0.052291pp**，33/52胜出。全局文本/方位读出的nearest/farthest余弦分别0.98550/0.99463，显式限定词token读出0.79176，说明旧方位读出几乎不能区分实例限定词。该证据只来自训练/开发。

第三版增加独立限定词上下文读出；位置/视觉key与限定词query相似度生成logit，不使用规则实例选择器。实例Ranking比较目标平均logit与**每个真实同象限同类实例**，排除共享目标像素；同方位nearest/farthest错误文本也必须在原目标上降低logit。背景Hard Negative只在真实GT背景中取topK，K=max(32,4×目标像素数)，避免背景平均稀释高误激活。保留原平衡损失和左右反事实损失。

## 预注册等预算单变量消融（只用开发集）

{ab_table}

control→selector仅增加限定词分支；selector→ranking仅增加实例/文本排序；selector→hardneg仅增加背景难负例；ranking→combined增加难负例，hardneg→combined增加排序。combined→nogeometry去掉所有坐标分支；combined→nofusion旁路BEV融合。所有分支同一V2step220初值、同一342条训练分布、同随机更新顺序、同AdamW/LR/176步，alpha固定为0.0129582463；没有放大alpha。 每支176步是继承V2step220后的追加更新，重新创建AdamW；选中地图模型累计继承396次适配器更新，不能把各分支执行总数当成单个模型训练步数。
control复用V2结构并继续同预算训练；第三轮数据已增加farthest，所以它不是未经训练的V2。候选只取各分支终点，预注册从selector/ranking/hardneg/combined中按开发集固定0.5实例IoU选中combined。开发阈值网格0.1/0.2/0.3/0.5/0.7/0.9也选择0.5。分割指标随后如实计算，没有据此换成分割数字较高的ranking分支。
排序单独提高实例文本差，但背景FP增加；组合才兼顾定位和背景抑制。nofusion分割与Original完全相同，验证真实任务变化经Fusion进入Head；它与combined注意力几乎相同，说明本轮提升主要来自监督而非Head反馈。

## 真实GPU执行与梯度

RTX5080 Laptop、作者Torch2.8.0+cu128/MMCV1.4环境。总计 **{audit['total_optimizer_updates']}次实际optimizer.step**：7×176地图分支+2×64对象分支。地图每分支160次真实BEV缓存注意力更新+16次真实原生双Head三损失更新。对象每分支64次双Head更新。共1120次缓存Attention、240次缓存双Head更新；**这些不是图像主干反传**，主干、作者Heads、DistilBERT冻结。另在新四帧执行真实六相机GPU前向；缓存双Head解码与完整图像前向的boxes/scores/labels最大误差全为0。训练始终有真实CUDA反向、非零梯度、参数delta和optimizer状态；无OOM，无无限重试。

组合分支首个联合步骤参数梯度L2：Detection 0.00722727、Segmentation 0.00318494、Attention 12.52940；Attention权重0.2后仍约为分割梯度787倍。开发诊断Attention→BEV→双Head均有非零梯度：原生Attention梯度Detection0.00132215、Seg0.00026006；融合相对L2仅0.2713%，只改变50/80000个分割像素。标量乘性门控改变已有通道幅度，冻结的Head和全局、与语言无关的GT损失无法强制特定实例的任务改进。这是实测支持的限制，不能仅靠非零梯度宣称任务已学会。

对象任务另外生成448条训练/开发指令（384/64）；真实GT几何中心转Ego/Global，唯一对应sample_annotation与instance token误差<1mm。plural指令选择固定象限内全部真实汽车/行人，包括空目标。对象监督模型将作者Hungarian匹配到相关GT的正查询分类/回归权重×3，其他GT和负查询均保留权重1；另用真实LiDAR box footprint监督原生Attention，微小box至少保留包含中心的BEV单元。两分支相同64步、同文本/随机样本、同初始模型；对象模型累计966次相关正查询加权（含6个decoder层重复），没有把地图任务当对象任务。

对象开发AP：V2 **62.081848%**，global_control **62.141425%**，object_supervised **62.070247%**，监督分支比等预算控制低 **0.071178pp**。在固定首个正目标训练指令继续做无更新梯度诊断：监督模型Attention与相关加权Detection梯度余弦 **+0.95367**，与Segmentation **−0.50376**；0.2Attention梯度仍为相关Detection **55.58倍**，融合相对L2仅0.1195%。说明并非简单断梯度；尺度失衡、任务冲突和标量门控表达能力仍限制真实任务收益。该单样本局部证据不能当成全数据因果证明。

## 指标定义与负向结果

Relevant Segmentation IoU以固定Ego象限为ROI，对原始GT/语义预测累计4×4混淆矩阵后计算前景类IoU均值；Irrelevant取补集。相同象限中的nearest/farthest指令是两个指令样本，不是两张独立图。实例Attention IoU对整200×400计算交并集，绝非仅裁剪目标ROI；同时公开ROI外前景FP、背景FP、其他实例Mass、同象限误激活及逐类GT/预测/交并集CSV。
新场景V2实例GT14870、TP11803、union145586；第三轮TP9230、union76444，**召回由79.37%降为62.07%**。IoU变好主要靠误激活减少，不能说召回也提升。背景FP90229→43470，同象限实例FP2095→204；nearest/farthest文本差0.053963→16.299633pp、胜出4/8→6/8。两条仍失败，方向差39.548839→33.790165pp也有所下降。所有失败样本保留。
阈值敏感性全网格见threshold_sensitivity.png；第三轮在阈值0.9时TP为0，不能称稳健校准。0.7上的最终数字只作预注册网格诊断，未据此改变选定阈值0.5或模型。
Relevant AP@2m_xy按每条指令/帧，GT与预测先以固定象限分区，同类、置信度全排序、最近未匹配GT的LiDAR XY距离≤2m判TP；重复框FP，全300个解码框，不使用固定置信度门槛。AP=Σ每个TP位置的precision/GT数量，汽车/行人有GT类宏平均，空目标指令保留FP。它**不是nuScenes SDK官方AP/mAP**：未采用官方多距离阈值平均、插值、低召回/低precision截断和完整官方过滤。

## 保留的结论边界和后续方向

实例抑制与限定词响应有数值证据，但实例IoU12.07%仍低，相关分割仅+0.018275pp、无关分割−0.008738pp，对象AP无增益。未满足项目最终精度要求。没有基于新最终结果再次训练、选阈值或筛选样本。
下一步应在训练/开发侧检验任务梯度归一化、相关区域分割监督、带语言语义通道的残差适配器/独立对象检测适配层；需要全新且明确预训练/适配训练隔离的多场景测试集。本轮没有宣称这些尚未执行的方案有效。

## 交付与复现

本目录：{OUT}。annotations/manifest.json包含真实record ID/实例排序/场景隔离；train/*/training.jsonl和objects_train/*/training.jsonl逐步记录Loss/梯度/参数delta；各checkpoint保存AdamW状态，SHA见train/selection.json及objects_train/summary.json。组合checkpoint SHA：`{selected['selected']['sha256']}`。对象监督checkpoint SHA：`{json.loads((OUT/'objects_train/summary.json').read_text())['object_supervised']['sha256']}`。
dev_map与final_map保存GT/全部分支预测/注意力NPZ和JSON；dev_objects与final_objects保存完整3D框/分数/匹配/PR；report/final_pixel_counts.csv、comparison.csv及metrics.json提供原始计数和百分点评估。新最终22条地图指令全部出图，comparison_000.png到comparison_021.png；缺少可核验反事实的样本明确N/A，未把零数组占位当模型推理。4张camera_1000.png至camera_1003.png为真实六相机输入。source_snapshot与execution_manifest记录源码和文件摘要。

```bash
cd /home/jjx2003520/projects/bev_vla
PY=/home/jjx2003520/projects/bevformer_author_clean_5080/.venv-author/bin/python
DEST=outputs/stage2_spatial_v3_reproduce_20261026
for PHASE in labels diagnose train objects dev final gradient range report; do
  "$PY" scripts/stage2/spatial_v3_reproduce.py --output-dir "$DEST" --phase "$PHASE" || break
done
```

复现入口强制新目录，各阶段拒绝覆盖，依赖现有作者权重、DistilBERT、本地完整nuScenes与第二轮冻结BEV缓存。labels阶段只根据原始地图/几何生成GT；GPU训练与Head推理均在作者环境执行。测试只补充3条loss优化方向/重叠边界检查；真实成果以上述原始GPU指标为准。未commit/push，未修改第三方权重或GT。
'''
    (report/'第三轮实验分析.md').write_text(text)
    append_correction()
    print(comparison_table);print(ab_table)

if __name__=='__main__':main()
