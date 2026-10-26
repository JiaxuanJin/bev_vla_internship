"""第三阶段交付：优先真实目标邻域语义/对象AP，逐项公开分母及证据限制。"""
from __future__ import annotations
import collections
import csv
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.stage2.stage3_metric_review import OUT,V3,write,sha,ratio,semantic_scores

BRANCHES=('original','v2','combined','joint_control','gradient_balanced','local_balanced')
LABELS={'original':'Original','v2':'V2冻结','combined':'V3冻结','joint_control':'等预算原损失对照','gradient_balanced':'梯度平衡','local_balanced':'平衡+邻域CE'}

def load(path):return json.loads(Path(path).read_text())
def pct(x):return f'{100*x:.6f}' if x is not None else 'N/A'
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])+'\n'

def strict_complete_distractors(strict_ids):
    source=load(OUT/'dev_new_predictions/predictions.json');pool=collections.defaultdict(collections.Counter)
    with (OUT/'complete_distractor_audit/per_distractor.csv').open() as f:
        for r in csv.DictReader(f):
            if r['instruction_id'] not in strict_ids:continue
            for suffix in ('all',r['target_class']):
                pool[(r['branch'],'semantic',suffix)].update({k:int(r[k]) for k in ('intersection','union','gt_pixels','predicted_pixels','roi_pixels')})
    for r in source['annotations']:
        if r['instruction_id'] not in strict_ids:continue
        with np.load(V3/'annotations'/r['frame_file']) as f:masks=f['masks'];classes=f['classes']
        with np.load(OUT/'labels'/(r['instruction_id']+'.npz')) as f:region=f['region']
        others=[masks[k]&region for k,c in enumerate(classes) if c==r['target_class'] and k!=r['selected_index'] and (masks[k]&region).any()]
        wrong=np.logical_or.reduce(others)&~masks[r['selected_index']] if others else np.zeros_like(region)
        with np.load(OUT/'dev_new_predictions'/r['prediction_file']) as f:
            for b in BRANCHES:
                if b+'_attention' not in f:continue
                a=f[b+'_attention'];pool[(b,'attention','all')].update(wrong_sum=float(a[wrong].sum()),total_sum=float(a.sum()),wrong_fp=int(((a>=.5)&wrong).sum()),wrong_pixels=int(wrong.sum()))
    result={}
    for b in BRANCHES:
        a=pool[(b,'attention','all')]
        result[b]=dict(distractor_neighborhood_semantic_IoU=semantic_scores(pool[(b,'semantic','all')]),
            distractor_by_target_class={c:semantic_scores(pool[(b,'semantic',c)]) for c in ('1','2')},
            wrong_instance_attention=dict(a),same_quadrant_wrong_instance_mass=ratio(a['wrong_sum'],a['total_sum']) if a else None)
    return result

def figures(folder,strict_ids,selected_sha):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    colors=ListedColormap(['#eeeeee','#dfa526','#3296bd','#8b68ae'])
    rows=load(OUT/'dev_new_predictions/predictions.json')['annotations'];lookup={(r['sample_token'],r['text']):r for r in rows};done=set();files=[]
    for row in rows:
        other=lookup.get((row['sample_token'],row.get('counterfactual_text')))
        if not other or row['instruction_id'] not in strict_ids or other['instruction_id'] not in strict_ids:continue
        key=tuple(sorted((row['instruction_id'],other['instruction_id'])))
        if key in done:continue
        done.add(key);pair=sorted((row,other),key=lambda r:r['qualifier']!='nearest')
        fig,axes=plt.subplots(2,5,figsize=(21,8),constrained_layout=True)
        for rr,r in enumerate(pair):
            cf=pair[1-rr]
            with np.load(OUT/'dev_new_predictions'/r['prediction_file']) as f:data={k:f[k] for k in f.files}
            with np.load(OUT/'dev_new_predictions'/cf['prediction_file']) as f:cp=f['joint_control']
            with np.load(OUT/'labels'/(r['instruction_id']+'.npz')) as f:roi=f['target_neighborhood'];target=f['target']
            panels=[('True map GT + fixed2m target neighborhood',data['gt'],colors,0,3),('V3 frozen semantic classes',data['combined'],colors,0,3),
                ('Step32 control semantic classes',data['joint_control'],colors,0,3),('Same-direction other-instance text classes',cp,colors,0,3),
                ('Step32 control Attention',data['joint_control_attention'],'magma',0,1)]
            for ax,(title,img,cmap,lo,hi) in zip(axes[rr],panels):
                ax.imshow(img,origin='lower',extent=(-30,30,-15,15),cmap=cmap,vmin=lo,vmax=hi,interpolation='nearest')
                x=-30+(np.arange(400)+.5)*.15;y=-15+(np.arange(200)+.5)*.15
                ax.contour(x,y,roi,levels=[.5],colors='black',linewidths=.6);ax.contour(x,y,target,levels=[.5],colors='#ef4545',linewidths=.7)
                ax.set_title(title,fontsize=9)
            changed=int(((data['joint_control']!=cp)&roi).sum())
            axes[rr,0].set_ylabel(r['qualifier']+f' | changed {changed}/{int(roi.sum())} local pixels')
        fig.suptitle(f"DEV sample={row['sample_token']} | {pair[0]['text']} / {pair[1]['text']}\nV3={load(V3/'train/selection.json')['selected']['sha256'][:12]} | control32={selected_sha[:12]} | semantic classes, not instance segmentation",fontsize=10)
        file=f'counterfactual_pair_{len(files):03d}.png';fig.savefig(folder/file,dpi=130);plt.close(fig);files.append(dict(file=file,instruction_ids=list(key),sample_token=row['sample_token']))
    return files

def main():
    report=OUT/'report';report.mkdir(exist_ok=False)
    # Export model provenance as part of the reproducible report, not only during delivery.
    model_rows=[]
    for branch,path in (('original',Path('/mnt/g/nus/bevformer_small_joint_author.pth')),
                        ('v2',Path(load(ROOT/'outputs/stage2_spatial_v2_20261026/train/selection.json')['selected']['checkpoint'])),
                        ('combined',Path(load(V3/'train/selection.json')['selected']['checkpoint']))):
        model_rows.append(dict(branch=branch,checkpoint=str(path),sha256=sha(path),role='author pretrained baseline' if branch=='original' else 'frozen or derived adapter'))
    for branch,r in load(OUT/'optimization/checkpoints.json').items():
        model_rows.append(dict(branch=branch,checkpoint=r['checkpoint'],sha256=sha(r['checkpoint']),role='frozen or derived adapter'))
    compact_source=load(OUT/'compact_inference/protocol.json')
    model_rows.append(dict(branch='joint_control_compact',checkpoint=compact_source['checkpoint'],sha256=compact_source['sha256'],role='frozen or derived adapter'))
    with (report/'checkpoint_sources.csv').open('x') as f:
        w=csv.DictWriter(f,fieldnames=list(model_rows[0]));w.writeheader();w.writerows(model_rows)
    frozen=load(OUT/'dev_frozen_metrics/metrics.json');new=load(OUT/'dev_new_metrics/metrics.json');strict=load(OUT/'strict_qualifier_audit/dev_metrics/metrics.json')
    qa=load(OUT/'strict_qualifier_audit/audit.json');obj=load(OUT/'dev_object_evaluation/metrics.json');diag=load(OUT/'gpu_diagnosis/diagnosis.json')
    compact=load(OUT/'compact_inference/protocol.json');latency=load(OUT/'compact_inference/latency/metrics.json');prototype_latency=load(OUT/'latency/metrics.json')
    selected=load(OUT/'optimization/selection.json');protocol=load(OUT/'protocol.json');coordinate=load(OUT/'coordinate_trace.json')
    valid={r['instruction_id'] for r in qa['annotations'] if r['split']=='dev' and r['status']=='PASS'}
    distractor=strict_complete_distractors(valid);write(report/'strict_complete_distractor_metrics.json',distractor)
    summary=[]
    for b in BRANCHES:
        m=strict['metrics'][b];a=m['attention'] or {}
        summary.append(dict(branch=b,target_neighborhood_semantic_iou=m['target_neighborhood_semantic_IoU']['iou'],
            target_pixel_recall=m['target_pixel_recall']['recall'],irrelevant_semantic_miou=m['instruction_irrelevant_semantic_mIoU']['foreground_miou'],
            instruction_quadrant_semantic_miou=m['instruction_quadrant_semantic_mIoU']['foreground_miou'],
            relevant_object_ap_2m_xy=obj['metrics'][b]['relevant']['macro_ap'],irrelevant_object_ap_2m_xy=obj['metrics'][b]['irrelevant']['macro_ap'],
            distractor_neighborhood_semantic_iou=distractor[b]['distractor_neighborhood_semantic_IoU']['iou'],
            target_attention_iou=a.get('target_attention_IoU'),target_attention_precision=a.get('target_attention_precision'),target_attention_recall=a.get('target_attention_recall'),
            background_attention_fp=a.get('background_fp'),background_pixels=a.get('background_pixels'),
            same_quadrant_wrong_instance_mass=distractor[b]['same_quadrant_wrong_instance_mass'],
            same_quadrant_wrong_instance_fp=distractor[b]['wrong_instance_attention'].get('wrong_fp'),
            counterfactual_attention_gap=m['counterfactual']['mean_target_attention_gap'],counterfactual_semantic_changed_pixels=m['counterfactual']['semantic_changed_pixels']))
    baseline=next(r for r in summary if r['branch']=='combined');control=next(r for r in summary if r['branch']=='joint_control')
    for r in summary:
        r['target_neighborhood_delta_vs_v3_pp']=(r['target_neighborhood_semantic_iou']-baseline['target_neighborhood_semantic_iou'])*100
        r['target_neighborhood_delta_vs_equal_budget_control_pp']=(r['target_neighborhood_semantic_iou']-control['target_neighborhood_semantic_iou'])*100
        r['relevant_ap_delta_vs_v3_pp']=(r['relevant_object_ap_2m_xy']-baseline['relevant_object_ap_2m_xy'])*100
    with (report/'comparison.csv').open('x') as f:w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    primary_table=table(['模型','目标邻域语义IoU%（63条）','Relevant对象AP%（64条）','无关象限语义mIoU%（63条）','目标邻域较等预算对照pp'],
        [[LABELS[r['branch']],pct(r['target_neighborhood_semantic_iou']),pct(r['relevant_object_ap_2m_xy']),pct(r['irrelevant_semantic_miou']),f"{r['target_neighborhood_delta_vs_equal_budget_control_pp']:+.6f}"] for r in summary])
    semantic_detail_table=table(['模型','目标实例像素语义Recall%','同类干扰邻域语义IoU%'],
        [[LABELS[r['branch']],pct(r['target_pixel_recall']),pct(r['distractor_neighborhood_semantic_iou'])] for r in summary])
    attention_table=table(['模型','目标Attention IoU%','Precision%','Recall%','同象限错误实例Mass%','背景AttentionFP','错误实例AttentionFP'],
        [[LABELS[r['branch']],pct(r['target_attention_iou']),pct(r['target_attention_precision']),pct(r['target_attention_recall']),pct(r['same_quadrant_wrong_instance_mass']),
          r['background_attention_fp'] if r['background_attention_fp'] is not None else 'N/A',r['same_quadrant_wrong_instance_fp'] if r['same_quadrant_wrong_instance_fp'] is not None else 'N/A'] for r in summary])
    (report/'comparison.md').write_text(primary_table)
    per_class=[]
    for b,m in strict['metrics'].items():
        for c,values in m['per_target_class'].items():per_class.append(dict(branch=b,target_class=int(c),**values['target']))
    with (report/'target_neighborhood_by_class.csv').open('x') as f:w=csv.DictWriter(f,fieldnames=list(per_class[0]));w.writeheader();w.writerows(per_class)
    logs=[]
    for p in sorted((OUT/'optimization').glob('*/training.jsonl')):
        rows=[json.loads(s) for s in p.read_text().splitlines()];assert len(rows)==32 and all(r['optimizer_step_executed'] and any(v['changed_elements'] for v in r['parameter_updates'].values()) for r in rows)
        logs.append(dict(branch=p.parent.name,optimizer_updates=len(rows),peak_allocated_mib=max(r['peak_allocated_mib'] for r in rows),
            active_parameter_numel=sum(v['numel'] for v in rows[0]['parameter_updates'].values() if v['gradient_l2'] is not None),
            first_task_gradients=rows[0]['task_gradients'],last_task_gradients=rows[-1]['task_gradients']))
    figures_dir=report/'figures';figures_dir.mkdir();images=figures(figures_dir,valid,selected['selected']['sha256'])
    assert len(images)==25
    train_probes=[r for r in diag['records'] if r['split']=='train' and r['branch']=='v3_frozen']
    norms={k:float(np.median([r['gradient']['unweighted_l2'][k] for r in train_probes])) for k in ('detection','segmentation','attention','target_neighborhood_ce')}
    ratios=[r['gradient']['weighted_l2']['attention']/r['gradient']['weighted_l2']['segmentation'] for r in train_probes]
    latency_rows=[]
    for b,scopes in latency['metrics'].items():
        for scope,t in scopes.items():latency_rows.append([b,scope,f"{t['mean_ms']:.3f}",f"{t['median_ms']:.3f}",f"{t['p95_ms']:.3f}",f"{t['peak_allocated_mib']:.3f}",t['iterations'],t['warmup']])
    timing_table=table(['分支','范围','均值ms','中位ms','p95ms','peak allocated MiB','次数','warmup'],latency_rows)
    result=dict(status='PASS',primary_protocol=dict(semantic_instructions=63,semantic_frames=8,semantic_scenes=1,
        object_instructions=64,object_frames=8,empty_object_prompts=obj['empty_prompts'],attention_threshold=.5,neighborhood_radius_m=2.,
        counterfactual_directed_pairs=50,counterfactual_unordered_pairs=25,selection_protocol='unchanged eligible65; strict63 audit is geometry-only, no re-selection'),
        comparison=summary,strict_semantic_metrics=strict,complete_distractor_metrics=distractor,object_metrics=obj,
        historical_eligible65_metrics=new,qualifier_audit=qa,diagnosis=diag,coordinate_trace=coordinate,training_logs=logs,total_optimizer_updates=96,
        selected=selected,latency=latency,prototype_latency=prototype_latency,compact_inference=compact,visualizations=images,
        statuses=dict(metrics='PASS',gpu_updates='PASS',gt_filter_and_matching='PASS',compact_exact_equivalence='PASS',
            perception_gain_from_gradient_balancing='NOT_VERIFIED: did not beat equal-budget control',new_channel_architecture='NOT_VERIFIED: zero-init diagnostic only, not trained',
            map_instruction_detection_ap='NOT_EVALUABLE',official_nuscenes_ap='NOT_EVALUABLE: custom AP only',GFLOPs='NOT_VERIFIED',
            strict_independent_generalization='BLOCKED: no new final inference; all opened holdouts excluded'),
        source_paths={name:str(path) for name,path in [('raw_target_counts',OUT/'strict_qualifier_audit/dev_metrics/per_instruction.csv'),('raw_counterfactual',OUT/'strict_qualifier_audit/dev_metrics/counterfactual.csv'),
            ('raw_distractor_counts',OUT/'complete_distractor_audit/per_distractor.csv'),('raw_object_predictions',OUT/'dev_object_evaluation/predictions.json'),('raw_object_matches',OUT/'dev_object_evaluation/matches.json')]})
    write(report/'metrics.json',result)
    payload=f'''# 第三阶段：指标定义与感知收益瓶颈（2026-10-26）

本轮先执行用户提供的只读审计，再从真实逐指令GT/预测建立物理邻域指标，完成训练/开发GPU根因实验和推理计时。**没有证据证明梯度平衡或邻域CE超过等预算对照；没有新增最终集精度或泛化结论。**

## 师傅考核的感知指标优先

{primary_table}

语义指标严格几何有效63条、8帧、1个开发场景；对象AP独立64条/同8帧，16条空目标，相关GT汽车53、行人224（按帧annotation计，不是277条独立物理轨迹）。语义/对象样本分母不同，绝不混成一项“实例分割检测精度”。上表63条是对旧候选协议做完整几何审计后的**附加严格结果**；模型仍按预先声明的65条开发协议选择joint_control，没有根据63条重选模型。
在原65条协议中，V2/V3/对照/梯度平衡/邻域CE的目标邻域语义IoU分别39.306777%/39.313206%/39.330321%/39.327591%/39.316041%；两套协议的结论一致。主表使用63条是为了真实指令对应，不是择优筛样本，所有65条原数值完整保留。

选中的是**原损失等预算对照**，不是新算法胜出。严格63条目标邻域IoU V3→对照 **39.303618%→39.321525%**，仅+0.017908pp；梯度平衡比对照−0.001428pp，邻域CE比对照−0.011269pp。Relevant对象AP V3→对照62.068678%→62.066851%（−0.001827pp），检测没有收益。没有为求正向数字继续增加步数。

## 指标语义、物理来源与分母

1. **instruction_quadrant_semantic_mIoU**：保留原Ego象限ROI，累计4×4 GT行/预测列混淆矩阵，再取有GT/预测并集的前景类IoU平均；nearest/farthest同象限共享ROI。它是语义指标，不能区分实例，不改名为实例分割。
2. **target_neighborhood_semantic_IoU**：以真实selected map record在指令象限的可见GT掩码为种子，原始0.15m网格做欧氏距离≤2m缓冲；只在原60m×30m窗口边界截断，不按预测或正确率改变区域。邻域内部GT取**目标类别的全部真实像素**，其他正确同类实例仍是语义正例。按每条指令累计二值intersection/union，再汇总相除；类1 Divider/类2 Crosswalk另列。衡量目标周围语义质量，**不等于语言条件实例分割IoU**。
3. **target_pixel_recall**：指定真实实例像素上，类别Head预测目标类的TP/实例GT像素；不同于Attention recall。邻域IoU则包含邻域内全部同类真值，两者分母清楚分开。
4. **distractor_neighborhood_semantic_IoU**：每个同象限、同类、非目标真实record也做相同2m缓冲，内部按真实类别评价；多个distractor邻域重叠时按“指令—record”重复计数，公开重复次数，绝不把正确distractor的存在算语义FP。complete_distractor_audit补入≥1像素的所有可见记录，主表的严格补充统计在report/strict_complete_distractor_metrics.json。
5. **target_attention_IoU/precision/recall**：固定阈值0.5，在完整200×400 GT物理窗口计算真实目标的交并集、TP/预测激活、TP/目标GT，不只裁ROI。同象限wrong-instance mass是这些真实其他record联合掩码上Attention总量/全窗口Attention总量，排除与selected完整record重叠的像素；背景FP仅指GT=0上的Attention阈值误激活。全部属于**实例级注意力定位**，不是语义Head输出的实例ID。

每个ROI公开sample token、真实instance id、类别、文本、物理半径、像素/面积、Ego/LiDAR目标中心、R/t及ego global pose：labels/manifest.json。GT200×400，x∈[-30,30]、y∈[-15,15]，像素中心步长0.15m；原生BEV150×150、LiDAR XY范围±51.2m，使用已校验标定grid_sample(align_corners=False)，不resize/翻转GT。每个指标均为指令重复权重，不是独立帧均值；8帧属于同一开发场景，不能代表稳定泛化。
分母/类别/逐条GT、预测、intersection、union见strict_qualifier_audit/dev_metrics/per_instruction.csv、counterfactual.csv、report/target_neighborhood_by_class.csv；目标标签空时明确NOT_EVALUABLE，不造零掩码/零精度。

{semantic_detail_table}

{attention_table}

上面Attention全图计数仍按指令重复；错误实例使用所有可见记录而非仅≥8像素候选，Mass分母是全窗口Attention总量。较低Mass不自动表示FP减少，必须同时看FP计数、precision和recall。模型权重来源及SHA详见report/checkpoint_sources.csv；逐条预测文件/掩码及SHA已写入per_instruction.csv。

## 完整实例覆盖发现的标签边界

旧V3候选仅保留≥8像素record。复核全部可见记录发现训练/开发各3个“指令—微小干扰实例”对应被旧候选门槛排除。完整几何规则下1条训练、2条开发指令不是唯一或不是所有可见记录中的farthest，明确标记NOT_EVALUABLE，详见strict_qualifier_audit/audit.json。完整旧65条与52条有向反事实不删除；附加严格版保留63条、50条有向/25对反事实。那1条训练侧无效指令**没有出现在本次固定32步序列中**，因此未被这96次更新使用。不能把“旧≥8候选集合内nearest/farthest”直接宣称为无条件所有可见实例的nearest/farthest。

## 同类、同象限实例反事实

主严格50条有向对照，V3目标Attention平均正确−错误差17.133268pp，对照19.245779pp；Attention胜出36/50。对照的语义Head仅改变314/135744个邻域像素（0.2313%，重复计数）；目标邻域语义IoU正确/错误为37.107235%/37.042493%，仅+0.064742pp，逐指令正确胜出20、打平20、失败10。不能用注意力热图大变化替代语义精度收益。
同时，**正确的全局类别图理想情况下本就可对nearest/farthest保持一致**：两条指令的全部语义GT相同。因此“图不变”本身也不证明grounding失败；它只说明当前全局语义Head没有条件实例输出，实例选择应由独立Attention指标约束。若项目真正要输出仅指定实例的mask，需要另定义有真实实例GT的条件实例Head，不能误罚其余正确同类类别。
全部25对严格有效对照逐对出图，未筛选有利样本；图中标明sample、文本和checkpoint。原始52条版本另外保留在dev_new_metrics，避免掩盖无效标签和历史可比性。

## Fusion梯度、Head响应和坐标链根因

六个训练场景各取第一条可配对指令，开发八帧各取第一条可配对指令；按元数据固定，不按性能选。无融合/V2冻结/V3冻结/零初始化通道诊断共56个真实CUDA分支前向，使用真实BEV缓存和作者双Head、冻结作者backbone/Heads及DistilBERT，诊断阶段optimizer.step=0。
准确率复算复用V2封存的训练流水线BEV缓存，包含当时固定seed的图像增强；各模型输入完全相同。它不是标准无增强nuScenes官方Val评价。端到端计时另用固定开发帧的作者测试流水线图像，三分支shape/输入一致，不能把两种缓存/流水线口径混用。
V3训练探针梯度L2中位值：Detection **{norms['detection']:.8f}**、全图Segmentation **{norms['segmentation']:.8f}**、Attention **{norms['attention']:.8f}**、目标邻域CE **{norms['target_neighborhood_ce']:.8f}**；同样本加权Attention/Segmentation范数比的中位数约 **{float(np.median(ratios)):.1f}倍**。所有未加权/加权范数、两两余弦、Head logits RMS、变化像素、BEV扰动在gpu_diagnosis/diagnosis.json；每次实际训练的梯度和参数增量在optimization/*/training.jsonl。
按训练探针中位||g_det+g_seg||/||g_att||、上限0.2，得到固定Attention权重 **{diag['proposed_attention_weight']:.12f}**，只由训练侧计算，没有用留出数值选权重。保留原感知损失；邻域CE按完整真实4类GT评价2m区域，不把其他实例涂成背景。
三分支同V3初始checkpoint、同32条随机更新顺序、相同AdamW(lr=3e-4,wd=.01)、clip5、相同固定alpha=0.0129582463，只有损失权重逐项变化。共**96次真实GPU optimizer.step**，每步参数变化非零，峰值allocated约{max(r['peak_allocated_mib'] for r in logs):.2f}MiB。均为**缓存真实BEV→原生双Head的反向**，不是图像backbone训练；另有完整六相机推理计时。没有增加alpha，也没有延长步数挑结果。
损失平衡修正了梯度主导比例，但开发任务精度没超过同预算对照，说明范数失衡不是已被证实的唯一根因。标量BEV乘性门控只改已有通道幅度，冻结Head与全局、指令无关的语义GT使条件实例能力不具可辨识性；零初始化通道支路只做过梯度诊断，**架构有效性NOT_VERIFIED**，未把它包装成训练成功模型。

只读执行作者实际BevFeatureSlicer及转置/flip的物理坐标ramp，56个训练/开发帧与Attention监督标定网格的平均采样位置差 **{coordinate['mean_sampling_offset_m']:.6f}m**，见coordinate_trace.json。它表明两条空间采样链存在结构差异；不能由此直接断言更换cropper能提高精度，因为作者冻结Head可能已适配原约定。本轮没有修改作者cropper、GT、权重或坐标约定。

## 检测输入、AP真实性与旧冻结最终结果

所有新Detection Loss逐样本恢复作者ObjectRangeFilter/ObjectNameFilter/limit_yaw，56帧与作者原生变换的box/label逐项一致；8个开发帧、每个6层作者Hungarian分配核验了正查询与过滤后GT的唯一几何中心及类别一致，输入/匹配证据gpu_diagnosis/gt_filter_equivalence.json、query_matching.json。没有沿用第三轮原缓存路径未过滤GT的问题。
本轮训练仍是地图语言+全局Detection Loss，没有新增对象指令监督，因此新对象AP是迁移评价，不是相关实例专用训练成果。64条开发对象指令都保留，包括16条空目标；逐类TP/FP/FN、全分数PR与真实annotation匹配在dev_object_evaluation/metrics.json、matches.json。
AP@2m_xy是**自定义非官方**：每条指令/帧按Ego象限分区、同类、分数降序匹配最近未匹配真实GT，LiDAR XY距离≤2m为TP；完整native top300预测、不设置信度阈值，AP=Σ每个TP处precision/GT数，汽车/行人有GT类宏平均。它不是固定阈值Precision，不是nuScenes多阈值/插值/截断/官方过滤AP。地图Divider/Crosswalk/Boundary没有对象3D box对应，地图指令AP始终NOT_EVALUABLE。
只读审计旧scene-0001冻结数组：32条对象指令22条空目标、汽车GT3/TP2/FP191/FN1、行人GT11/TP10/FP335/FN1，macro AP78.030303%；没有新的最终集前向，也没把开发AP替换该数字。旧目标Attention的TP/union/recall/precision和象限混淆矩阵从NPZ逐项复算与原报告完全一致，见frozen_array_audit/audit.json。

## P1：参数精简和真实推理时间

P0结束后，仅移除未启用的通道诊断包装器、闲置参数和零残差分配；核心alpha、Heads和Attention不变，**没有额外训练**。adapter参数532498→{compact['compact_parameters']}，移除{compact['removed_unused_parameters']}个闲置参数。65条地图+64条对象指令逐条比较真实BEV及三个原生Head输出张量，最大误差0，重新汇总语义/对象AP无精度代价；证据compact_inference/equivalence.json。精简checkpoint SHA：`{compact['sha256']}`，源自选中32步对照。GFLOPs未测，标记NOT_VERIFIED；没有声称减少作者Heads或训练了新通道架构。

以下为精简推理测量时段，同一RTX5080 Laptop、batch=1、6×3×736×1280 FP32图像、原生BEV[1,256,150,150]，同步+warmup+重复计时：

{timing_table}

缓存Head：文本向量与真实BEV已缓存，排除BERT/图像主干；完整六相机：含图像tensor scatter、backbone、BEV encoder、BERT/adapter（启用时）、双Head和decode，排除磁盘IO、GT构建与固定样本标定预计算。每次清空prevBEV避免时序缓存。Original计时前将BERT/adapter移到CPU；所有非当前帧训练缓存已从特征字典释放，峰值含实际模块/残留诊断Tensor驻留，不是fresh-process最小内存。
更早的诊断包装时段Original/new对照完整均值为273.082/286.932ms；本时段为314.745/338.311ms，Original也明显变慢，存在时段/负载漂移，**不能跨时段把变化归因于精简，更不能宣称已有延迟加速**。两时段原始每次耗时都保留。已证实的是参数减少、输出等价和上述实际延迟，尚未验证生产负载下稳定加速。

## 状态、边界与下一步

- PASS：逐指令2m真实邻域语义、Attention定位、完整干扰实例、几何严格反事实、原始计数与checkpoint追溯；实际CUDA梯度和96次参数更新；作者过滤/匹配；精简129条输出等价；两类同步推理计时。
- NOT_VERIFIED：梯度平衡/邻域CE的额外精度收益（实测未超过对照）；新语言通道架构效果；轻量化稳定延迟加速；GFLOPs；真正条件实例分割Head。
- BLOCKED：严格多场景独立泛化与新模型最终精度。scene-0553、scene-0001和旧10条指令均未用于本轮调参、训练、阈值选择；只有旧最终数组只读复核。没有把一个开发场景当稳定泛化。

当前证据支持继续把目标实例Grounding与全局语义感知目标分开定义，先修复候选可见性歧义，再在训练/开发侧验证真正的语言条件实例输出与可控通道残差；也应在保留作者预训练约定下单独检验Fusion到Head的空间映射。上述后续方案尚未执行，不宣称有效。

## 文件与完整复现

主输出`{OUT}`；报告JSON/CSV见report/metrics.json、comparison.csv；物理邻域逐条manifest与NPZ见labels/；原65条历史可比、严格63条/50反事实、全可见干扰分别有独立目录。optimization三个分支各step001/032 checkpoint，SHA见checkpoints.json；精简推理权重见compact_inference/fusion_compact_step032.pt。图像report/figures/counterfactual_pair_000.png至024.png覆盖全部25对。
实际GPU日志：gpu_diagnosis_run.log、gpu_training_run.log、gpu_evaluation_latency_run.log、gpu_compact_inference_run.log；硬件hardware.json。每条指标可追溯source prediction NPZ、metric label NPZ、sample/instance、阈值及checkpoint。3条有限单测仅核查语义GT包含其他正确同类实例、米制欧氏邻域、汇总交并分母，不能替代算法指标。

```bash
cd /home/jjx2003520/projects/bev_vla
bash outputs/stage3_metric_review_20261026/reproduce_stage3.sh outputs/stage3_metric_review_reproduce_20261026
```

复现脚本拒绝已有输出目录，所有实验写入新的BEV_STAGE3_OUTPUT；依赖冻结前三轮资产、作者Torch/MMCV环境、本地DistilBERT和nuScenes。原CPU数组指标复算可单独执行scripts/stage2/stage3_metric_review.py各phase；完整GPU命令在复现脚本中逐步列出。前三轮历史报告、权重、GT与其他未跟踪文件保持原样；当天Logbook仅追加一份，Git白名单另列，未git add/commit/push。
'''
    (report/'中文分析报告.md').write_text(payload)
    print(primary_table);print(timing_table)

if __name__=='__main__':main()
