# Step 35：方向消融的退化帧与训练/推理坐标链静态审计

## 1. Step34 全量结果（累计像素 IoU）

- 验证帧数：6019；覆盖场景：150。
- 原始前景 mIoU：4.6416%。
- CCW90 前景 mIoU：19.6000%。
- 变化：+14.9584 个百分点。
- 逐帧改善 5867 / 退化 150 / 不变 2。

## 2. 例外场景（平均逐帧 mIoU 差值最低）

注意：本节是 **逐帧 mIoU 差值的算术平均**，不是按场景像素累计的 mIoU。
| scene_token | 帧数 | 改善 | 退化 | 平均逐帧差值(pp) |
|---|---:|---:|---:|---:|
| `c164a8e8e8b8489f964f711f472789be` | 41 | 14 | 27 | -1.436 |
| `afd73f70ff7d46d6b772d341c08e31a5` | 40 | 39 | 1 | +1.296 |
| `50ff554b3ecb4d208849d042b7643715` | 41 | 40 | 1 | +1.841 |
| `3ada261efee347cba2e7557794f1aec8` | 40 | 31 | 9 | +2.058 |
| `8e3364691ee94c698458e0913a29af78` | 40 | 22 | 18 | +3.922 |
| `91f797db8fb34ae5b32ba85eecae47c9` | 40 | 40 | 0 | +4.190 |
| `325cef682f064c55a255f2625c533b75` | 41 | 41 | 0 | +4.304 |
| `d7bacba9119840f78f3c804134ceece0` | 40 | 38 | 2 | +4.345 |
| `84e056bd8e994362a37cba45c0f75558` | 40 | 24 | 16 | +4.428 |
| `6a24a80e2ea3493c81f5fdb9fe78c28a` | 41 | 40 | 1 | +4.604 |
| `7365495b74464629813b41eacdb711af` | 40 | 40 | 0 | +4.652 |
| `905cfed4f0fc46679e8df8890cca4141` | 40 | 40 | 0 | +4.980 |

## 3. 退化最明显的 12 个样本

| index | scene_token | 原始帧 IoU | CCW90 帧 IoU | 差值(pp) |
|---:|---|---:|---:|---:|
| 3720 | `d3b86ca0a17840109e9e049b3dd40037` | 16.0524% | 3.8878% | -12.165 |
| 3715 | `c164a8e8e8b8489f964f711f472789be` | 17.6266% | 5.4647% | -12.162 |
| 3718 | `c164a8e8e8b8489f964f711f472789be` | 18.5950% | 6.8547% | -11.740 |
| 3719 | `c164a8e8e8b8489f964f711f472789be` | 19.5555% | 8.3240% | -11.231 |
| 3716 | `c164a8e8e8b8489f964f711f472789be` | 17.1117% | 6.2380% | -10.874 |
| 3717 | `c164a8e8e8b8489f964f711f472789be` | 17.1222% | 6.2503% | -10.872 |
| 3713 | `c164a8e8e8b8489f964f711f472789be` | 15.2888% | 4.6632% | -10.626 |
| 3721 | `d3b86ca0a17840109e9e049b3dd40037` | 18.4752% | 7.8698% | -10.605 |
| 3714 | `c164a8e8e8b8489f964f711f472789be` | 14.5959% | 4.2514% | -10.345 |
| 3712 | `c164a8e8e8b8489f964f711f472789be` | 15.6768% | 5.5207% | -10.156 |
| 3704 | `c164a8e8e8b8489f964f711f472789be` | 14.7402% | 5.3168% | -9.423 |
| 3703 | `c164a8e8e8b8489f964f711f472789be` | 14.1227% | 4.8805% | -9.242 |

## 4. 源码中需要核实的坐标链入口

以下仅为静态代码位置取证。**尚未证明具体是哪个源码环节错误**。

### 分割头 BEV 方向变换

- `BEVFormer-seg/projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_seg_head.py:53`: `class BevFeatureSlicer(nn.Module):`
- `BEVFormer-seg/projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_seg_head.py:166`: `self.feat_cropper = BevFeatureSlicer(self.det_grid_conf, self.map_grid_conf)`
- `BEVFormer-seg/projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_seg_head.py:286`: `seg_bev = torch.rot90(seg_bev, k=-1, dims=[2, 3])`
- `BEVFormer-seg/projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_seg_head.py:287`: `seg_bev = torch.flip(seg_bev, dims=[3])`
- `BEVFormer-seg/projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_seg_head.py:289`: `seg_bev = self.feat_cropper(seg_bev)`

### BevFeatureSlicer 定义

- `BEVFormer-seg/projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_head.py:50`: `class BevFeatureSlicer(nn.Module):`
- `BEVFormer-seg/projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_seg_head.py:53`: `class BevFeatureSlicer(nn.Module):`

### GT 地图坐标与语义标签入口

- `BEVFormer-seg/projects/configs/bevformer/bevformer_small_seg.py:193`: `dict(type='CustomCollect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img', 'semantic_indices'])`
- `BEVFormer-seg/projects/configs/bevformer/bevformer_small_seg.py:214`: `dict(type='CustomCollect3D', keys=['img', 'semantic_indices'])`
- `BEVFormer-seg/projects/mmdet3d_plugin/apis/test.py:101`: `target_semantic_indices = data['semantic_indices'][0].unsqueeze(0)`
- `BEVFormer-seg/projects/mmdet3d_plugin/apis/test.py:102`: `one_hot = target_semantic_indices.new_full(semantic.shape, 0)`
- `BEVFormer-seg/projects/mmdet3d_plugin/apis/test.py:103`: `one_hot.scatter_(1, target_semantic_indices, 1)`
- `BEVFormer-seg/projects/mmdet3d_plugin/apis/test.py:129`: `target_semantic_indices = data['semantic_indices'][0].unsqueeze(0)`
- `BEVFormer-seg/projects/mmdet3d_plugin/apis/test.py:130`: `one_hot = target_semantic_indices.new_full(semantic.shape, 0)`
- `BEVFormer-seg/projects/mmdet3d_plugin/apis/test.py:131`: `one_hot.scatter_(1, target_semantic_indices, 1)`
- `BEVFormer-seg/projects/mmdet3d_plugin/apis/test.py:164`: `pred_semantic_indices = torch.sum(pred * indices, axis=1).int()`
- `BEVFormer-seg/projects/mmdet3d_plugin/apis/test.py:165`: `target_semantic_indices = data['semantic_indices'][0].cuda()`
- `BEVFormer-seg/projects/mmdet3d_plugin/apis/test.py:167`: `semantic_map_iou_val(pred_semantic_indices,`
- `BEVFormer-seg/projects/mmdet3d_plugin/apis/test.py:168`: `target_semantic_indices)`
- `BEVFormer-seg/projects/mmdet3d_plugin/apis/test.py:261`: `pred_semantic_indices = torch.sum(pred * indices, axis=1).int()`
- `BEVFormer-seg/projects/mmdet3d_plugin/apis/test.py:262`: `target_semantic_indices = data['semantic_indices'][0].cuda()`
- `BEVFormer-seg/projects/mmdet3d_plugin/apis/test.py:264`: `semantic_map_iou_val(pred_semantic_indices,`
- `BEVFormer-seg/projects/mmdet3d_plugin/apis/test.py:265`: `target_semantic_indices)`
- `BEVFormer-seg/projects/mmdet3d_plugin/bevformer/detectors/bevformer.py:115`: `semantic_indices,`
- `BEVFormer-seg/projects/mmdet3d_plugin/bevformer/detectors/bevformer.py:143`: `seg_loss_inputs = [semantic_indices, outs]`
- `BEVFormer-seg/projects/mmdet3d_plugin/bevformer/detectors/bevformer.py:151`: `seg_loss_inputs = [semantic_indices, outs]`
- `BEVFormer-seg/projects/mmdet3d_plugin/bevformer/detectors/bevformer.py:204`: `semantic_indices=None,`
- `BEVFormer-seg/projects/mmdet3d_plugin/bevformer/detectors/bevformer.py:255`: `gt_labels_3d, semantic_indices,`
- `BEVFormer-seg/projects/mmdet3d_plugin/bevformer/detectors/bevformer_fp16.py:35`: `semantic_indices=None,`
- `BEVFormer-seg/projects/mmdet3d_plugin/datasets/pipelines/bevsegmentation.py:79`: `masks = self.maps[location].get_map_mask(`
- `BEVFormer-seg/projects/mmdet3d_plugin/datasets/pipelines/bevsegmentation.py:99`: `semantic_indices = np.sum(semantic_masks * indices, axis=0)`
- `BEVFormer-seg/projects/mmdet3d_plugin/datasets/pipelines/bevsegmentation.py:100`: `semantic_indices = np.where(semantic_indices > 5, 6, semantic_indices)`
- `BEVFormer-seg/projects/mmdet3d_plugin/datasets/pipelines/bevsegmentation.py:104`: `'semantic_indices': torch.from_numpy(semantic_indices).long(),`
- `BEVFormer-seg/projects/mmdet3d_plugin/datasets/pipelines/binimg.py:62`: `'semantic_indices': torch.from_numpy(bin_img)`
- `BEVFormer-seg/projects/mmdet3d_plugin/datasets/pipelines/rasterize.py:58`: `semantic_indices = np.sum(semantic_masks * indices, axis=0)`
- `BEVFormer-seg/projects/mmdet3d_plugin/datasets/pipelines/rasterize.py:63`: `'semantic_indices': torch.from_numpy(semantic_indices).long(),`
- `BEVFormer-seg/projects/mmdet3d_plugin/datasets/mtl_nuscenes_dataset_ego.py:269`: `e2g_rot = info['ego2global_rotation']`
- `BEVFormer-seg/projects/mmdet3d_plugin/datasets/mtl_nuscenes_dataset_ego.py:270`: `e2g_trans = info['ego2global_translation']`
- `BEVFormer-seg/projects/mmdet3d_plugin/datasets/mtl_nuscenes_dataset_ego.py:390`: `input_dict['lidar2ego_rots'] = torch.tensor(pyquaternion.Quaternion(`
- `BEVFormer-seg/projects/mmdet3d_plugin/datasets/mtl_nuscenes_dataset_ego.py:391`: `info['lidar2ego_rotation']).rotation_matrix)`
- `BEVFormer-seg/projects/mmdet3d_plugin/datasets/mtl_nuscenes_dataset_ego.py:392`: `input_dict['lidar2ego_trans'] = torch.tensor(`
- `BEVFormer-seg/projects/mmdet3d_plugin/datasets/mtl_nuscenes_dataset_ego.py:393`: `info['lidar2ego_translation'])`
- `BEVFormer-seg/projects/mmdet3d_plugin/datasets/mtl_nuscenes_dataset_ego.py:463`: `lidar2ego_translation = info['lidar2ego_translation']`
- `BEVFormer-seg/projects/mmdet3d_plugin/datasets/mtl_nuscenes_dataset_ego.py:464`: `lidar2ego_rotation = info['lidar2ego_rotation']`
- `BEVFormer-seg/projects/mmdet3d_plugin/datasets/mtl_nuscenes_dataset_ego.py:465`: `lidar2ego_rotation = pyquaternion.Quaternion(`
- `BEVFormer-seg/projects/mmdet3d_plugin/datasets/mtl_nuscenes_dataset_ego.py:466`: `lidar2ego_rotation).rotation_matrix`
- `BEVFormer-seg/projects/mmdet3d_plugin/datasets/mtl_nuscenes_dataset_ego.py:468`: `gt_bboxes_3d.rotate(lidar2ego_rotation.T)`
- `BEVFormer-seg/projects/mmdet3d_plugin/datasets/mtl_nuscenes_dataset_ego.py:469`: `gt_bboxes_3d.translate(lidar2ego_translation)`
- `BEVFormer-seg/projects/mmdet3d_plugin/datasets/mtl_nuscenes_dataset_ego.py:489`: `info['location'], info['ego2global_translation'], info['ego2global_rotation'])`
- `BEVFormer-seg/projects/mmdet3d_plugin/datasets/nuscenes_dataset.py:122`: `info['location'], info['ego2global_translation'], info['ego2global_rotation'])`
- `BEVFormer-seg/projects/mmdet3d_plugin/datasets/nuscenes_dataset.py:206`: `ego2global_translation=info['ego2global_translation'],`
- `BEVFormer-seg/projects/mmdet3d_plugin/datasets/nuscenes_dataset.py:207`: `ego2global_rotation=info['ego2global_rotation'],`
- 另有 52 处命中，未在此展开。

### Train/Test 配置关键入口

- `BEVFormer-seg/projects/configs/bevformer/bevformer_small_seg.py:22`: `det_grid_conf = {`
- `BEVFormer-seg/projects/configs/bevformer/bevformer_small_seg.py:23`: `'xbound': [-51.2, 51.2, 0.68],`
- `BEVFormer-seg/projects/configs/bevformer/bevformer_small_seg.py:24`: `'ybound': [-51.2, 51.2, 0.68],`
- `BEVFormer-seg/projects/configs/bevformer/bevformer_small_seg.py:29`: `map_grid_conf = {`
- `BEVFormer-seg/projects/configs/bevformer/bevformer_small_seg.py:30`: `'xbound': [-30.0, 30.0, 0.15],`
- `BEVFormer-seg/projects/configs/bevformer/bevformer_small_seg.py:31`: `'ybound': [-15.0, 15.0, 0.15],`
- `BEVFormer-seg/projects/configs/bevformer/bevformer_small_seg.py:36`: `grid_conf = map_grid_conf`
- `BEVFormer-seg/projects/configs/bevformer/bevformer_small_seg.py:57`: `bev_h_ = 150`
- `BEVFormer-seg/projects/configs/bevformer/bevformer_small_seg.py:58`: `bev_w_ = 150`
- `BEVFormer-seg/projects/configs/bevformer/bevformer_small_seg.py:87`: `bev_h=bev_h_,`
- `BEVFormer-seg/projects/configs/bevformer/bevformer_small_seg.py:88`: `bev_w=bev_w_,`
- `BEVFormer-seg/projects/configs/bevformer/bevformer_small_seg.py:97`: `det_grid_conf=det_grid_conf,`
- `BEVFormer-seg/projects/configs/bevformer/bevformer_small_seg.py:98`: `map_grid_conf=map_grid_conf,`
- `BEVFormer-seg/projects/configs/bevformer/bevformer_small_seg.py:152`: `row_num_embed=bev_h_,`
- `BEVFormer-seg/projects/configs/bevformer/bevformer_small_seg.py:153`: `col_num_embed=bev_w_,`
- `BEVFormer-seg/projects/configs/bevformer/bevformer_small_seg.py:181`: `train_pipeline = [`
- `BEVFormer-seg/projects/configs/bevformer/bevformer_small_seg.py:186`: `dict(type='RasterizeMapVectors', map_grid_conf=map_grid_conf),`
- `BEVFormer-seg/projects/configs/bevformer/bevformer_small_seg.py:196`: `test_pipeline = [`
- `BEVFormer-seg/projects/configs/bevformer/bevformer_small_seg.py:199`: `dict(type='RasterizeMapVectors', map_grid_conf=map_grid_conf),`
- `BEVFormer-seg/projects/configs/bevformer/bevformer_small_seg.py:225`: `pipeline=train_pipeline,`
- `BEVFormer-seg/projects/configs/bevformer/bevformer_small_seg.py:230`: `grid_conf=grid_conf,`
- `BEVFormer-seg/projects/configs/bevformer/bevformer_small_seg.py:231`: `bev_size=(bev_h_, bev_w_),`
- `BEVFormer-seg/projects/configs/bevformer/bevformer_small_seg.py:238`: `grid_conf=grid_conf,`
- `BEVFormer-seg/projects/configs/bevformer/bevformer_small_seg.py:240`: `pipeline=test_pipeline,  bev_size=(bev_h_, bev_w_),`
- `BEVFormer-seg/projects/configs/bevformer/bevformer_small_seg.py:243`: `grid_conf=grid_conf,`
- `BEVFormer-seg/projects/configs/bevformer/bevformer_small_seg.py:246`: `pipeline=test_pipeline, bev_size=(bev_h_, bev_w_),`
- `BEVFormer-seg/projects/configs/bevformer/bevformer_small_seg.py:270`: `evaluation = dict(interval=1, pipeline=test_pipeline)`

## 5. 下一步需核验的事实

- Train 与 Val 的 `semantic_indices` 生成逻辑、地图参考系和栅格方向是否一致。
- BEVFormer encoder 输出 BEV 的轴顺序，分割头 `rot90/flip` 与 Slicer 的轴和取样范围。
- GT 以 Ego 为参考、预测 BEV 以 LiDAR 为参考时，是否使用了真实外参进行完整坐标映射。
- 针对退化帧保存预测/GT 叠图以解释 150 帧例外，并检查 `ped_crossing` 提升较小的原因。
- 若需修复，必须明确训练端与推理端一致的映射定义，重新跑固定评估协议。
