# Step 51：训练/推理 BEV 空间协议静态审计

## 1. 审计范围

本实验不运行模型，不改源码和权重，比较 checkpoint 内保存的训练配置、
独立作者环境与工作环境的静态代码，并追踪 BEV → Slicer → GT 坐标链。
checkpoint 配置只证明元数据内容，**不能证明训练时实际执行的源码**。

## 2. checkpoint 身份

- checkpoint_path: `/mnt/g/nus/bevformer_small_seg.pth`
- checkpoint_sha256: `b9e369ba2e4514050fefe914f57565a31b918edaa735958a1f877184ad75dc68`
- checkpoint_meta_epoch: `4`
- checkpoint_meta_iter: `112520`
- checkpoint_mmc_version: `1.4.0`
- checkpoint_config_sha256: `ee91cc57b3bfbe900729747cc43cb5a82af378e5b7cbba6862ed05f3d2cf9378`
- checkpoint_has_seg_decoder: `True`

## 3. 三份配置关键字段

| 字段 | 训练=作者推理 | 训练=工作推理 | 作者=工作 |
|---|---|---|---|
| `model.type` | 一致 | 一致 | 一致 |
| `model.video_test_mode` | 一致 | 一致 | 一致 |
| `model.pts_bbox_head.type` | 一致 | 一致 | 一致 |
| `model.pts_bbox_head.task` | 一致 | 一致 | 一致 |
| `model.pts_bbox_head.bev_h` | 一致 | 一致 | 一致 |
| `model.pts_bbox_head.bev_w` | 一致 | 一致 | 一致 |
| `model.pts_bbox_head.det_grid_conf` | 一致 | 一致 | 一致 |
| `model.pts_bbox_head.map_grid_conf` | 一致 | 一致 | 一致 |
| `det_grid_conf` | 一致 | 一致 | 一致 |
| `map_grid_conf` | 一致 | 一致 | 一致 |
| `point_cloud_range` | 一致 | 一致 | 一致 |
| `img_norm_cfg` | 一致 | 一致 | 一致 |
| `train_pipeline` | 一致 | 一致 | 一致 |
| `test_pipeline` | 一致 | 一致 | 一致 |
| `data.train.pipeline` | 一致 | 一致 | 一致 |
| `data.val.pipeline` | 一致 | 一致 | 一致 |
| `data.test.pipeline` | 一致 | 一致 | 一致 |
| `data.train.grid_conf` | 一致 | 一致 | 一致 |
| `data.test.grid_conf` | 一致 | 一致 | 一致 |
| `data.train.bev_size` | 一致 | 一致 | 一致 |
| `data.test.bev_size` | 一致 | 一致 | 一致 |
| `data.test.test_mode` | 一致 | 一致 | 一致 |
| `data.nonshuffler_sampler` | 一致 | 一致 | 一致 |
| `queue_length` | 一致 | 一致 | 一致 |

## 4. 关键函数源码 AST 对照

AST SHA256 用于比较可执行语法结构；不一致需要查看具体代码差异。

| 位置 | 作者与工作环境 AST |
|---|---|
| `head/BevFeatureSlicer.__init__` | 一致 |
| `head/BevFeatureSlicer.forward` | 一致 |
| `head/BEVFormerHead.forward` | 一致 |
| `encoder/BEVFormerEncoder.get_reference_points` | 一致 |
| `encoder/BEVFormerEncoder.point_sampling` | 一致 |
| `raster/line_geom_to_mask` | 一致 |
| `raster/mask_for_lines` | 一致 |
| `raster/preprocess_map` | 一致 |
| `vector/VectorizedLocalMap.gen_vectorized_samples` | 一致 |
| `raster_pipeline/RasterizeMapVectors.__init__` | 一致 |
| `raster_pipeline/RasterizeMapVectors.__call__` | 一致 |

## 5. 作者环境的有效空间相关语句

### head
**BevFeatureSlicer.__init__**
```python
L68: self.map_x = torch.arange(
L71: self.map_y = torch.arange(
L75: self.norm_map_x = self.map_x / (- bev_start_position[0])
L76: self.norm_map_y = self.map_y / (- bev_start_position[1])
L78: self.map_grid = torch.stack(torch.meshgrid(
L79: self.norm_map_x, self.norm_map_y), dim=2).permute(1, 0, 2)
```
**BevFeatureSlicer.forward**
```python
L92: grid = self.map_grid.unsqueeze(0).type_as(
L95: return F.grid_sample(x, grid=grid, mode='bilinear', align_corners=True)
```
**BEVFormerHead.forward**
```python
L275: bev_embed, hs, init_reference, inter_references = outputs
L295: seg_bev = bev_embed.reshape(self.bev_h, self.bev_w, bs, -1).permute(2, 3, 1, 0)  # b, c , w, h
L296: seg_bev = torch.flip(seg_bev, dims=[2])
L317: seg_bev = self.feat_cropper(seg_bev)
L319: seg_pred = self.seg_decoder(seg_bev)
L357: 'bev_embed': bev_embed,
L369: 'bev_embed': bev_embed,
L377: 'bev_embed': bev_embed,
```
### encoder
**BEVFormerEncoder.get_reference_points**
```python
L81: ref_y, ref_x = torch.meshgrid(
L87: ref_y = ref_y.reshape(-1)[None] / H
L88: ref_x = ref_x.reshape(-1)[None] / W
L89: ref_2d = torch.stack((ref_x, ref_y), -1)
```
**BEVFormerEncoder.point_sampling**
```python
L97: lidar2img = []
L99: lidar2img.append(img_meta['lidar2img'])
L100: lidar2img = np.asarray(lidar2img)
L101: lidar2img = reference_points.new_tensor(lidar2img)  # (B, N, 4, 4) # 坐标变换矩阵，将3d点映射到2d图像平面上去 #[1, 6, 4, 4]
L117: num_cam = lidar2img.size(1)
L122: lidar2img = lidar2img.view(
L125: reference_points_cam = torch.matmul(lidar2img.to(torch.float32),
```
### raster_pipeline
**RasterizeMapVectors.__init__**
```python
```
**RasterizeMapVectors.__call__**
```python
L52: semantic_masks, instance_masks, forward_masks, backward_masks = preprocess_map(
L55: semantic_masks = semantic_masks.numpy()
L56: num_cls = semantic_masks.shape[0]
L58: semantic_indices = np.sum(semantic_masks * indices, axis=0)
L61: 'semantic_map': torch.from_numpy(semantic_masks),
```
### vector
**VectorizedLocalMap.gen_vectorized_samples**
```python
L55: def gen_vectorized_samples(self, location, ego2global_translation, ego2global_rotation):
L56: map_pose = ego2global_translation[:2]
L57: rotation = Quaternion(ego2global_rotation)
L59: patch_box = (map_pose[0], map_pose[1],
L60: self.patch_size[0], self.patch_size[1])
L61: patch_angle = quaternion_yaw(rotation) / np.pi * 180
L64: patch_box, patch_angle, self.line_classes, location)
L68: patch_box, patch_angle, self.ped_crossing_classes, location)
L73: patch_box, patch_angle, self.polygon_classes, location)
```
### raster
**line_geom_to_mask**
```python
L48: def line_geom_to_mask(layer_geom, confidence_levels, local_box, canvas_size, thickness, idx, type='index', angle_class=36):
L53: canvas_h = canvas_size[0]
L54: canvas_w = canvas_size[1]
L55: scale_height = canvas_h / patch_h
L56: scale_width = canvas_w / patch_w
L58: trans_x = -patch_x + patch_w / 2.0
L59: trans_y = -patch_y + patch_h / 2.0
L61: map_mask = np.zeros(canvas_size, np.uint8)
L71: new_line, [1.0, 0.0, 0.0, 1.0, trans_x, trans_y])
L73: new_line, xfact=scale_width, yfact=scale_height, origin=(0, 0))
```
**mask_for_lines**
```python
```
**preprocess_map**
```python
L94: def preprocess_map(vectors, patch_size, canvas_size, max_channel, thickness, angle_class):
L105: local_box = (0.0, 0.0, patch_size[0], patch_size[1])
L114: vector_num_list[i], confidence_levels, local_box, canvas_size, thickness, idx)
L117: vector_num_list[i], confidence_levels, local_box, canvas_size, thickness + 4, 1)
L120: vector_num_list[i], confidence_levels, local_box, canvas_size, thickness, 1, type='forward', angle_class=angle_class)
L123: vector_num_list[i], confidence_levels, local_box, canvas_size, thickness, 1, type='backward', angle_class=angle_class)
L137: semantic_masks = instance_masks != 0
L138: semantic_masks = torch.from_numpy(semantic_masks)
L141: return semantic_masks, instance_masks, forward_masks, backward_masks
```

## 6. 当前结论边界

- 本次对比能识别 checkpoint 配置与当前源码的**静态不一致**；
  未发现不一致也不能证明训练执行的源码相同。
- 当前前景 mIoU 不能通过静态哈希推导，也不应据此声称已修复。
- 下一步应根据差异结果选择单帧 runtime hook，验证每个空间张量
  的轴向、形状、裁剪网格方向与地图 GT，而不是直接对输出旋转拟合。
- 两套源码相同且配置一致时，优先核实训练时的 git commit、
  真实标签来源与 checkpoint 对应关系。
