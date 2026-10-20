# Step 36：训练/推理坐标链源码证据

记录日期：2026-10-20

## 1. 目标与解释边界

该文件提取本地实际源码、固定退化帧信息与张量轴排列恒等式。
**不对根因预设结论；不修改权重、GT、分割头或任何模型文件。**

## 2. 张量空间变换的合成测试

- 非正方形合成网格（3×5）测试：通过。
- 原版 `rot90(-1) + flip(dim=3)` = 源网格转置：通过。
- Step34 在 Slicer 前增加 `flip(dim=2)` = 源网格逆时针旋转 90°：通过。
- 上述只是张量轴排列恒等式，**不等于物理 LiDAR→Ego 标定验证**。

## 3. Step34 异常帧回溯

- Step34 CSV 总记录数：6019。
- 焦点样本数：12；保存至 `/home/jjx2003520/projects/bev_vla/outputs/drivelm_check/root_cause/step36_coordinate_contract/focus_frames.csv`。
- 重点对照 3715、3718、3719 帧是否属于同一场景，
  以及周围帧的连贯退化是否与地图内容/轨迹有关。

## 4. 关键源码全文片段（以本地行号为准）

### `projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_seg_head.py` 第 1–120 行

```text
   1 | # ---------------------------------------------
   2 | # Copyright (c) OpenMMLab. All rights reserved.
   3 | # ---------------------------------------------
   4 | #  Modified by Zhiqi Li
   5 | # ---------------------------------------------
   6 | 
   7 | import copy
   8 | import torch
   9 | import torch.nn as nn
  10 | import torch.nn.functional as F
  11 | from mmcv.cnn import Linear, bias_init_with_prob
  12 | from mmcv.utils import TORCH_VERSION, digit_version
  13 | 
  14 | from mmdet.core import (multi_apply, reduce_mean)
  15 | from mmdet.models.utils.transformer import inverse_sigmoid
  16 | from mmdet.models import HEADS
  17 | from mmdet.models.dense_heads import DETRHead
  18 | from mmdet3d.core.bbox.coders import build_bbox_coder
  19 | from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox
  20 | from mmcv.runner import force_fp32, auto_fp16
  21 | from ..modules.builder import build_seg_encoder
  22 | #from mmdet.models.builder import build_loss
  23 | from mmseg.models.builder import build_loss
  24 | from mmdet.models.dense_heads.base_dense_head import BaseDenseHead
  25 | from mmdet.models.dense_heads.dense_test_mixins import BBoxTestMixin
  26 | 
  27 | from mmdet.models.utils import build_transformer
  28 | 
  29 | def calculate_birds_eye_view_parameters(x_bounds, y_bounds, z_bounds):
  30 |     """
  31 |     Parameters
  32 |     ----------
  33 |         x_bounds: Forward direction in the ego-car.
  34 |         y_bounds: Sides
  35 |         z_bounds: Height
  36 | 
  37 |     Returns
  38 |     -------
  39 |         bev_resolution: Bird's-eye view bev_resolution
  40 |         bev_start_position Bird's-eye view first element
  41 |         bev_dimension Bird's-eye view tensor spatial dimension
  42 |     """
  43 |     bev_resolution = torch.tensor(
  44 |         [row[2] for row in [x_bounds, y_bounds, z_bounds]])
  45 |     bev_start_position = torch.tensor(
  46 |         [row[0] + row[2] / 2.0 for row in [x_bounds, y_bounds, z_bounds]])
  47 |     bev_dimension = torch.tensor([(row[1] - row[0]) / row[2]
  48 |                                  for row in [x_bounds, y_bounds, z_bounds]], dtype=torch.long)
  49 | 
  50 |     return bev_resolution, bev_start_position, bev_dimension
  51 | 
  52 | 
  53 | class BevFeatureSlicer(nn.Module):
  54 |     # crop the interested area in BEV feature for semantic map segmentation
  55 |     def __init__(self, grid_conf, map_grid_conf):
  56 |         super().__init__()
  57 | 
  58 |         if grid_conf == map_grid_conf:
  59 |             self.identity_mapping = True
  60 |         else:
  61 |             self.identity_mapping = False
  62 | 
  63 |             bev_resolution, bev_start_position, bev_dimension = calculate_birds_eye_view_parameters(
  64 |                 grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'],
  65 |             )
  66 | 
  67 |             map_bev_resolution, map_bev_start_position, map_bev_dimension = calculate_birds_eye_view_parameters(
  68 |                 map_grid_conf['xbound'], map_grid_conf['ybound'], map_grid_conf['zbound'],
  69 |             )
  70 | 
  71 |             self.map_x = torch.arange(
  72 |                 map_bev_start_position[0], map_grid_conf['xbound'][1], map_bev_resolution[0])
  73 | 
  74 |             self.map_y = torch.arange(
  75 |                 map_bev_start_position[1], map_grid_conf['ybound'][1], map_bev_resolution[1])
  76 | 
  77 |             # convert to normalized coords
  78 |             self.norm_map_x = self.map_x / (- bev_start_position[0])
  79 |             self.norm_map_y = self.map_y / (- bev_start_position[1])
  80 |             # vision 1 失败
  81 |             self.map_grid = torch.stack(torch.meshgrid(
  82 |                 self.norm_map_x, self.norm_map_y), dim=2).permute(1, 0, 2)
  83 | 
  84 |             # self.map_grid = torch.stack(torch.meshgrid(
  85 |             #     self.norm_map_x, self.norm_map_y, indexing='xy'), dim=2)
  86 | 
  87 |              # vision 2 test
  88 |             # self.map_grid = torch.stack(torch.meshgrid(
  89 |             #     self.norm_map_x, self.norm_map_y), dim=2)
  90 | 
  91 |     def forward(self, x):
  92 |         # x: bev feature map tensor of shape (b, c, h, w)
  93 |         if self.identity_mapping:
  94 |             return x
  95 |         else:
  96 |             grid = self.map_grid.unsqueeze(0).type_as(
  97 |                 x).repeat(x.shape[0], 1, 1, 1)
  98 | 
  99 |             return F.grid_sample(x, grid=grid, mode='bilinear', align_corners=True)
 100 | 
 101 | @HEADS.register_module()
 102 | class BEVFormerHead_seg(BaseDenseHead, BBoxTestMixin):
 103 |     """Head of Detr3D.
 104 |     Args:
 105 |         with_box_refine (bool): Whether to refine the reference points
 106 |             in the decoder. Defaults to False.
 107 |         as_two_stage (bool) : Whether to generate the proposal from
 108 |             the outputs of encoder.
 109 |         transformer (obj:`ConfigDict`): ConfigDict is used for building
 110 |             the Encoder and Decoder.
 111 |         bev_h, bev_w (int): spatial shape of BEV queries.
 112 |     """
 113 | 
 114 |     def __init__(self,
 115 |                  *args,
 116 |                  with_box_refine=False,
 117 |                  as_two_stage=False,
 118 |                  transformer=None,
 119 |                  bbox_coder=None,
 120 |                  seg_encoder=None,
```

### `projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_seg_head.py` 第 145–190 行

```text
 145 |             self.code_weights = code_weights
 146 |         else:
 147 |             self.code_weights = [1.0, 1.0, 1.0,
 148 |                                  1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]
 149 | 
 150 |         # TODO
 151 |         # add muti task，用于确定这三种任务
 152 |         self.task = task
 153 |         self.bbox_coder = build_bbox_coder(bbox_coder)
 154 |         self.transformer = build_transformer(transformer)
 155 |         self.embed_dims = self.transformer.embed_dims
 156 |         self.pc_range = self.bbox_coder.pc_range
 157 |         self.real_w = self.pc_range[3] - self.pc_range[0]
 158 |         self.real_h = self.pc_range[4] - self.pc_range[1]
 159 |         self.num_cls_fcs = num_cls_fcs - 1
 160 |         self.code_weights = nn.Parameter(torch.tensor(
 161 |             self.code_weights, requires_grad=False), requires_grad=False)
 162 |         # 如果包含分割任务，则实例化分割子网络以及loss
 163 |         if self.task.get('seg'):
 164 |             self.map_grid_conf = map_grid_conf
 165 |             self.det_grid_conf = det_grid_conf
 166 |             self.feat_cropper = BevFeatureSlicer(self.det_grid_conf, self.map_grid_conf)
 167 |             self.seg_decoder = build_seg_encoder(seg_encoder)
 168 |             self.loss_seg = build_loss(loss_seg)
 169 | 
 170 |     def _init_layers(self):
 171 |         """Initialize classification branch and regression branch of head."""
 172 |         # 初始化网络层参数时需要根据任务本身来进行
 173 |         if self.task.get('det'):
 174 |             cls_branch = []
 175 |             for _ in range(self.num_reg_fcs):
 176 |                 cls_branch.append(Linear(self.embed_dims, self.embed_dims))
 177 |                 cls_branch.append(nn.LayerNorm(self.embed_dims))
 178 |                 cls_branch.append(nn.ReLU(inplace=True))
 179 |             cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
 180 |             fc_cls = nn.Sequential(*cls_branch)
 181 | 
 182 |             reg_branch = []
 183 |             for _ in range(self.num_reg_fcs):
 184 |                 reg_branch.append(Linear(self.embed_dims, self.embed_dims))
 185 |                 reg_branch.append(nn.ReLU())
 186 |             reg_branch.append(Linear(self.embed_dims, self.code_size))
 187 |             reg_branch = nn.Sequential(*reg_branch)
 188 | 
 189 |             def _get_clones(module, N):
 190 |                 return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
```

### `projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_seg_head.py` 第 255–315 行

```text
 255 |                 grid_length=(self.real_h / self.bev_h,
 256 |                              self.real_w / self.bev_w),
 257 |                 bev_pos=bev_pos,
 258 |                 img_metas=img_metas,
 259 |                 prev_bev=prev_bev,
 260 |             )
 261 |         else:
 262 |             outputs = self.transformer(
 263 |                 mlvl_feats,
 264 |                 bev_queries,
 265 |                 object_query_embeds,
 266 |                 self.bev_h,
 267 |                 self.bev_w,
 268 |                 grid_length=(self.real_h / self.bev_h,
 269 |                              self.real_w / self.bev_w),
 270 |                 bev_pos=bev_pos,
 271 |                 reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501
 272 |                 cls_branches=self.cls_branches if self.as_two_stage else None,
 273 |                 img_metas=img_metas,
 274 |                 prev_bev=prev_bev
 275 |         )
 276 | 
 277 |         bev_embed, hs, init_reference, inter_references = outputs
 278 | 
 279 |         # add segmentation task
 280 |         # TODO : 添加分割任务预测网络
 281 |         if self.task.get('seg'):
 282 |             # bev_queries(Tensor): (bev_h * bev_w, bs, c) 这是一个非常重要的地方，但是感觉这样的操作并无不同
 283 |             seg_bev = bev_embed.reshape(self.bev_h, self.bev_w, bs, -1).permute(2, 3, 0, 1)
 284 | 
 285 |             # 将bev空间特征从lidar系转换到图像系
 286 |             seg_bev = torch.rot90(seg_bev, k=-1, dims=[2, 3])
 287 |             seg_bev = torch.flip(seg_bev, dims=[3])
 288 |             # crop 分割区域并插值到分割标注的尺寸
 289 |             seg_bev = self.feat_cropper(seg_bev)
 290 |             # 这里要注意特征图大小
 291 |             seg_pred = self.seg_decoder(seg_bev)
 292 | 
 293 |         if self.task.get('det'):
 294 |             hs = hs.permute(0, 2, 1, 3)
 295 |             outputs_classes = []
 296 |             outputs_coords = []
 297 |             for lvl in range(hs.shape[0]):
 298 |                 if lvl == 0:
 299 |                     reference = init_reference
 300 |                 else:
 301 |                     reference = inter_references[lvl - 1]
 302 |                 reference = inverse_sigmoid(reference)
 303 |                 outputs_class = self.cls_branches[lvl](hs[lvl])
 304 |                 tmp = self.reg_branches[lvl](hs[lvl])
 305 | 
 306 |                 # TODO: check the shape of reference
 307 |                 assert reference.shape[-1] == 3
 308 |                 tmp[..., 0:2] += reference[..., 0:2]
 309 |                 tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
 310 |                 tmp[..., 4:5] += reference[..., 2:3]
 311 |                 tmp[..., 4:5] = tmp[..., 4:5].sigmoid()
 312 |                 tmp[..., 0:1] = (tmp[..., 0:1] * (self.pc_range[3] -
 313 |                                  self.pc_range[0]) + self.pc_range[0])
 314 |                 tmp[..., 1:2] = (tmp[..., 1:2] * (self.pc_range[4] -
 315 |                                  self.pc_range[1]) + self.pc_range[1])
```

### `projects/configs/bevformer/bevformer_small_seg.py` 第 15–42 行

```text
  15 | plugin_dir = 'projects/mmdet3d_plugin/'
  16 | 
  17 | # If point cloud range is changed, the models should also change their point
  18 | # cloud range accordingly
  19 | point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
  20 | voxel_size = [0.2, 0.2, 8]
  21 | 
  22 | det_grid_conf = {
  23 |     'xbound': [-51.2, 51.2, 0.68],
  24 |     'ybound': [-51.2, 51.2, 0.68],
  25 |     'zbound': [-10.0, 10.0, 20.0],
  26 |     'dbound': [1.0, 60.0, 1.0],
  27 | }
  28 | 
  29 | map_grid_conf = {
  30 |     'xbound': [-30.0, 30.0, 0.15],
  31 |     'ybound': [-15.0, 15.0, 0.15],
  32 |     'zbound': [-10.0, 10.0, 20.0],
  33 |     'dbound': [1.0, 60.0, 1.0],
  34 | }
  35 | 
  36 | grid_conf = map_grid_conf
  37 | 
  38 | img_norm_cfg = dict(
  39 |     mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
  40 | # For nuScenes we usually do 10-class detection
  41 | class_names = [
  42 |     'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
```

### `projects/configs/bevformer/bevformer_small_seg.py` 第 54–110 行

```text
  54 | _pos_dim_ = _dim_//2
  55 | _ffn_dim_ = _dim_*2
  56 | _num_levels_ = 1
  57 | bev_h_ = 150
  58 | bev_w_ = 150
  59 | queue_length = 3 # each sequence contains `queue_length` frames.
  60 | 
  61 | model = dict(
  62 |     type='BEVFormer',
  63 |     use_grid_mask=True,
  64 |     video_test_mode=True,
  65 |     img_backbone=dict(
  66 |         type='ResNet',
  67 |         depth=101,
  68 |         num_stages=4,
  69 |         out_indices=(3,),
  70 |         frozen_stages=1,
  71 |         norm_cfg=dict(type='BN2d', requires_grad=False),
  72 |         norm_eval=True,
  73 |         style='caffe',
  74 |         with_cp=True, # using checkpoint to save GPU memory
  75 |         dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False), # original DCNv2 will print log when perform load_state_dict
  76 |         stage_with_dcn=(False, False, True, True)),
  77 |     img_neck=dict(
  78 |         type='FPN',
  79 |         in_channels=[2048],
  80 |         out_channels=_dim_,
  81 |         start_level=0,
  82 |         add_extra_convs='on_output',
  83 |         num_outs=_num_levels_,
  84 |         relu_before_extra_convs=True),
  85 |     pts_bbox_head=dict(
  86 |         type='BEVFormerHead',
  87 |         bev_h=bev_h_,
  88 |         bev_w=bev_w_,
  89 |         num_query=900,
  90 |         num_classes=10,
  91 |         in_channels=_dim_,
  92 |         sync_cls_avg_factor=True,
  93 |         with_box_refine=False,
  94 |         as_two_stage=False,
  95 |         task={'seg':True,
  96 |               'det':False},
  97 |         det_grid_conf=det_grid_conf,
  98 |         map_grid_conf=map_grid_conf,
  99 |         seg_encoder=dict(
 100 |             type='SegEncode',
 101 |             inC=256,
 102 |             outC=4),
 103 |         loss_seg=dict(
 104 |             type='CrossEntropyLoss',
 105 |             use_sigmoid=False,
 106 |             loss_weight=3.0,
 107 |             class_weight=[0.5, 2.0, 2.0, 2.0]),
 108 |         transformer=dict(
 109 |             type='PerceptionTransformer',
 110 |             rotate_prev_bev=True,
```

### `projects/configs/bevformer/bevformer_small_seg.py` 第 175–250 行

```text
 175 | 
 176 | dataset_type = 'CustomNuScenesDataset'
 177 | data_root = 'data/nuscenes/'
 178 | file_client_args = dict(backend='disk')
 179 | 
 180 | 
 181 | train_pipeline = [
 182 |     dict(type='LoadMultiViewImageFromFiles', to_float32=True),
 183 |     dict(type='PhotoMetricDistortionMultiViewImage'),
 184 |     dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
 185 |     # convert map labels
 186 |     dict(type='RasterizeMapVectors', map_grid_conf=map_grid_conf),
 187 |     dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
 188 |     dict(type='ObjectNameFilter', classes=class_names),
 189 |     dict(type='NormalizeMultiviewImage', **img_norm_cfg),
 190 |     dict(type='RandomScaleImageMultiViewImage', scales=[0.8]),
 191 |     dict(type='PadMultiViewImage', size_divisor=32),
 192 |     dict(type='DefaultFormatBundle3D', class_names=class_names),
 193 |     dict(type='CustomCollect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img', 'semantic_indices'])
 194 | ]
 195 | 
 196 | test_pipeline = [
 197 |     dict(type='LoadMultiViewImageFromFiles', to_float32=True),
 198 |     # convert map labels
 199 |     dict(type='RasterizeMapVectors', map_grid_conf=map_grid_conf),
 200 |     dict(type='NormalizeMultiviewImage', **img_norm_cfg),
 201 |     # dict(type='PadMultiViewImage', size_divisor=32),
 202 |     dict(
 203 |         type='MultiScaleFlipAug3D',
 204 |         img_scale=(1600, 900),
 205 |         pts_scale_ratio=1,
 206 |         flip=False,
 207 |         transforms=[
 208 |             dict(type='RandomScaleImageMultiViewImage', scales=[0.8]),
 209 |             dict(type='PadMultiViewImage', size_divisor=32),
 210 |             dict(
 211 |                 type='DefaultFormatBundle3D',
 212 |                 class_names=class_names,
 213 |                 with_label=False),
 214 |             dict(type='CustomCollect3D', keys=['img', 'semantic_indices'])
 215 |         ])
 216 | ]
 217 | 
 218 | data = dict(
 219 |     samples_per_gpu=1,
 220 |     workers_per_gpu=4,
 221 |     train=dict(
 222 |         type=dataset_type,
 223 |         data_root=data_root,
 224 |         ann_file=data_root + 'nuscenes_infos_temporal_train.pkl',
 225 |         pipeline=train_pipeline,
 226 |         classes=class_names,
 227 |         modality=input_modality,
 228 |         test_mode=False,
 229 |         use_valid_flag=True,
 230 |         grid_conf=grid_conf,
 231 |         bev_size=(bev_h_, bev_w_),
 232 |         queue_length=queue_length,
 233 |         # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
 234 |         # and box_type_3d='Depth' in sunrgbd and scannet dataset.
 235 |         box_type_3d='LiDAR'),
 236 |     val=dict(type=dataset_type,
 237 |              data_root=data_root,
 238 |              grid_conf=grid_conf,
 239 |              ann_file=data_root + 'nuscenes_infos_temporal_val.pkl',
 240 |              pipeline=test_pipeline,  bev_size=(bev_h_, bev_w_),
 241 |              classes=class_names, modality=input_modality, samples_per_gpu=1),
 242 |     test=dict(type=dataset_type,
 243 |               grid_conf=grid_conf,
 244 |               data_root=data_root,
 245 |               ann_file=data_root + 'nuscenes_infos_temporal_val.pkl',
 246 |               pipeline=test_pipeline, bev_size=(bev_h_, bev_w_),
 247 |               classes=class_names, modality=input_modality),
 248 |     shuffler_sampler=dict(type='DistributedGroupSampler'),
 249 |     nonshuffler_sampler=dict(type='DistributedSampler')
 250 | )
```

### `projects/mmdet3d_plugin/datasets/pipelines/rasterize.py` 第 1–120 行

```text
   1 | import torch
   2 | import numpy as np
   3 | 
   4 | from mmdet.datasets.builder import PIPELINES
   5 | from ..utils import preprocess_map
   6 | 
   7 | import warnings
   8 | warnings.filterwarnings('ignore')
   9 | 
  10 | 
  11 | @PIPELINES.register_module()
  12 | class RasterizeMapVectors(object):
  13 |     """Load multi channel images from a list of separate channel files.
  14 | 
  15 |     Expects results['img_filename'] to be a list of filenames.
  16 | 
  17 |     Args:
  18 |         to_float32 (bool): Whether to convert the img to float32.
  19 |             Defaults to False.
  20 |         color_type (str): Color type of the file. Defaults to 'unchanged'.
  21 |     """
  22 | 
  23 |     def __init__(self,
  24 |                  map_grid_conf=None,
  25 |                  map_max_channel=3,
  26 |                  map_thickness=5,
  27 |                  map_angle_class=36
  28 |                  ):
  29 | 
  30 |         self.map_max_channel = map_max_channel
  31 |         self.map_thickness = map_thickness
  32 |         self.map_angle_class = map_angle_class
  33 | 
  34 |         map_xbound, map_ybound = map_grid_conf['xbound'], map_grid_conf['ybound']
  35 | 
  36 |         # patch_size: 在 y, x 方向上的坐标 range
  37 |         patch_h = map_ybound[1] - map_ybound[0]
  38 |         patch_w = map_xbound[1] - map_xbound[0]
  39 | 
  40 |         # canvas_size: 在 y, x 方向上的 bev 尺寸
  41 |         canvas_h = int(patch_h / map_ybound[2])
  42 |         canvas_w = int(patch_w / map_xbound[2])
  43 | 
  44 |         self.map_patch_size = (patch_h, patch_w)
  45 |         self.map_canvas_size = (canvas_h, canvas_w)
  46 | 
  47 |     def __call__(self, results):
  48 |         vectors = results['vectors']
  49 |         for vector in vectors:
  50 |             vector['pts'] = vector['pts'][:, :2]
  51 | 
  52 |         semantic_masks, instance_masks, forward_masks, backward_masks = preprocess_map(
  53 |             vectors, self.map_patch_size, self.map_canvas_size, self.map_max_channel, self.map_thickness, self.map_angle_class)
  54 | 
  55 |         semantic_masks = semantic_masks.numpy()
  56 |         num_cls = semantic_masks.shape[0]
  57 |         indices = np.arange(1, num_cls + 1).reshape(-1, 1, 1)
  58 |         semantic_indices = np.sum(semantic_masks * indices, axis=0)
  59 | 
  60 |         results.update({
  61 |             'semantic_map': torch.from_numpy(semantic_masks),
  62 |             'instance_map': torch.from_numpy(instance_masks),
  63 |             'semantic_indices': torch.from_numpy(semantic_indices).long(),
  64 |             'forward_direction': torch.from_numpy(forward_masks),
  65 |             'backward_direction': torch.from_numpy(backward_masks),
  66 |         })
  67 | 
  68 |         return results
```

### `projects/mmdet3d_plugin/datasets/pipelines/bevsegmentation.py` 第 40–110 行

```text
  40 |     def show_seg(self, labels):
  41 | 
  42 |         mask_colors = [
  43 |             np.random.randint(0, 256, (1, 3), dtype=np.uint8)
  44 |             for _ in range(len(self.classes) + 1)
  45 |         ]
  46 |         img = np.zeros((200, 400, 3))
  47 | 
  48 |         for index, mask_ in enumerate(labels):
  49 |             color_mask = mask_colors[index]
  50 |             mask_ = mask_.astype(bool)
  51 |             img[mask_] = color_mask
  52 | 
  53 |         return img
  54 | 
  55 |     def __call__(self, results):
  56 | 
  57 |         location, ego2global_translation, ego2global_rotation = \
  58 |             results['location'], results['ego2global_translation'], results['ego2global_rotation']
  59 |         map_pose = ego2global_translation[:2]
  60 |         rotation = Quaternion(ego2global_rotation)
  61 | 
  62 |         patch_box = (map_pose[0], map_pose[1], self.patch_size[0], self.patch_size[1])
  63 |         patch_angle = quaternion_yaw(rotation) / np.pi * 180
  64 | 
  65 |         mappings = {}
  66 |         for name in self.classes:
  67 |             if name == "drivable_area*":
  68 |                 mappings[name] = ["road_segment", "lane"]
  69 |             elif name == "divider":
  70 |                 mappings[name] = ["road_divider", "lane_divider"]
  71 |             else:
  72 |                 mappings[name] = [name]
  73 | 
  74 |         layer_names = []
  75 |         for name in mappings:
  76 |             layer_names.extend(mappings[name])
  77 |         layer_names = list(set(layer_names))
  78 | 
  79 |         masks = self.maps[location].get_map_mask(
  80 |             patch_box=patch_box,
  81 |             patch_angle=patch_angle,
  82 |             layer_names=layer_names,
  83 |             canvas_size=self.canvas_size,
  84 |         )
  85 |         # masks = masks[:, ::-1, :].copy()
  86 |         # masks = masks.transpose(0, 2, 1)
  87 |         masks = masks.astype(np.bool)
  88 | 
  89 |         num_classes = len(self.classes)
  90 |         labels = np.zeros((num_classes, *self.canvas_size), dtype=np.long)
  91 |         for k, name in enumerate(self.classes):
  92 |             for layer_name in mappings[name]:
  93 |                 index = layer_names.index(layer_name)
  94 |                 labels[k, masks[index]] = 1
  95 | 
  96 |         semantic_masks = labels  # 这里已经进行了one_hot编码了
  97 |         num_cls = semantic_masks.shape[0]
  98 |         indices = np.arange(1, num_cls + 1).reshape(-1, 1, 1)
  99 |         semantic_indices = np.sum(semantic_masks * indices, axis=0)
 100 |         semantic_indices = np.where(semantic_indices > 5, 6, semantic_indices)
 101 | 
 102 |         results.update({
 103 |             'semantic_map': torch.from_numpy(semantic_masks),
 104 |             'semantic_indices': torch.from_numpy(semantic_indices).long(),
 105 |         })
 106 |         return results
```

### `projects/mmdet3d_plugin/datasets/mtl_nuscenes_dataset_ego.py` 第 255–287 行

```text
 255 |         future_indices = []
 256 |         for t in range(1, self.n_future + 1):
 257 |             index_t = index + t
 258 |             if index_t < len(self.data_infos) and self.data_infos[index_t]['scene_token'] == current_scene_token:
 259 |                 future_indices.append(index_t)
 260 |             else:
 261 |                 future_indices.append(-1)
 262 | 
 263 |         return previous_indices, future_indices
 264 | 
 265 |     @staticmethod
 266 |     def get_egopose_from_info(info):
 267 |         # ego2global transformation (lidar_ego)
 268 |         e2g_trans_matrix = np.zeros((4, 4), dtype=np.float32)
 269 |         e2g_rot = info['ego2global_rotation']
 270 |         e2g_trans = info['ego2global_translation']
 271 |         e2g_trans_matrix[:3, :3] = pyquaternion.Quaternion(
 272 |             e2g_rot).rotation_matrix
 273 |         e2g_trans_matrix[:3, 3] = np.array(e2g_trans)
 274 |         e2g_trans_matrix[3, 3] = 1.0
 275 | 
 276 |         return e2g_trans_matrix
 277 | 
 278 |     def get_egomotions(self, indices):
 279 |         # get ego_motion for each frame
 280 |         future_egomotions = []
 281 |         for index in indices:
 282 |             cur_info = self.data_infos[index]
 283 |             ego_motion = np.eye(4, dtype=np.float32)
 284 |             next_frame = index + 1
 285 | 
 286 |             # 如何处理 invalid frame
 287 |             if index != -1 and next_frame < len(self.data_infos) and self.data_infos[next_frame]['scene_token'] == cur_info['scene_token']:
```

### `projects/mmdet3d_plugin/datasets/mtl_nuscenes_dataset_ego.py` 第 376–402 行

```text
 376 | 
 377 |         # for past frames, we need images, camera paramters, depth(optional)
 378 |         img_infos = []
 379 |         for prev_index in prev_indices:
 380 |             if prev_index >= 0:
 381 |                 img_infos.append(self.data_infos[prev_index]['cams'])
 382 |             else:
 383 |                 # get the information of current frame for invalid frames
 384 |                 img_infos.append(info['cams'])
 385 | 
 386 |         # current frame
 387 |         img_infos.append(info['cams'])
 388 |         input_dict['img_info'] = img_infos
 389 | 
 390 |         input_dict['lidar2ego_rots'] = torch.tensor(pyquaternion.Quaternion(
 391 |             info['lidar2ego_rotation']).rotation_matrix)
 392 |         input_dict['lidar2ego_trans'] = torch.tensor(
 393 |             info['lidar2ego_translation'])
 394 | 
 395 |         # for future frames, we need detection labels
 396 |         if not self.test_submission:
 397 |             # generate detection labels for current + future frames
 398 |             label_frames = [index] + future_indices
 399 |             detection_ann_infos = []
 400 |             for label_frame in label_frames:
 401 |                 if label_frame >= 0:
 402 |                     detection_ann_infos.append(
```

### `projects/mmdet3d_plugin/datasets/mtl_nuscenes_dataset_ego.py` 第 470–512 行

```text
 470 | 
 471 |         # # reverse
 472 |         # gt_bboxes_3d.translate([-x for x in lidar2ego_translation])
 473 |         # gt_bboxes_3d.rotate(np.linalg.inv(lidar2ego_rotation.T))
 474 | 
 475 |         anns_results = dict(
 476 |             gt_bboxes_3d=gt_bboxes_3d,
 477 |             gt_labels_3d=gt_labels_3d,
 478 |             gt_names=gt_names_3d,
 479 |             instance_tokens=gt_instance_tokens,
 480 |             gt_valid_flag=gt_valid_flag,
 481 |             gt_vis_tokens=gt_vis_tokens,
 482 |         )
 483 | 
 484 |         return anns_results
 485 | 
 486 |     def get_map_ann_info(self, info):
 487 |         # get the annotations of HD maps
 488 |         vectors = self.vector_map.gen_vectorized_samples(
 489 |             info['location'], info['ego2global_translation'], info['ego2global_rotation'])
 490 | 
 491 |         # type0 = 'divider'
 492 |         # type1 = 'pedestrain'
 493 |         # type2 = 'boundary'
 494 | 
 495 |         for vector in vectors:
 496 |             pts = vector['pts']
 497 |             vector['pts'] = np.concatenate(
 498 |                 (pts, np.zeros((pts.shape[0], 1))), axis=1)
 499 | 
 500 |         return vectors
 501 | 
 502 |     def evaluate(self,
 503 |                  results,
 504 |                  metric='bbox',
 505 |                  logger=None,
 506 |                  jsonfile_prefix=None,
 507 |                  result_names=['pts_bbox'],
 508 |                  show=False,
 509 |                  out_dir=None,
 510 |                  pipeline=None):
 511 |         """Evaluation in nuScenes protocol.
 512 | 
```

### `projects/mmdet3d_plugin/datasets/nuscenes_dataset.py` 第 100–140 行

```text
 100 |         queue = [] # 设置时序依赖数据保存在队列里
 101 |         index_list = list(range(index-self.queue_length, index)) # 取出后三个
 102 |         random.shuffle(index_list) # 打乱
 103 |         index_list = sorted(index_list[1:]) #三选2，增加不确定性
 104 |         index_list.append(index) # 构成t-2， t-1， t的数据索引
 105 |         for i in index_list:
 106 |             i = max(0, i)
 107 |             input_dict = self.get_data_info(i)
 108 |             if input_dict is None:
 109 |                 return None
 110 |             self.pre_pipeline(input_dict)
 111 |             example = self.pipeline(input_dict)
 112 |             if self.filter_empty_gt and \
 113 |                     (example is None or ~(example['gt_labels_3d']._data != -1).any()):
 114 |                 return None
 115 |             queue.append(example) # 将以及处理好的数据送入队列，随后将送入网络
 116 |         return self.union2one(queue)
 117 | 
 118 |     def get_map_ann_info(self, info):
 119 | 
 120 |         # get the annotations of HD maps
 121 |         vectors = self.vector_map.gen_vectorized_samples(
 122 |             info['location'], info['ego2global_translation'], info['ego2global_rotation'])
 123 | 
 124 |         # type0 = 'divider'
 125 |         # type1 = 'pedestrain'
 126 |         # type2 = 'boundary'
 127 | 
 128 |         for vector in vectors:
 129 |             pts = vector['pts']
 130 |             vector['pts'] = np.concatenate(
 131 |                 (pts, np.zeros((pts.shape[0], 1))), axis=1)
 132 | 
 133 |         return vectors
 134 | 
 135 |     def union2one(self, queue):
 136 |         imgs_list = [each['img'].data for each in queue]
 137 |         metas_map = {}
 138 |         prev_scene_token = None
 139 |         prev_pos = None
 140 |         prev_angle = None
```

### `projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_head.py` 第 45–112 行

```text
  45 |                                  for row in [x_bounds, y_bounds, z_bounds]], dtype=torch.long)
  46 | 
  47 |     return bev_resolution, bev_start_position, bev_dimension
  48 | 
  49 | 
  50 | class BevFeatureSlicer(nn.Module):
  51 |     # crop the interested area in BEV feature for semantic map segmentation
  52 |     def __init__(self, grid_conf, map_grid_conf):
  53 |         super().__init__()
  54 | 
  55 |         if grid_conf == map_grid_conf:
  56 |             self.identity_mapping = True
  57 |         else:
  58 |             self.identity_mapping = False
  59 | 
  60 |             bev_resolution, bev_start_position, bev_dimension = calculate_birds_eye_view_parameters(
  61 |                 grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'],
  62 |             )
  63 | 
  64 |             map_bev_resolution, map_bev_start_position, map_bev_dimension = calculate_birds_eye_view_parameters(
  65 |                 map_grid_conf['xbound'], map_grid_conf['ybound'], map_grid_conf['zbound'],
  66 |             )
  67 | 
  68 |             self.map_x = torch.arange(
  69 |                 map_bev_start_position[0], map_grid_conf['xbound'][1], map_bev_resolution[0])
  70 | 
  71 |             self.map_y = torch.arange(
  72 |                 map_bev_start_position[1], map_grid_conf['ybound'][1], map_bev_resolution[1])
  73 | 
  74 |             # convert to normalized coords
  75 |             self.norm_map_x = self.map_x / (- bev_start_position[0])
  76 |             self.norm_map_y = self.map_y / (- bev_start_position[1])
  77 |             # vision 1 失败
  78 |             self.map_grid = torch.stack(torch.meshgrid(
  79 |                 self.norm_map_x, self.norm_map_y), dim=2).permute(1, 0, 2)
  80 |             # self.map_grid = torch.stack(torch.meshgrid(
  81 |             #     self.norm_map_x, self.norm_map_y, indexing='xy'), dim=2)
  82 | 
  83 |              # vision 2 test
  84 |             # self.map_grid = torch.stack(torch.meshgrid(
  85 |             #     self.norm_map_x, self.norm_map_y), dim=2)
  86 | 
  87 |     def forward(self, x):
  88 |         # x: bev feature map tensor of shape (b, c, h, w)
  89 |         if self.identity_mapping:
  90 |             return x
  91 |         else:
  92 |             grid = self.map_grid.unsqueeze(0).type_as(
  93 |                 x).repeat(x.shape[0], 1, 1, 1)
  94 | 
  95 |             return F.grid_sample(x, grid=grid, mode='bilinear', align_corners=True)
  96 | 
  97 | 
  98 | @HEADS.register_module()
  99 | class BEVFormerHead(DETRHead):
 100 |     """Head of Detr3D.
 101 |     Args:
 102 |         with_box_refine (bool): Whether to refine the reference points
 103 |             in the decoder. Defaults to False.
 104 |         as_two_stage (bool) : Whether to generate the proposal from
 105 |             the outputs of encoder.
 106 |         transformer (obj:`ConfigDict`): ConfigDict is used for building
 107 |             the Encoder and Decoder.
 108 |         bev_h, bev_w (int): spatial shape of BEV queries.
 109 |     """
 110 | 
 111 |     def __init__(self,
 112 |                  *args,
```

### `projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_head.py` 第 210–285 行

```text
 210 |     def init_weights(self):
 211 |         """Initialize weights of the DeformDETR head."""
 212 |         self.transformer.init_weights()
 213 |         # 如果有检测任务，才初始化分类分支
 214 |         if self.task.get('det'):
 215 |             if self.loss_cls.use_sigmoid:
 216 |                 bias_init = bias_init_with_prob(0.01)
 217 |                 for m in self.cls_branches:
 218 |                     nn.init.constant_(m[-1].bias, bias_init)
 219 | 
 220 |     @auto_fp16(apply_to=('mlvl_feats'))
 221 |     def forward(self, mlvl_feats, img_metas, prev_bev=None,  only_bev=False):
 222 |         """Forward function.
 223 |         Args:
 224 |             mlvl_feats (tuple[Tensor]): Features from the upstream
 225 |                 network, each is a 5D-tensor with shape
 226 |                 (B, N, C, H, W).
 227 |             prev_bev: previous bev featues
 228 |             only_bev: only compute BEV features with encoder. 
 229 |         Returns:
 230 |             all_cls_scores (Tensor): Outputs from the classification head, \
 231 |                 shape [nb_dec, bs, num_query, cls_out_channels]. Note \
 232 |                 cls_out_channels should includes background.
 233 |             all_bbox_preds (Tensor): Sigmoid outputs from the regression \
 234 |                 head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
 235 |                 Shape [nb_dec, bs, num_query, 9].
 236 |         """
 237 | 
 238 |         bs, num_cam, _, _, _ = mlvl_feats[0].shape
 239 |         dtype = mlvl_feats[0].dtype
 240 |         object_query_embeds = self.query_embedding.weight.to(dtype)
 241 |         bev_queries = self.bev_embedding.weight.to(dtype)
 242 | 
 243 |         bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
 244 |                                device=bev_queries.device).to(dtype)
 245 |         bev_pos = self.positional_encoding(bev_mask).to(dtype)
 246 | 
 247 |         if only_bev:  # only use encoder to obtain BEV features, TODO: refine the workaround
 248 |             return self.transformer.get_bev_features(
 249 |                 mlvl_feats,
 250 |                 bev_queries,
 251 |                 self.bev_h,
 252 |                 self.bev_w,
 253 |                 grid_length=(self.real_h / self.bev_h,
 254 |                              self.real_w / self.bev_w),
 255 |                 bev_pos=bev_pos,
 256 |                 img_metas=img_metas,
 257 |                 prev_bev=prev_bev,
 258 |             )
 259 |         else:
 260 |             outputs = self.transformer(
 261 |                 mlvl_feats,
 262 |                 bev_queries,
 263 |                 object_query_embeds,
 264 |                 self.bev_h,
 265 |                 self.bev_w,
 266 |                 grid_length=(self.real_h / self.bev_h,
 267 |                              self.real_w / self.bev_w),
 268 |                 bev_pos=bev_pos,
 269 |                 reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501
 270 |                 cls_branches=self.cls_branches if self.as_two_stage else None,
 271 |                 img_metas=img_metas,
 272 |                 prev_bev=prev_bev
 273 |         )
 274 | 
 275 |         bev_embed, hs, init_reference, inter_references = outputs
 276 | 
 277 |         # add segmentation task
 278 |         # TODO : 添加分割任务预测网络
 279 |         if self.task.get('seg'):
 280 |             # bev_queries(Tensor): (bev_h * bev_w, bs, c) 这是一个非常重要的地方，但是感觉这样的操作并无不同
 281 |             # 最初不做转换的版本，直接resize epoch 10 miou 0.07，这个肯定是不正确的 ❌
 282 |             # seg_bev = bev_embed.reshape(self.bev_h, self.bev_w, bs, -1).permute(2, 3, 0, 1)
 283 |             
 284 |             # 将bev空间特征从lidar系转换到图像系 
 285 |             
```

## 5. 与坐标转换相关的其他代码位置

- `projects/mmdet3d_plugin/apis/test.py:101`: `target_semantic_indices = data['semantic_indices'][0].unsqueeze(0)`
- `projects/mmdet3d_plugin/apis/test.py:102`: `one_hot = target_semantic_indices.new_full(semantic.shape, 0)`
- `projects/mmdet3d_plugin/apis/test.py:103`: `one_hot.scatter_(1, target_semantic_indices, 1)`
- `projects/mmdet3d_plugin/apis/test.py:129`: `target_semantic_indices = data['semantic_indices'][0].unsqueeze(0)`
- `projects/mmdet3d_plugin/apis/test.py:130`: `one_hot = target_semantic_indices.new_full(semantic.shape, 0)`
- `projects/mmdet3d_plugin/apis/test.py:131`: `one_hot.scatter_(1, target_semantic_indices, 1)`
- `projects/mmdet3d_plugin/apis/test.py:164`: `pred_semantic_indices = torch.sum(pred * indices, axis=1).int()`
- `projects/mmdet3d_plugin/apis/test.py:165`: `target_semantic_indices = data['semantic_indices'][0].cuda()`
- `projects/mmdet3d_plugin/apis/test.py:167`: `semantic_map_iou_val(pred_semantic_indices,`
- `projects/mmdet3d_plugin/apis/test.py:168`: `target_semantic_indices)`
- `projects/mmdet3d_plugin/apis/test.py:261`: `pred_semantic_indices = torch.sum(pred * indices, axis=1).int()`
- `projects/mmdet3d_plugin/apis/test.py:262`: `target_semantic_indices = data['semantic_indices'][0].cuda()`
- `projects/mmdet3d_plugin/apis/test.py:264`: `semantic_map_iou_val(pred_semantic_indices,`
- `projects/mmdet3d_plugin/apis/test.py:265`: `target_semantic_indices)`
- `projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_head.py:95`: `return F.grid_sample(x, grid=grid, mode='bilinear', align_corners=True)`
- `projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_head.py:164`: `self.feat_cropper = BevFeatureSlicer(self.det_grid_conf, self.map_grid_conf)`
- `projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_head.py:288`: `# seg_bev = torch.rot90(seg_bev, k=-1, dims=[2, 3])`
- `projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_head.py:300`: `#seg_bev = torch.rot90(seg_bev, k=1, dims=[2, 3])`
- `projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_head.py:303`: `#seg_bev = torch.rot90(seg_bev, k=-1, dims=[2, 3])`
- `projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_head.py:307`: `#seg_bev = torch.rot90(seg_bev, k=-1, dims=[2, 3])`
- `projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_head.py:317`: `seg_bev = self.feat_cropper(seg_bev)`
- `projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_seg_head.py:99`: `return F.grid_sample(x, grid=grid, mode='bilinear', align_corners=True)`
- `projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_seg_head.py:166`: `self.feat_cropper = BevFeatureSlicer(self.det_grid_conf, self.map_grid_conf)`
- `projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_seg_head.py:286`: `seg_bev = torch.rot90(seg_bev, k=-1, dims=[2, 3])`
- `projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_seg_head.py:289`: `seg_bev = self.feat_cropper(seg_bev)`
- `projects/mmdet3d_plugin/bevformer/detectors/bevformer.py:115`: `semantic_indices,`
- `projects/mmdet3d_plugin/bevformer/detectors/bevformer.py:143`: `seg_loss_inputs = [semantic_indices, outs]`
- `projects/mmdet3d_plugin/bevformer/detectors/bevformer.py:151`: `seg_loss_inputs = [semantic_indices, outs]`
- `projects/mmdet3d_plugin/bevformer/detectors/bevformer.py:204`: `semantic_indices=None,`
- `projects/mmdet3d_plugin/bevformer/detectors/bevformer.py:255`: `gt_labels_3d, semantic_indices,`
- `projects/mmdet3d_plugin/bevformer/detectors/bevformer_fp16.py:35`: `semantic_indices=None,`
- `projects/mmdet3d_plugin/bevformer/modules/decoder.py:69`: `reference_points=None,`
- `projects/mmdet3d_plugin/bevformer/modules/decoder.py:77`: `reference_points (Tensor): The reference`
- `projects/mmdet3d_plugin/bevformer/modules/decoder.py:92`: `intermediate_reference_points = []`
- `projects/mmdet3d_plugin/bevformer/modules/decoder.py:95`: `reference_points_input = reference_points[..., :2].unsqueeze(`
- `projects/mmdet3d_plugin/bevformer/modules/decoder.py:100`: `reference_points=reference_points_input,`
- `projects/mmdet3d_plugin/bevformer/modules/decoder.py:108`: `assert reference_points.shape[-1] == 3`
- `projects/mmdet3d_plugin/bevformer/modules/decoder.py:110`: `new_reference_points = torch.zeros_like(reference_points)`
- `projects/mmdet3d_plugin/bevformer/modules/decoder.py:111`: `new_reference_points[..., :2] = tmp[`
- `projects/mmdet3d_plugin/bevformer/modules/decoder.py:112`: `..., :2] + inverse_sigmoid(reference_points[..., :2])`
- `projects/mmdet3d_plugin/bevformer/modules/decoder.py:113`: `new_reference_points[..., 2:3] = tmp[`
- `projects/mmdet3d_plugin/bevformer/modules/decoder.py:114`: `..., 4:5] + inverse_sigmoid(reference_points[..., 2:3])`
- `projects/mmdet3d_plugin/bevformer/modules/decoder.py:116`: `new_reference_points = new_reference_points.sigmoid()`
- `projects/mmdet3d_plugin/bevformer/modules/decoder.py:118`: `reference_points = new_reference_points.detach()`
- `projects/mmdet3d_plugin/bevformer/modules/decoder.py:123`: `intermediate_reference_points.append(reference_points)`
- `projects/mmdet3d_plugin/bevformer/modules/decoder.py:127`: `intermediate_reference_points)`
- `projects/mmdet3d_plugin/bevformer/modules/decoder.py:129`: `return output, reference_points`
- `projects/mmdet3d_plugin/bevformer/modules/decoder.py:238`: `reference_points=None,`
- `projects/mmdet3d_plugin/bevformer/modules/decoder.py:259`: `reference_points (Tensor):  The normalized reference`
- `projects/mmdet3d_plugin/bevformer/modules/decoder.py:310`: `if reference_points.shape[-1] == 2:`
- `projects/mmdet3d_plugin/bevformer/modules/decoder.py:313`: `sampling_locations = reference_points[:, :, None, :, None, :] \`
- `projects/mmdet3d_plugin/bevformer/modules/decoder.py:316`: `elif reference_points.shape[-1] == 4:`
- `projects/mmdet3d_plugin/bevformer/modules/decoder.py:317`: `sampling_locations = reference_points[:, :, None, :, None, :2] \`
- `projects/mmdet3d_plugin/bevformer/modules/decoder.py:319`: `* reference_points[:, :, None, :, None, 2:] \`
- `projects/mmdet3d_plugin/bevformer/modules/decoder.py:323`: `f'Last dim of reference_points must be'`
- `projects/mmdet3d_plugin/bevformer/modules/decoder.py:324`: `f' 2 or 4, but get {reference_points.shape[-1]} instead.')`
- `projects/mmdet3d_plugin/bevformer/modules/encoder.py:51`: `def get_reference_points(H, W, Z=8, num_points_in_pillar=4, dim='3d', bs=1, device='cuda', dtype=torch.float):`
- `projects/mmdet3d_plugin/bevformer/modules/encoder.py:58`: `reference_points should be.`
- `projects/mmdet3d_plugin/bevformer/modules/encoder.py:94`: `@force_fp32(apply_to=('reference_points', 'img_metas'))`
- `projects/mmdet3d_plugin/bevformer/modules/encoder.py:95`: `def point_sampling(self, reference_points, pc_range,  img_metas):`
- `projects/mmdet3d_plugin/bevformer/modules/encoder.py:101`: `lidar2img = reference_points.new_tensor(lidar2img)  # (B, N, 4, 4) # 坐标变换矩阵，将3d点映射到2d图像平面上去 #[1, 6, 4, 4]`
- `projects/mmdet3d_plugin/bevformer/modules/encoder.py:102`: `reference_points = reference_points.clone() # [1, 4, 22500, 3]`
- `projects/mmdet3d_plugin/bevformer/modules/encoder.py:104`: `reference_points[..., 0:1] = reference_points[..., 0:1] * \`
- `projects/mmdet3d_plugin/bevformer/modules/encoder.py:106`: `reference_points[..., 1:2] = reference_points[..., 1:2] * \`
- `projects/mmdet3d_plugin/bevformer/modules/encoder.py:108`: `reference_points[..., 2:3] = reference_points[..., 2:3] * \`
- `projects/mmdet3d_plugin/bevformer/modules/encoder.py:111`: `reference_points = torch.cat(`
- `projects/mmdet3d_plugin/bevformer/modules/encoder.py:112`: `(reference_points, torch.ones_like(reference_points[..., :1])), -1) # [1, 4, 22500, 4]`
- `projects/mmdet3d_plugin/bevformer/modules/encoder.py:114`: `reference_points = reference_points.permute(1, 0, 2, 3)`
- `projects/mmdet3d_plugin/bevformer/modules/encoder.py:116`: `D, B, num_query = reference_points.size()[:3]`
- `projects/mmdet3d_plugin/bevformer/modules/encoder.py:119`: `reference_points = reference_points.view(`
- `projects/mmdet3d_plugin/bevformer/modules/encoder.py:125`: `reference_points_cam = torch.matmul(lidar2img.to(torch.float32),`
- `projects/mmdet3d_plugin/bevformer/modules/encoder.py:126`: `reference_points.to(torch.float32)).squeeze(-1) # 计算处每个参考点在6个view的坐标 [4, 1, 6, 22500, 4], 最后一维是坐标的齐次值`
- `projects/mmdet3d_plugin/bevformer/modules/encoder.py:129`: `bev_mask = (reference_points_cam[..., 2:3] > eps)`
- `projects/mmdet3d_plugin/bevformer/modules/encoder.py:130`: `reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(`
- `projects/mmdet3d_plugin/bevformer/modules/encoder.py:131`: `reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps) # 坐标系转换？x，y坐标/z 将从齐次坐标系转换到笛卡尔直角坐标系，这时候还是像素坐标呢？`
- `projects/mmdet3d_plugin/bevformer/modules/encoder.py:133`: `reference_points_cam[..., 0] /= img_metas[0]['img_shape'][0][1] # 这里的物理意义是什么？归一化了`
- `projects/mmdet3d_plugin/bevformer/modules/encoder.py:134`: `reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0][0] #[4, 1, 6, 22500, 2]现在的维度`
- `projects/mmdet3d_plugin/bevformer/modules/encoder.py:136`: `bev_mask = (bev_mask & (reference_points_cam[..., 1:2] > 0.0)`
- `projects/mmdet3d_plugin/bevformer/modules/encoder.py:137`: `& (reference_points_cam[..., 1:2] < 1.0)`
- `projects/mmdet3d_plugin/bevformer/modules/encoder.py:138`: `& (reference_points_cam[..., 0:1] < 1.0)`
- `projects/mmdet3d_plugin/bevformer/modules/encoder.py:139`: `& (reference_points_cam[..., 0:1] > 0.0)) # 归一化之后将越界的点过滤`
- `projects/mmdet3d_plugin/bevformer/modules/encoder.py:146`: `reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4) # [6, 1, 22500, 4, 2]`
- `projects/mmdet3d_plugin/bevformer/modules/encoder.py:149`: `return reference_points_cam, bev_mask`
- `projects/mmdet3d_plugin/bevformer/modules/encoder.py:172`: `reference_points (Tensor): The reference`
- `projects/mmdet3d_plugin/bevformer/modules/encoder.py:188`: `ref_3d = self.get_reference_points(`
- `projects/mmdet3d_plugin/bevformer/modules/encoder.py:190`: `ref_2d = self.get_reference_points(`
- `projects/mmdet3d_plugin/bevformer/modules/encoder.py:193`: `reference_points_cam, bev_mask = self.point_sampling(`
- `projects/mmdet3d_plugin/bevformer/modules/encoder.py:227`: `reference_points_cam=reference_points_cam,`
- `projects/mmdet3d_plugin/bevformer/modules/encoder.py:301`: `reference_points_cam=None,`
- `projects/mmdet3d_plugin/bevformer/modules/encoder.py:369`: `reference_points=ref_2d,`
- `projects/mmdet3d_plugin/bevformer/modules/encoder.py:390`: `reference_points=ref_3d,`
- `projects/mmdet3d_plugin/bevformer/modules/encoder.py:391`: `reference_points_cam=reference_points_cam,`
- `projects/mmdet3d_plugin/bevformer/modules/spatial_cross_attention.py:75`: `@force_fp32(apply_to=('query', 'key', 'value', 'query_pos', 'reference_points_cam'))`
- `projects/mmdet3d_plugin/bevformer/modules/spatial_cross_attention.py:83`: `reference_points=None,`
- `projects/mmdet3d_plugin/bevformer/modules/spatial_cross_attention.py:85`: `reference_points_cam=None,`
- `projects/mmdet3d_plugin/bevformer/modules/spatial_cross_attention.py:104`: `reference_points (Tensor):  The normalized reference`
- `projects/mmdet3d_plugin/bevformer/modules/spatial_cross_attention.py:136`: `D = reference_points_cam.size(3)`
- `projects/mmdet3d_plugin/bevformer/modules/spatial_cross_attention.py:146`: `reference_points_rebatch = reference_points_cam.new_zeros(`
- `projects/mmdet3d_plugin/bevformer/modules/spatial_cross_attention.py:150`: `for i, reference_points_per_img in enumerate(reference_points_cam):`
- `projects/mmdet3d_plugin/bevformer/modules/spatial_cross_attention.py:153`: `reference_points_rebatch[j, i, :len(index_query_per_img)] = reference_points_per_img[j, index_query_per_img]`
- `projects/mmdet3d_plugin/bevformer/modules/spatial_cross_attention.py:163`: `reference_points=reference_points_rebatch.view(bs*self.num_cams, max_len, D, 2), spatial_shapes=spatial_shapes,`
- `projects/mmdet3d_plugin/bevformer/modules/spatial_cross_attention.py:280`: `reference_points=None,`
- `projects/mmdet3d_plugin/bevformer/modules/spatial_cross_attention.py:299`: `reference_points (Tensor):  The normalized reference`
- `projects/mmdet3d_plugin/bevformer/modules/spatial_cross_attention.py:350`: `if reference_points.shape[-1] == 2:`
- `projects/mmdet3d_plugin/bevformer/modules/spatial_cross_attention.py:360`: `bs, num_query, num_Z_anchors, xy = reference_points.shape`
- `projects/mmdet3d_plugin/bevformer/modules/spatial_cross_attention.py:361`: `reference_points = reference_points[:, :, None, None, None, :, :]`
- `projects/mmdet3d_plugin/bevformer/modules/spatial_cross_attention.py:367`: `sampling_locations = reference_points + sampling_offsets`
- `projects/mmdet3d_plugin/bevformer/modules/spatial_cross_attention.py:374`: `elif reference_points.shape[-1] == 4:`
- `projects/mmdet3d_plugin/bevformer/modules/spatial_cross_attention.py:378`: `f'Last dim of reference_points must be'`
- `projects/mmdet3d_plugin/bevformer/modules/spatial_cross_attention.py:379`: `f' 2 or 4, but get {reference_points.shape[-1]} instead.')`
- （仅展示前 110 处；完整源文件在本地仓库。）

## 6. 后续判断需要核对的问题

- `RasterizeMapVectors` 的输入 vectors 属于 Ego 还是 LiDAR 坐标？
- BEV encoder 的 `(row, col)` 对应哪个物理轴及正方向？
- `BevFeatureSlicer` 采用哪个 reference frame、网格中心与方向？
- Train 与 Val 是否使用相同地图 rasterization 和同一坐标定义？
- 原始 checkpoint 的训练代码版本与当前推理代码是否一致？
- CCW90 后仍退化的帧是否集中于特定地图或场景？
