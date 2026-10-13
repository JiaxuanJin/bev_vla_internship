# DriveLM 数据集结构与标注格式说明

## 1. 数据集定位与版本

DriveLM 通过驾驶场景中的图像和问题—答案（QA）关联视觉感知、预测、规划与驾驶决策。实验使用 **DriveLM–nuScenes v1.1** 相关资源，并通过 nuScenes `sample token` 关联 BEVFormer 感知数据。它不是只包含二维目标检测框的普通图片数据集。

数据有两类完全不同的用途：

- **DriveLM 官方 QA 训练数据**：六路相机图像 + 问答 JSON；通过官方 `extract_data.py → convert_data.py → convert2llama.py` 转换成 LLaMA-Adapter 使用的格式。
- **BEVFormer 感知数据**：图片 + 时间、标定、车辆位姿及 nuScenes 3D 真值/地图数据；通过本项目脚本构造 PKL 供 3D 检测和 BEV 地图分割使用。

**不要将 `test_llama.json` 和 `nuscenes_infos_temporal_*.pkl` 混为一谈。** 前者是语言问答记录，后者是感知模型的样本信息。

## 2. 本地真实数据目录

```text
bev_vla/
├── data/
│   ├── drivelm/
│   │   ├── QA_dataset_nus/
│   │   │   ├── v1_1_train_nus.json
│   │   │   └── v1_1_val_nus_q_only.json
│   │   ├── nuscenes/samples/
│   │   │   ├── CAM_FRONT/                 # 六路之一
│   │   │   ├── CAM_FRONT_LEFT/
│   │   │   ├── CAM_FRONT_RIGHT/
│   │   │   ├── CAM_BACK/
│   │   │   ├── CAM_BACK_LEFT/
│   │   │   └── CAM_BACK_RIGHT/
│   │   ├── val_data/                     # DriveLM val 六路图像
│   │   ├── nuscenes_infos_temporal_drivelm799_officialscene.pkl
│   │   └── nuscenes_infos_temporal_drivelm799_independent.pkl
│   └── nuscenes_trainval_meta/v1.0-trainval/   # nuScenes 官方标注/元数据
├── third_party/DriveLM/challenge/        # 下载的三个官方 QA 转换脚本
├── outputs/drivelm_official_preprocess/full/
│   ├── test.json
│   ├── test_eval.json
│   ├── test_llama.json
│   ├── summary.json
│   └── official_extract_stdout.log
└── BEVFormer/data/nuscenes/              # 用于 Mini 实验的数据
```

`data/` 在本机可能包含符号链接；以上是项目逻辑路径，不代表必须复制所有图像到 GitHub。验证集完整图像还保留在 G 盘下载/解压目录中。

## 3. 数据规模与全量预处理结果

| 内容 | 10 月 12 日实际结果 |
|---|---:|
| DriveLM v1.1 训练场景 | 696 |
| 训练关键帧 | 4,072 |
| 六路训练相机 JPEG | 24,432 = 4,072 × 6 |
| 官方转换后 QA 记录 | 29,450 |
| `test_llama.json` 独立图像路径 | 24,432 |
| 缺失图像路径 / 不合法对话结构 | 0 / 0 |
| 转换执行耗时 | 49.38 秒 |
| DriveLM 验证部分 | 149 个场景、799 个关键帧（实际用于与 nuScenes 官方 val 匹配的子集） |

官方 Challenge README 的示例曾列 29,448 条训练 QA；本次实跑结果是 **29,450**。这 2 条差异可能涉及数据版本和转换细节，**尚未完成逐条差分分析**。正式提交时以本机 `summary.json` 为准，不能声称“逐条完全复现参考数量”。

## 4. 标注关联：为什么 token、标定、场景这么重要？

| 概念 | 大白话 | 用在本项目哪里 |
|---|---|---|
| `scene` / 场景 | 一段道路行驶记录 | 标识视频所属驾驶片段；地图查询需要真实场景 |
| `sample_token` | 一帧数据的唯一身份证 | 把 DriveLM QA、六张图和 nuScenes 3D GT 按帧一一对齐 |
| `sample_data` | 某个传感器的具体采样文件 | 从 token 找到六路图像、参考点云等 |
| `calibrated_sensor` | 相机内外参 | 将 3D 坐标投影到相机图像（`lidar2img`） |
| `ego_pose` / `can_bus` | 车辆在世界中的位置、方向、运动 | 关联时空 BEV、坐标转换及地图裁剪 |
| `sample_annotation` | GT 3D 物体标签 | 计算检测 mAP/NDS |
| 地图层/栅格标签 | 可行驶道路及地图要素的空间真值 | 与分割预测比较，计算 IoU |

**重要区别**：nuScenes `sample token` 标识一帧，`scene_token` 标识一段场景，不能互换。实际完整字段名及嵌套结构以现有 JSON / PKL 为准。

## 5. 六路相机与模型输入

`CAM_FRONT / CAM_FRONT_LEFT / CAM_FRONT_RIGHT / CAM_BACK / CAM_BACK_LEFT / CAM_BACK_RIGHT` 构成 360° 环视视角。DriveLM 799 帧检测数据加载测试在 0、399、798 号样本均通过：图像张量单帧 `(6, 3, 736, 1280)`，六路 `lidar2img` 矩阵齐全。加入模型 batch 维后形状 `(1, 6, 3, 736, 1280)`。

## 6. DriveLM 官方 QA 转换流程（已实跑）

在项目根目录执行过：

```bash
python scripts/run_drivelm_official_preprocess.py --mode smoke
python scripts/run_drivelm_official_preprocess.py --mode full
```

流程及输出如下：

```text
v1_1_train_nus.json
    └── 官方 extract_data.extract_data()   → test.json
            └── 官方 convert_data.loop_test() → test_eval.json
                    └── 官方 convert2llama.convert2llama() → test_llama.json
```

我们自写的 `run_drivelm_official_preprocess.py` 只负责**指定完整训练 JSON、隔离输出目录、调用官方函数、固定选项顺序随机种子、校验图片路径与数据结构**；转换逻辑仍是从官方 DriveLM 仓库取得的原始函数。

校验通过的 `test_llama.json` 单条记录结构：

```python
{
    "id": "<唯一标识，示意>",
    "image": ["<相机图像路径1>", "<相机图像路径2>", "... 共6项"],
    "conversations": [
        {"from": "human", "value": "<问题/指令>"},
        {"from": "gpt", "value": "<答案>"}
    ]
}
```

**这段是字段说明示意，不是实际原始 QA 文本**。实测 29,450 条均通过 `id/image/conversations` 格式测试。正式生成文件由于大小较大，按数据集许可处理，**不纳入 Git 仓库**。

## 7. BEVFormer 数据 PKL 与两种时序策略

`data/drivelm/nuscenes_infos_temporal_drivelm799_independent.pkl`：供独立帧 3D 检测使用，避免将 DriveLM 稀疏问答帧当作连续采样错误复用历史 BEV。

`data/drivelm/nuscenes_infos_temporal_drivelm799_officialscene.pkl`：保留真实 `scene_token`，供地图 GT 查询及 BEV 分割使用；分割模型显式设置 `video_test_mode=False`，禁止跨稀疏帧复用历史 BEV。

这两份感知 PKL 是**本项目为稀疏子集额外构建的适配文件**，**不是** DriveLM 官方 `extract_data.py` 的输出，也不是标准完整 nuScenes Trainval 训练索引。

## 8. 元数据存在不等于完整训练资源

本地 `BEVFormer/data/nuscenes/nuscenes_infos_temporal_train.pkl` 实测仅有 **323 帧、8 场景**，验证 PKL **81 帧、2 场景**，属于 nuScenes Mini 级别。完整 `v1.0-trainval` JSON 元数据存在，但数据盘的 `samples/LIDAR_TOP` 仍为 Mini 规模（404），完整 LiDAR、Sweeps 及 CAN Bus 不齐；因此**未声称已经生成完整 BEVFormer Trainval 训练 PKL**。

## 9. 官方参考资料

- [DriveLM 原始仓库](https://github.com/OpenDriveLab/DriveLM)
- [DriveLM Challenge：数据转换、baseline 使用方式](https://github.com/OpenDriveLab/DriveLM/blob/main/challenge/README.md)
- [BEVFormer 官方仓库](https://github.com/fundamentalvision/BEVFormer)
- [nuScenes 官方主页](https://www.nuscenes.org/)
