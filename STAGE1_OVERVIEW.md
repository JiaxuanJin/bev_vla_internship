# DriveLM × BEVFormer-Small｜第一阶段项目说明

本文件可作为 **GitHub 仓库首页 README 的参考稿**；**不会自动覆盖现有 README.md**。

## 工作目标

建立并验证摄像头 BEV 感知基线，完成 DriveLM–nuScenes 官方问答数据转换、BEVFormer-Small 检测与 BEV 地图分割 GPU 实验，整理可复现的代码与结果。

## 核心结果

| 数据 | 3D mAP | NDS | BEV 前景 mIoU |
|---|---:|---:|---:|
| nuScenes Mini (81 帧) | 32.50% | 37.71% | 6.38% |
| DriveLM 799 帧 **独立帧子集** | **13.39%** | **25.79%** | **5.12%** |

指标真实计算，但自定义 799 帧独立帧协议不等同于完整官方 nuScenes Val；分割准确率偏低，空间对齐仍待研究。没有进行重训练。

## 文档导航

- [阶段一交付导航](docs/stage1/00_阶段一交付导航.md)
- [数据集与标注格式](docs/stage1/01_DriveLM数据集结构与标注说明.md)
- [BEVFormer 原理和代码结构](docs/stage1/02_BEVFormer原理与代码结构.md)
- [基线性能报告](docs/stage1/03_基线性能测试报告.md)
- [可视化索引](docs/stage1/04_可视化成果索引.md)
- [复现与交付检查](docs/stage1/05_复现和GitHub提交前检查.md)
- [2026-10-12 Logbook](docs/logbook_2026-10-12.md)

## 运行环境与代码

WSL Ubuntu 22.04、PyTorch 2.8.0+cu128、NVIDIA RTX 5080 Laptop GPU。实验脚本在 `scripts/`，官方依赖和权重需按开源项目说明另行安装/取得。

一键检查本地交付是否完整：

```bash
cd ~/projects/bev_vla
python scripts/check_stage1_submission.py
```

第三方来源：[BEVFormer](https://github.com/fundamentalvision/BEVFormer) · [DriveLM](https://github.com/OpenDriveLab/DriveLM) · [nuScenes Devkit](https://github.com/nutonomy/nuscenes-devkit)
