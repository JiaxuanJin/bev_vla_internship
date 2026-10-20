# 2026-10-20 BEVFormer 分割根因排查证据索引

本目录从本机 `outputs/drivelm_check/root_cause/` 筛选轻量实验产物；
完整权重、训练产物、数据集、环境文件及冗长编译日志均未纳入。
独立帧、完整 Val、Mini Val 时序的 IoU 不应直接横向比较。

## 文件清单

| 相对 Git 路径 | 大小 KiB | SHA256 |
|---|---:|---|
| `results/root_cause_2026-10-20/author_runtime/step50_mini_val_tail.txt` | 7.2 | `eb82ef8afd1d5a0b05eebd3ca651b56f8346364fd58fe83db462964eeb609101` |
| `results/root_cause_2026-10-20/author_runtime/step54_mini_ab_tail.txt` | 7.2 | `cf5e78a912c4003a69f03ee816c43566d4e70b7f18eb18963f381ffe7a94144c` |
| `results/root_cause_2026-10-20/camera_lidar_overlay/results.json` | 8.0 | `5c1bb49980887221928399bbf927f9871ed1eae528dc6e75c91969a7c55d0a00` |
| `results/root_cause_2026-10-20/camera_lidar_overlay/val_0000_lidar_overlay.jpg` | 746.1 | `d3d2420aed7e5a3912ebe81e4e61890c463bb1c461eb878f221c413c0e9e6626` |
| `results/root_cause_2026-10-20/camera_lidar_overlay/val_0040_lidar_overlay.jpg` | 891.9 | `c07c519b5ab73aec580a3fd7bd2544298f850e7291b33c0c295e9e13e5671aac` |
| `results/root_cause_2026-10-20/camera_lidar_overlay/val_0080_lidar_overlay.jpg` | 901.3 | `4d55adda980e4ef72a060fcc48da333e14d333bb41fb75b78ec3abd8ead30eb1` |
| `results/root_cause_2026-10-20/step34_full_val_orientation_ab/per_frame.csv` | 837.0 | `874cf62572dcf3a6708a612616ba970c605468979e116cbf1a3af20e4f383a76` |
| `results/root_cause_2026-10-20/step34_full_val_orientation_ab/results.json` | 2.6 | `6490a329fddbc7ca74ad441e5f1c7c1b5d9bc16eae7a0c414661cf6bfdde6a80` |
| `results/root_cause_2026-10-20/step35_coordinate_audit/audit.md` | 12.5 | `04d4454f663acb8ba047d161d9e1c81d7c5dd9e7cc92fd3436420b59b00d41b9` |
| `results/root_cause_2026-10-20/step35_coordinate_audit/scene_summary.csv` | 9.2 | `d02a26384c23cd61e3f17ee036b86aa2ee3ff0fea434559882236f541dc70a42` |
| `results/root_cause_2026-10-20/step35_coordinate_audit/worsened_frames.csv` | 20.5 | `589147a1497d5f058a50289a7669825763f3d3132e7837ea74bcaca2718bfc29` |
| `results/root_cause_2026-10-20/step36_coordinate_contract/focus_frames.csv` | 1.8 | `67c496b267317279958c144facab61354fdbb66fa549a576c385643cc1a947b2` |
| `results/root_cause_2026-10-20/step36_coordinate_contract/source_evidence.md` | 52.1 | `fba0a37fb228b8e3921fd187d0d026b823b194827cd8994dbbdde4da10d5fa9b` |
| `results/root_cause_2026-10-20/step37_targeted_geometry/images/val_0000.png` | 28.3 | `54e4e3dd8893ebb1f68b397516c60446377afce1f303a07f08548320c6a4b85f` |
| `results/root_cause_2026-10-20/step37_targeted_geometry/images/val_3000.png` | 26.1 | `c09ef3b16314fb313a22d5d3d7ae9fc4a437282121cf6f2b72724ef7c07acbcb` |
| `results/root_cause_2026-10-20/step37_targeted_geometry/images/val_3715.png` | 25.9 | `70f1e3be6019becbb81848bb353f6c80c79f4dee6422ba548091438824255f6e` |
| `results/root_cause_2026-10-20/step37_targeted_geometry/images/val_3718.png` | 25.6 | `d576d7f8e57686a5dee6a9fde7d9842c7bb652e2379503113dbd7c65f13ba587` |
| `results/root_cause_2026-10-20/step37_targeted_geometry/images/val_3719.png` | 25.9 | `ace131c0c6dd97b3b16842b33508963e791b5ed32df9d1568fa177e2b685bd1e` |
| `results/root_cause_2026-10-20/step37_targeted_geometry/images/val_6018.png` | 37.9 | `a051090dd32179448da09792e7e5bf5777da46cd8fdaee97864e6cafa8c6f5a8` |
| `results/root_cause_2026-10-20/step37_targeted_geometry/per_frame.csv` | 2.1 | `d3c3e72beaf1b8f385b083dfdea0ab3da4b49a994a6187d518882ed16a08ed1b` |
| `results/root_cause_2026-10-20/step37_targeted_geometry/results.json` | 7.7 | `6234ddda14626425054285ceaf15c097dc63f031a50082b25506e9e53e7d6ccf` |
| `results/root_cause_2026-10-20/step38_protocol_audit/audit.md` | 1.1 | `b8d3e82e197eccb831fe82eba443d98b7d6a5f3eda53ddee9a972791ac993337` |
| `results/root_cause_2026-10-20/step38_protocol_audit/results.json` | 5.8 | `bec57af6d71849b84d919a343c9afeaf2ad3eaea6574890c6cbb94ca67c56198` |
| `results/root_cause_2026-10-20/step39_orientation_screen/per_frame.csv` | 7.5 | `69a6f78b822e8984cdadfc5672a6f118bcddee55e35bf5c9071a47d66da1ab8a` |
| `results/root_cause_2026-10-20/step39_orientation_screen/results.json` | 17.8 | `b0ff2e3b92023f5be316f6aca8cb60c8f173c5d6cfd548d991e196931be66a64` |
| `results/root_cause_2026-10-20/step40_train_head_ab_repair/per_frame_val.csv` | 1.4 | `c4e06eb48be7a5acc8a39002643482a4a7fa12c987ce75cb1de5cf223dd37e8a` |
| `results/root_cause_2026-10-20/step40_train_head_ab_repair/results.json` | 3.3 | `7999e44173c033a2ff83891546038553b1ceb538f956855bcfe7b239d02b2285` |
| `results/root_cause_2026-10-20/step40_train_head_ab_repair/train_loss.csv` | 3.1 | `07775747e4ac2b9c239af2bf136f74eeba708cc67ed581be8403d3123bfe4f1b` |
| `results/root_cause_2026-10-20/step41_physical_adapter/coordinate_contract.json` | 5.1 | `2297d403fd786b6531acb41f89997e5fd31552cfeb19378ae71d24f5932f11ea` |
| `results/root_cause_2026-10-20/step42_upstream_entry_ab/per_frame_manual.csv` | 2.7 | `8e60e5ff38a1fb8df56adf18a049edb165124aa7595d4fdf8d4bdccf4455bc5e` |
| `results/root_cause_2026-10-20/step42_upstream_entry_ab/results.json` | 1.1 | `04c803322c36553b0e8419e8f28cba1a29c34aa35ab0b97bee2e7014bee71716` |
| `results/root_cause_2026-10-20/step43_issue37_map_y_ab/baseline_seg_minival_metrics.json` | 1.7 | `962a9094f9b1a3b50bd1770e15603af218ac1ceadc3f49ee9ea32bf3a968329c` |
| `results/root_cause_2026-10-20/step43_issue37_map_y_ab/baseline_seg_minival_per_frame.csv` | 12.3 | `c2b2ca99004162fe463c61da36a493f9a91a4af65b3a9f07ec4368652f914b83` |
| `results/root_cause_2026-10-20/step43_issue37_map_y_ab/comparison.json` | 0.3 | `07fc28a200335cff1d4bdd1cb89ee374ee780de13051b4d81d713c6720313e02` |
| `results/root_cause_2026-10-20/step43_issue37_map_y_ab/issue37_seg_minival_metrics.json` | 1.7 | `042bef9023d65a19efd3d786fcd968d27bf3dcf952efe99750527b617958ef0e` |
| `results/root_cause_2026-10-20/step43_issue37_map_y_ab/issue37_seg_minival_per_frame.csv` | 12.2 | `e469f53eebe29cc9b4400a5e8e81e8978af733378c34dffc3ca8ad2f872af56b` |
| `results/root_cause_2026-10-20/step44_gt_alignment/results.json` | 3.6 | `3e5c66504e96edb76ccdced7e5638cdffcc21a0c12ca0217ec0fbcfbda00b9ac` |
| `results/root_cause_2026-10-20/step45_cuda_operator_audit/results.json` | 0.8 | `90608d95ca074dc3f1a4de858462ed9ec2dafb3ced73284954e959771f032173` |
| `results/root_cause_2026-10-20/step46_projection_runtime/results.json` | 5.7 | `557d01e736c8fc5b5685eb45a225303eb766dceade16daa541d716dda56065a1` |
| `results/root_cause_2026-10-20/step51_coordinate_contract/audit.md` | 7.3 | `a75ad8be10a478bdaddee16df101c787396a2c860e01724e326de170a4411818` |
| `results/root_cause_2026-10-20/step51_coordinate_contract/results.json` | 36.5 | `939ee7aa6131161a7dbf44a957103c8b2dd6e65d0a264f38a78e7b7835adfb77` |
| `results/root_cause_2026-10-20/step52_runtime_coordinate_contract/audit.md` | 0.8 | `521fcefdf893a7edb7f7ab494697131819c173af087cd80827bf312325ac85c6` |
| `results/root_cause_2026-10-20/step52_runtime_coordinate_contract/results.json` | 3.0 | `2396e6458fe66ae1e9230f139e1edaecd478fe32b13a987777d1994b4a7aa053` |
| `results/root_cause_2026-10-20/step53_runtime_calibration_ab/audit.md` | 0.6 | `aed58ce9edf2e89ea8ac6aecfda19380e74c051ba1b4eb7d9400e04ebbc28331` |
| `results/root_cause_2026-10-20/step53_runtime_calibration_ab/results.json` | 2.2 | `0fab1005578ce57b57e2bda4a3d3662cc948e4309b5ae0064efe8bb2c6c00a62` |
| `results/root_cause_2026-10-20/step54_author_mini_calibration_ab/audit.md` | 0.8 | `2ae406efd3d403eb4cea3af63aaa4cef2b6ca2bb198b564a8117f110dd8bca4e` |
| `results/root_cause_2026-10-20/step54_author_mini_calibration_ab/results.json` | 8.8 | `eef4cdc4399536b9e3e3a7531f5b0f5eaa5732bf47d1451d10b8882f4b53fc2b` |

## 未纳入说明

- 二进制模型权重和 decoder 训练产物不提交；仅在本机保留。
- Step 50/54 独立环境日志为尾部摘录；完整原始日志仍在隔离工程 reports。
- 本目录用于核查实验记录，不表示所有候选修复已通过正式验证。
