# Step70：检测/分割联合架构与语言融合审计

- 当前状态：`READY_FOR_REAL_JOINT_FORWARD`
- 已核对本地检测、分割、联合配置，检查 Head 任务开关和权重键。
- 不能把单独检测/分割 checkpoint 拼成联合训练的证据。
- 共享 BEV 的正确插入点为 `PerceptionTransformer.get_bev_features` 的输出，
  该输出先传给检测 Decoder，同时为分割 Head 使用。
- Step70 的前向实测（如执行 `--live`）只验证双 Head 接口和开关回退，
  不代表真实联合反向传播或自然语言语义理解。
- 真实检测+分割+注意力联合训练与物理坐标协议，需下一步专项验证。

- 作者联合 checkpoint 参考：https://drive.google.com/file/d/1a42LRJhI9FoIeOORTwEf0N2Vtu98wP9O/view?usp=share_link
- 审计 JSON：`outputs/stage2_step70/step70_audit.json`
- 可选实测 JSON：`outputs/stage2_step70/step70_joint_live.json`
