# R1 BitBot 仿真交付 · 2026-11-09

本队四台 R1 使用 FreiBots 行走和 PAiD R1 单次踢球。本次发布对应两场 35 秒的 MuJoCo 共享世界测试。BitBot 原生导航输出 `cmd_vel`，仿真平衡反馈负责停步接管与踢后恢复。

## 模型

| 技能 | 训练 checkpoint | ONNX 接口 | 文件 |
|---|---|---|---|
| 踢球 | `continued_control_109995` 的 `model_109995.pt` | 143 维观测，显式 LSTM 状态，24 动作，50 Hz | [policy.onnx](kick/policy.onnx) · [配套元数据](kick/metadata.json) |
| 行走 | FreiBots `model_10000.pt` | 415 维观测，24 动作，50 Hz | [上游 ONNX](https://github.com/pfrangl/freibots_rl_lab/blob/1f700a97656577865524b58e6d7df8d4aae5bfaa/data/checkpoints/r1_velocity/2026-10-22_21-21-33/exported/policy.onnx)（需原仓库权限） |

踢球包发布两场测试中实际加载的 ONNX 字节。用相同训练 checkpoint 重导后，200 个单步样本和 200 步递推的动作、隐藏状态及记忆状态与实测 ONNX 完全一致。重导文件的计算图完全一致，字节差异来自 ONNX 的 `producer_version` 字段。发布包保留实测原文件。`metadata.json` 保留冻结接口字节，其中 `checkpoint` 字段来自接口元数据的早期基线；实际踢球权重来源以 [发布清单](manifest.json) 为准。行走权重固定在上游原仓库的对应提交。 [ONNX 验证](kick/validation.json)

## 已验证进度

- 无守门员的 PAiD 核心区试验中，early stop 的进球率为静止球 **2,889/3,150（91.71%）**、滚动球 **2,770/3,150（87.94%）**。同一 R1 的固定 8 秒对照在 6,300 个核心区试验中逐例判分一致。
- 5 个无守门员的单机走停踢场景全部进球并严格恢复。这一组测试通过 ROS 速度接口接入行走，接近速度由脚本给出。
- 八机共享仿真跑完裁判暂停恢复与对手巡航两段场景。两段各有 1,750 个同步控制帧，八台均健康收尾，机器人间接触和中途状态重置均为零。本队两次射门均被守门员挡下，进球验收未通过。
- 两场均完成单次踢球与严格恢复，几何检测未发现二次空踢。踢后 READY 共 1,888 个控制样本，关节速度 RMS 无超过 0.1 rad/s 的样本。 [八机测试指标](team_validation.json)

## 运行范围

Early stop 由适配器根据球的有效出球和机器人的实测支撑触发。ONNX 提供网络推理，完整交接依赖适配器中的末端参考和反馈站立。八机测试使用仿真真值、脚本对手及现有 R1 物理资产，完整自主比赛与实机仍待验证。

在 `china4070` 的既有隔离环境运行，输出目录须为新目录。该环境已有用于测试的 BitBot Pixi/ROS 2、冻结 R1 模型和参考动作，以及原私有仓库导出的行走权重和配套运行契约。

```bash
cd ~/Downloads/freibots_walk_bitbots_20261109
bash exp/r1_bitbots_adapter/run_freibots_team.sh runs/team/repro_pause \
  --scenario pause_team --seconds 35
```

踢球工作基于 TeleHuman 的 PAiD 仓库，沿用其 CC BY-NC 4.0 许可说明。 [PAiD 上游](https://github.com/TeleHuman/HumanoidSoccer) · [论文](https://arxiv.org/abs/2602.05310)
