# Controlsft

> [!IMPORTANT]
> 本仓库的 `main` 分支基于官方 LTX-2，目前用于研究视频 reference 控制和
> Clean RGB SRA 辅助监督。当前 Stage-1 消融使用**单个 Part16 控制 + 全量微调**，
> 代码同时保留 `Part16 + Depth` 两个有序 reference 的能力。
>
> 官方基线：[`Lightricks/LTX-2@9377758`](https://github.com/Lightricks/LTX-2/tree/9377758131b1ffde4b7f766804590a6617bf2ab9)

## 当前训练流程

```text
Part16 control ── VAE tokens ── [Part16 clean tokens | noisy RGB target tokens] ── LTX-2.3
                                                                                 │
                                                  ├─ official flow-matching loss
clean RGB target ── VAE x0 ────────────────────────────────────────────────└─ Clean RGB SRA loss
```

- Part16 是 clean reference token：`timestep=0`，不加噪，不计算生成 loss。
- RGB target 使用官方 flow-matching 训练目标。
- SRA 从指定 transformer 中间层的 RGB target token 预测 detached clean RGB VAE latent `x0`。
- 总 loss 为 `L_total = L_official_flow + λ(step) * L_clean_rgb_sra`。
- 当前消融配置使用 `training_mode: full`，不是 LoRA。
- 没有加入 foreground、Part16/Depth 重建、XYZ、蒸馏或其他辅助 loss。

## 数据与 token 顺序

当前单 Part16 数据使用：

| Dataset column | 含义 | 预处理输出 |
| --- | --- | --- |
| `video` | RGB target | `latents/` |
| `reference_video` | Part16 control | `reference_latents/` |
| `caption` | 文本条件 | `conditions/` |

单控制 token 布局：

```text
[reference_latents (Part16) | RGB target]
```

如果增加 `reference_video_1`（当前约定为 Depth），它会被预处理到
`reference_latents_1/`，token 顺序为：

```text
[reference_latents (Part16) | reference_latents_1 (Depth) | RGB target]
```

训练和 validation 使用相同顺序。RGB、Part16 以及可选 Depth 必须在进入 VAE 前保持
clip 起止时间、帧数和目标分辨率对齐。

## Clean RGB SRA projector

[sra.py](packages/ltx-trainer/src/ltx_trainer/sra.py) 实现的 projector 是逐 token residual MLP：

```text
LayerNorm → Linear → GELU → (Residual MLP Block) × K → LayerNorm → Linear
```

`K = clean_rgb_sra_num_layers - 2`。每个 residual block 都有可学习 scale，初始值为
`1 / sqrt(K)`；输出层使用小方差权重和 zero bias 初始化。

当前 SRA 消融参数：

| 参数 | 当前值 | 说明 |
| --- | ---: | --- |
| `clean_rgb_sra_loss_weight` | `0.1` | warmup 完成后的权重 |
| `clean_rgb_sra_hidden_layer` | `16` | 第 16 个 transformer block，**1-based** |
| `clean_rgb_sra_hidden_dim` | `1024` | MLP hidden width |
| `clean_rgb_sra_num_layers` | `3 / 5 / 8` | Linear 总层数消融 |
| `clean_rgb_sra_warmup_steps` | `100` | SRA loss weight 线性 warmup |
| `clean_rgb_sra_beta` | `0.05` | SmoothL1 beta |
| `clean_rgb_sra_learning_rate` | `2e-5` | SRA head 独立 LR |

SRA head 会包含在全量 checkpoint 中，也会单独导出为：

```text
<output_dir>/checkpoints/clean_rgb_sra_head_step_XXXXX.pt
```

W&B 新增 `train/clean_rgb_sra_raw`、`train/clean_rgb_sra_loss` 和
`train/clean_rgb_sra_weight`。

## 当前消融实验

[configs/ablations](packages/ltx-trainer/configs/ablations/) 包含四组对照：

| 实验 | SRA 层 | MLP 层数 |
| --- | ---: | ---: |
| Baseline | — | — |
| SRA MLP-3 | 16 | 3 |
| SRA MLP-5 | 16 | 5 |
| SRA MLP-8 | 16 | 8 |

四组配置都是 1000 optimizer steps、effective global batch size 128、constant LR、
`shifted_logit_normal` timestep sampling，并使用 4 节点、32 GPU FULL_SHARD FSDP。

单机启动示例：

```bash
cd packages/ltx-trainer
uv run accelerate launch scripts/train.py configs/ablations/part16_stage1_baseline.yaml
```

4 节点启动时，在四台机器上使用同一 `RUN_ID` 执行：

```bash
scripts/launch_4node_fsdp.sh baseline_001 configs/ablations/part16_stage1_baseline.yaml
```

## 单 Part16 控制推理

全量训练 checkpoint 是 transformer 权重，不是完整 pipeline checkpoint。推理时使用官方
base checkpoint 提供模型 metadata、VAE 等组件，再覆盖训练后的 transformer 权重：

```bash
cd packages/ltx-trainer
uv run python scripts/infer_part16_control.py \
  --base-checkpoint /path/to/ltx-2.3-22b-dev.safetensors \
  --trained-checkpoint /path/to/model_weights_step_01000.safetensors \
  --gemma-root /path/to/gemma \
  --control-video /path/to/part16.mp4 \
  --prompt "A person follows the Part16 control motion." \
  --width 768 --height 512 --num-frames 121 \
  --output-path outputs/part16_result.mp4
```

[infer_part16_control.py](packages/ltx-trainer/scripts/infer_part16_control.py) 使用与训练一致的
`[Part16 clean tokens | noisy RGB target tokens]` 布局。SRA head 只用于训练辅助 loss，
推理时会自动忽略。分辨率、帧数和 reference scale factor 应与训练保持一致；
`--include-control` 可保存控制与生成结果的左右对比视频。

## 主要修改文件

| 文件 | 修改 |
| --- | --- |
| [process_dataset.py](packages/ltx-trainer/scripts/process_dataset.py) | 预处理第一/第二 reference 视频 |
| [flexible.py](packages/ltx-trainer/src/ltx_trainer/training_strategies/flexible.py) | reference 顺序、clean target latent 与 SRA loss |
| [sra.py](packages/ltx-trainer/src/ltx_trainer/sra.py) | 可配置深度的 residual MLP projector |
| [trainer.py](packages/ltx-trainer/src/ltx_trainer/trainer.py) | SRA hook/head、独立 LR、FSDP 训练与 checkpoint |
| [validation_runner.py](packages/ltx-trainer/src/ltx_trainer/validation_runner.py) | validation reference 布局与训练对齐 |
| [infer_part16_control.py](packages/ltx-trainer/scripts/infer_part16_control.py) | 单 Part16 全量 checkpoint 推理 |
| [launch_4node_fsdp.sh](packages/ltx-trainer/scripts/launch_4node_fsdp.sh) | 4 节点 FSDP 启动 |

查看相对官方基线的全部变更：

```bash
git diff 9377758..main -- packages/ltx-trainer
git log --oneline 9377758..main
```

相关回归测试覆盖参考条件顺序、预处理、SRA projector/loss、FSDP 保存与
validation 布局。
