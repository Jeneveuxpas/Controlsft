# Controlsft

> [!IMPORTANT]
> 本仓库的 `main` 分支基于官方 LTX-2，目前包含三条研究路径：Part16 reference 控制全量微调、
> Clean RGB SRA 辅助监督，以及“冻结的 Part16 condition teacher → 无 control student”中间表示蒸馏。
> SRA 实验比较 projector 深度、对齐层和 loss；蒸馏实验支持 same、独立 high/low、SRA interval
> 和 dual timestep 四种噪声配对。代码同时保留 `Part16 + Depth` 两个有序 reference 的能力。
>
> 官方基线：[`Lightricks/LTX-2@9377758`](https://github.com/Lightricks/LTX-2/tree/9377758131b1ffde4b7f766804590a6617bf2ab9)

## Part16 condition teacher / SRA 流程

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
- 没有加入 foreground、Part16/Depth 重建或 XYZ loss。

新增的 teacher/student 蒸馏可以与 Clean RGB SRA 联合启用：

```text
                               same x0, same Gaussian ε, same text
                                              │
                ┌─────────────────────────────┴─────────────────────────────┐
                │                                                           │
student: no reference, x_τ ── full LTX forward ── block l ── optional MLP   │
                │                                           │               │
                └─ official flow loss                       │ L_repr         │
                                                            │               │
teacher: [clean Part16 | x_min] ── frozen LTX to block k ───┘ detached target
```

teacher 输出会先切掉 Part16 reference 前缀，只使用 target-video hidden 与 student target token 对齐。

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

当前 A-F 联合消融中的 Clean `x_0` SRA 参数：

| 参数 | 当前值 | 说明 |
| --- | ---: | --- |
| `clean_rgb_sra_loss_weight` | `0.4` | warmup 完成后的权重 |
| `clean_rgb_sra_hidden_layer` | `8` | 第 8 个 transformer block，**1-based** |
| `clean_rgb_sra_hidden_dim` | `1024` | MLP hidden width |
| `clean_rgb_sra_num_layers` | `5` | Linear 总层数 |
| `clean_rgb_sra_warmup_steps` | `100` | SRA loss weight 线性 warmup |
| `clean_rgb_sra_loss_type` | `cosine` | 对齐 detached clean RGB latent `x_0` |
| `clean_rgb_sra_beta` | `0.05` | 仅 SmoothL1 模式使用 |
| `clean_rgb_sra_learning_rate` | `2e-5` | SRA head 独立 LR |

SRA head 会包含在全量 checkpoint 中，也会单独导出为：

```text
<output_dir>/checkpoints/clean_rgb_sra_head_step_XXXXX.pt
```

W&B 新增 `train/clean_rgb_sra_raw`、`train/clean_rgb_sra_loss` 和
`train/clean_rgb_sra_weight`。

## Condition Teacher → Unconditioned Student 蒸馏

### 训练语义

- `model.load_checkpoint` 用 condition-teacher checkpoint 初始化 student；A-F 中两者初始权重相同。
- student 的 `video.conditions: []`，训练和推理均不依赖 Part16 token。
- frozen teacher 使用 `teacher_conditions` 读取 Part16 reference；reference 保持 clean、`timestep=0`，
  不进入生成 loss，也会在 hidden alignment 前被切掉。
- student 完整运行 48 个 block 并计算官方 flow-matching loss；teacher 只运行到
  `teacher_hidden_layer`，不执行后续 block 和 output head。
- 两路始终共享同一个 clean target `x0` 和同一份 Gaussian noise `ε`；变化的是每个 token 使用的
  timestep，而不是重新采样另一份噪声。
- 总目标为 `L_total = L_flow + λ_repr(step) L_repr + λ_sra(step) L_clean_rgb_sra`；最后一项在
  `clean_rgb_sra_loss_weight > 0` 时启用，A-F 当前默认开启。teacher 全程
  `eval + inference_mode + detach`，只有 student、
  representation projector 和可选 Clean RGB SRA head 更新。

### Noise mode

| `noise_mode` | Student target token | Teacher target token | 含义 |
| --- | --- | --- | --- |
| `same` | 全部使用 `t` | 全部使用 `t` | 只比较 condition/层/projector 的影响 |
| `independent_high_low` | 全部使用 `max(t,s)` | 全部使用 `min(t,s)` | 独立采样后排序，每个样本明确一高一低 |
| `sra` | 全部使用 `t` | 全部使用 `clamp(t-U(0,x),0,1)` | SRA 的区间式 cleaner teacher |
| `dual_timestep` | 每个 token 以概率 `p` 使用独立采样的 `s`，否则使用 `t` | 全部使用 `min(t,s)` | 保持 student token 的 timestep 边际分布 |

`sra_timestep_max_gap` 是上式的 `x`，默认 `0.2`。`dual_timestep_second_probability` 只在
`dual_timestep` 中生效；视频配置使用 `p=0.1`，这是逐 token Bernoulli 的期望比例，
不保证每个样本恰好 10%。LTX video AdaLN 使用逐 token `t/s`；LTX-2.3 student prompt
AdaLN 使用 sample-level base `t`，teacher prompt AdaLN 使用 `min(t,s)`。

### Projector 与表示 loss

student 可直接对齐，或使用逐 token 两层 MLP：

```text
Linear(D, 1024) → SiLU → Linear(1024, D)
```

支持三种 `loss_type`，都先沿 hidden channel 归约，再用 `video_loss_mask` 对有效 target token 求均值：

| `loss_type` | 每 token 定义 | 建议用途 |
| --- | --- | --- |
| `cosine` | `1 - cosine_similarity(student, teacher)` | 跨层或经过 projector 的方向对齐 |
| `l1` | `mean(abs(student - teacher))` | 同层、同 feature basis 的直接回归 |
| `l2` | `mean((student - teacher)^2)` | 同层 MSE；不是欧氏范数 |

L1/L2 与 cosine 的数值尺度不同，不能直接把相同 `loss_weight` 理解成相同监督强度。短跑时应同时观察
`train/denoising_loss`、`train/representation_distillation_loss` 和 student/teacher feature norm。

### A-F 实验矩阵

| 实验 | Student → Teacher 层 | Noise | Projector | Loss | 主要问题 | 配置 |
| --- | --- | --- | --- | --- | --- | --- |
| A | 24 → 24 | same | none | L1 | condition 信息能否直接蒸馏到同层 | [配置](packages/ltx-trainer/configs/ablations/part16_stage2_distill_a_l24_direct_same.yaml) |
| B | 24 → 24 | same | 2-layer MLP | L1 | projector 是否提高可对齐性 | [配置](packages/ltx-trainer/configs/ablations/part16_stage2_distill_b_l24_mlp_same.yaml) |
| C | 16 → 34 | same | 2-layer MLP | cosine | 浅层 student 是否能学习深层语义 | [配置](packages/ltx-trainer/configs/ablations/part16_stage2_distill_c_l16_l34_mlp_same.yaml) |
| D | 16 → 34 | teacher low；student high | 2-layer MLP | cosine | 独立一高一低是否增强监督 | [配置](packages/ltx-trainer/configs/ablations/part16_stage2_distill_d_l16_l34_mlp_teacher_cleaner.yaml) |
| E | 16 → 34 | teacher `min(t,s)`；student token-wise `t/s` | 2-layer MLP | cosine | dual timestep 是否进一步有效 | [配置](packages/ltx-trainer/configs/ablations/part16_stage2_distill_e_l16_l34_mlp_self_flow.yaml) |
| F | 16 → 34 | teacher `clamp(t-U(0,0.2))`；student `t` | 2-layer MLP | cosine | 小间隔 SRA 是否优于独立大间隔 | [配置](packages/ltx-trainer/configs/ablations/part16_stage2_distill_f_l16_l34_mlp_sra.yaml) |

A↔B、C↔D、D↔E 是单核心变量对比；D↔F 用于比较独立大 gap 与 SRA 限定小 gap。
A-F 共同叠加第 8 层 Clean `x_0` SRA 辅助 loss（cosine、权重 0.4、100-step warmup），因此这组配置
比较的是 representation distillation 的主变量，而不是是否启用 Clean RGB SRA。
B→C 同时改变 layer pair 和 loss，因此不能单独归因于层选择；
如果要做严格层消融，需要补一组 `B-cosine` 或 `C-l1`。A/B 当前 `loss_weight=0.8` 是待短跑校准值，
L1 无界且可能受 feature norm 影响。

### 与 SRA / Self-Flow 原文的边界

这里实现的是 Self-Flow-inspired dual-timestep representation distillation，不是论文的完整复现：

- 原文使用 EMA self-teacher；本仓库使用训练前已得到的固定 Part16 condition teacher。
- 原文 teacher/student 条件相同；这里 teacher 有 Part16 reference，student 没有。
- `dual_timestep` 中 student 保留未排序的 `t/s` 逐 token 混合，teacher 才使用 `min(t,s)`；
  mask 与数值大小无关，因此 student token 仍保持原始 timestep 边际分布。
- A-F 默认 `high_noise_probability: 0.0`，未启用论文 appendix 的 5% `[0.95, 1.0]` 高噪声覆盖。
  若消融该策略，必须对所有比较组统一设置 `high_noise_probability: 0.05`，并在严格复现其 scheduler 时
  设置 `uniform_prob: 0.0`。

参考：[SRA 官方实现](https://github.com/vvvvvjdy/SRA)；
[`Self-Supervised Flow Matching for Scalable Multi-Modal Synthesis`](https://arxiv.org/abs/2603.06507)。

### Checkpoint、FSDP 与恢复训练

- distillation projector 是 student 的正式子模块，会进入 optimizer、FSDP 和 full checkpoint；推理加载器
  会过滤 `representation_distillation_head.*`。
- teacher 在每个 rank 上完整复制为 BF16，不进入 FSDP、optimizer 或 student checkpoint。显存预算必须包含
  一整套 teacher；多 GPU 节点启动时也要考虑每个进程并发读取完整 checkpoint 的 CPU RAM/IO。
- A-F 显式固定 `teacher_checkpoint`。恢复 student 时可以把 `model.load_checkpoint` 改为 student checkpoint，
  但不能让 teacher 路径跟着改变。
- A-F 当前 `checkpoints.no_resume: true`，每次启动都会从加载的模型权重重新计 step 0，不恢复 optimizer。
  需要断点恢复时应改为 `false`；training-state fingerprint 会校验 projector、层、loss、noise mode 和
  teacher checkpoint，发生变化时拒绝恢复旧 optimizer/global step。
- 动态 block hook 不支持 `torch.compile`；蒸馏模式检测到 Dynamo 会在启动阶段报错。请使用普通 FSDP
  配置，不使用 `fsdp_compile.yaml`。
- 当前 optimizer/head 分组按仓库标准 FSDP 配置设计，要求 `fsdp_use_orig_params: true`；不要在自定义
  Accelerate 配置中关闭它。
- 配置中的 `/workspace/...` 都是训练环境模板路径，提交任务前必须替换 checkpoint、manifest、数据根目录
  和输出目录。

新增 W&B 指标包括：

```text
train/representation_distillation_raw
train/representation_distillation_loss
train/representation_distillation_weight
train/representation_student_norm
train/representation_teacher_norm
train/representation_teacher_sigma
train/representation_second_timestep_fraction  # 仅 dual_timestep
train/distillation_projector_learning_rate      # 使用 MLP 独立 LR 时
```

## 当前消融实验

### Stage-1：MLP 深度

[configs/ablations](packages/ltx-trainer/configs/ablations/) 包含四组对照：

| 实验 | SRA 层 | MLP 层数 |
| --- | ---: | ---: |
| Baseline | — | — |
| SRA MLP-3 | 16 | 3 |
| SRA MLP-5 | 16 | 5 |
| SRA MLP-8 | 16 | 8 |

四组配置都是 1000 optimizer steps、effective global batch size 128、constant LR、
`shifted_logit_normal` timestep sampling，并使用 4 节点、32 GPU FULL_SHARD FSDP。

### Stage-2：对齐层、loss 与 Pure SFT

Stage-2 固定 SRA projector 为 5 层、hidden dim 为 1024，实验如下：

| 实验 | Part16 control | SRA 对齐层（1-based） | SRA loss | SRA weight | 配置 |
| --- | --- | ---: | --- | ---: | --- |
| Layer 8 | 有 | 8 | Smooth L1 | 0.1 | [配置](packages/ltx-trainer/configs/ablations/part16_stage2_sra_mlp5_layer8.yaml) |
| Layer 24 | 有 | 24 | Smooth L1 | 0.1 | [配置](packages/ltx-trainer/configs/ablations/part16_stage2_sra_mlp5_layer24.yaml) |
| Layer 16 Cosine | 有 | 16 | `1 - cosine_similarity` | 0.1 | [配置](packages/ltx-trainer/configs/ablations/part16_stage2_sra_mlp5_layer16_cosine.yaml) |
| Pure SFT | 无 | — | 无 | 0.0 | [配置](packages/ltx-trainer/configs/ablations/part16_stage2_pure_sft.yaml) |
| No-control SFT + SRA | 无 | 16 | Smooth L1 | 0.1 | [配置](packages/ltx-trainer/configs/ablations/part16_stage2_no_control_sra_mlp5_layer16_l1.yaml) |

五份配置统一保持：

- 同一个 step-40000 FP32 初始 transformer checkpoint；
- transformer LR `4e-6`、AdamW、constant scheduler；
- 1000 optimizer steps、effective global batch size 128；
- BF16 FSDP FULL_SHARD，FP32 optimizer state；
- 每 200 step 保存，最多保留 4 个 checkpoint；
- 同一训练 manifest、seed 和 timestep sampler。

前三组使用 Part16 reference control；Pure SFT 的 `video.conditions: []` 且
`clean_rgb_sra_loss_weight: 0.0`，因此不会读取 `reference_latents`，也不会创建
SRA head。No-control SFT + SRA 同样设置 `video.conditions: []`，不会读取或向
transformer 输入 `reference_latents`，但会在第 16 层使用 5 层 SRA head，以主视频自身的
clean x0 latent 作为 Smooth L1 监督目标。Cosine 组当前先保留 weight 0.1 做短跑尺度检查；应在 100-step warmup 完成后观察
`clean_rgb_sra_loss / denoising_loss`，再决定是否降低权重。

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

[infer_part16_control.py](packages/ltx-trainer/scripts/infer_part16_control.py) 使用与 condition-teacher 训练一致的
`[Part16 clean tokens | noisy RGB target tokens]` 布局。SRA 和 distillation projector 都只用于训练辅助 loss，
推理加载时会自动忽略。分辨率、帧数和 reference scale factor 应与训练保持一致；
`--include-control` 可保存控制与生成结果的左右对比视频。

注意：A-F 的 distilled student 训练时没有 reference token，标准评估应走无 control 的 T2V/student 推理，
不能把 Part16 control 脚本的结果当作 student 蒸馏效果。这个脚本主要用于 condition teacher 和带 Part16
控制的 SRA checkpoint。

如果测试集已经包含 `signal_latent` 和 `te`，可以跳过原始视频、VAE encoder 和
Gemma，直接按 manifest 批量生成。目标噪声尺寸由 signal latent 推导，不读取
`video_latent`：

```bash
cd packages/ltx-trainer
uv run python scripts/infer_part16_precomputed.py \
  --base-checkpoint /path/to/ltx-2.3-22b-dev.safetensors \
  --trained-checkpoint /path/to/model_weights_step_01000.safetensors \
  --manifest-path /path/to/test.jsonl \
  --output-dir outputs/test_precomputed \
  --start-index 0 \
  --num-samples 16
```

默认读取 manifest 的 `reference_latents` 和 `conditions` 字段，对应
`signal_latent` 和 `te`。默认 `guidance_scale=1`；如需 CFG，使用
`--guidance-scale 4 --negative-te /path/to/negative_te.pt`。批次包含不同 FPS 时，
建议按 FPS 分开运行并传入 `--frame-rate`。

## 主要修改文件

| 文件 | 修改 |
| --- | --- |
| [process_dataset.py](packages/ltx-trainer/scripts/process_dataset.py) | 预处理第一/第二 reference 视频 |
| [flexible.py](packages/ltx-trainer/src/ltx_trainer/training_strategies/flexible.py) | reference/SRA 处理、paired 输入、共享噪声与三种 timestep 模式 |
| [sra.py](packages/ltx-trainer/src/ltx_trainer/sra.py) | 可配置深度的 residual MLP projector |
| [distillation.py](packages/ltx-trainer/src/ltx_trainer/distillation.py) | 逐 token distillation MLP 与 cosine/L1/L2 loss |
| [model.py](packages/ltx-core/src/ltx_core/model/transformer/model.py) | teacher 中间层 extraction 与 checkpoint 边界外 post-block callback |
| [timestep_samplers.py](packages/ltx-trainer/src/ltx_trainer/timestep_samplers.py) | 可选高噪声区间采样覆盖 |
| [trainer.py](packages/ltx-trainer/src/ltx_trainer/trainer.py) | frozen teacher、联合辅助 loss、独立 LR、FSDP 与 checkpoint |
| [validation_runner.py](packages/ltx-trainer/src/ltx_trainer/validation_runner.py) | validation reference 布局与训练对齐 |
| [infer_part16_control.py](packages/ltx-trainer/scripts/infer_part16_control.py) | 单 Part16 全量 checkpoint 推理 |
| [infer_part16_precomputed.py](packages/ltx-trainer/scripts/infer_part16_precomputed.py) | 从 signal latent + TE 批量推理 |

查看相对官方基线的全部变更：

```bash
git diff 9377758..main -- packages/ltx-trainer
git log --oneline 9377758..main
```

相关回归测试覆盖参考条件顺序、预处理、SRA projector/loss、paired noise/timestep、distillation loss、
高噪声 sampler、FSDP 保存与 validation 布局。当前尚缺 trainer 单步、真实多 rank FSDP backward、
checkpoint resume 和 teacher early-stop hidden 等价性的蒸馏集成测试。
