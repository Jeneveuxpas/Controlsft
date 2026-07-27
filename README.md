# Controlsft

> [!IMPORTANT]
> 本仓库的 `main` 分支是基于官方 LTX-2 的 Controlsft 研究分支，当前只增加：
> **两个有序视频控制条件（Part16 + Depth）**和 **Clean RGB SRA 辅助监督**。
> 官方基线为 [`Lightricks/LTX-2@9377758`](https://github.com/Lightricks/LTX-2/tree/9377758131b1ffde4b7f766804590a6617bf2ab9)。

## Controlsft：相比官方代码修改了什么

### 1. 目标与明确边界

当前实现研究的是两个控制视频共同约束 RGB 视频生成：

```text
Part16 control ─┐
                ├─ VAE tokens ── [Part16 | Depth | noisy RGB target] ── LTX-2
Depth control ──┘                                      │
                                                      ├─ official flow-matching loss
clean RGB target ── VAE x0 ───────────────────────────└─ Clean RGB SRA loss
```

- Part16 和 Depth 只作为 clean reference token：`timestep=0`，不加噪，不计算生成 loss。
- RGB target 使用官方 flow-matching 训练目标。
- Clean RGB SRA 从指定 transformer 中间层的 **RGB target token** 预测 detached clean RGB VAE latent `x0`。
- 总 loss 为 `L_total = L_official_flow + λ(step) * L_clean_rgb_sra`。
- 没有加入 foreground loss、Part16/Depth 重建 loss、XYZ loss、蒸馏 loss或其他辅助 loss。
- `packages/ltx-core`、`packages/ltx-pipelines`、官方 transformer 主体和 VAE 均未修改。

### 2. 两个控制条件的数据与 token 顺序

每条数据必须是一一对应的四列：

| Dataset column | 含义 | 预处理输出 |
| --- | --- | --- |
| `video` | RGB target | `latents/` |
| `reference_video` | 第一个控制，当前约定为 Part16 | `reference_latents/` |
| `reference_video_1` | 第二个控制，当前约定为 Depth | `reference_latents_1/` |
| `caption` | 文本条件 | `conditions/` |

最终训练和验证的 token 布局都固定为：

```text
[reference_latents (Part16) | reference_latents_1 (Depth) | RGB target]
```

官方 flexible strategy 每次都会把 reference prepend 到序列前面，因此按配置正序应用两个
reference 会意外得到反序。本分支按配置的逆序执行 prepend，使最终 token 顺序仍与 YAML
声明顺序一致。验证器也显式重排为同一布局，避免出现“训练是一个顺序、validation 是另一个顺序”。

数据代码只保证文件/latent 的对应关系，不会判断视频内容是否时间对齐。RGB、Part16、Depth 必须在进入
VAE 前具有相同 clip 起止时间、帧数和目标分辨率；不要用旧控制视频缩放来替代同一 mesh 的原分辨率渲染。

### 3. Clean RGB SRA

SRA head 位于 [`packages/ltx-trainer/src/ltx_trainer/sra.py`](packages/ltx-trainer/src/ltx_trainer/sra.py)，
结构为逐 token 的：

```text
LayerNorm → Linear → GELU → 3 × residual linear block → LayerNorm → Linear
```

训练时在 transformer block `clean_rgb_sra_hidden_layer` 注册临时 forward hook，只截取 target 段 hidden：

1. SRA head 输出与 patchified clean RGB VAE latent 维度一致。
2. target `x0` 在计算 SRA loss 前 `detach()`，梯度只回到 SRA head 和 transformer/LoRA。
3. 使用 SmoothL1 loss，并沿用 RGB target 的有效 loss mask。
4. `clean_rgb_sra_loss_weight` 在前 `clean_rgb_sra_warmup_steps` 内线性 warm up。
5. SRA head 可使用独立学习率，并作为单独的 `.pt` 文件保存/加载。

当前示例配置：

| 参数 | 示例值 | 作用 |
| --- | ---: | --- |
| `clean_rgb_sra_loss_weight` | `0.05` | warmup 完成后的 SRA loss 权重 |
| `clean_rgb_sra_hidden_layer` | `8` | 捕获 block index 8（从 0 开始，即第 9 个 block）的输出 |
| `clean_rgb_sra_hidden_dim` | `1024` | SRA projector hidden width |
| `clean_rgb_sra_warmup_steps` | `100` | loss 权重线性 warmup |
| `clean_rgb_sra_beta` | `0.05` | SmoothL1 beta |
| `clean_rgb_sra_learning_rate` | `null` | `null` 表示跟随主 optimizer LR |
| `clean_rgb_sra_checkpoint` | `null` | 可选的 SRA head warm-start checkpoint |

SRA checkpoint 保存到：

```text
<output_dir>/checkpoints/clean_rgb_sra_head_step_XXXXX.pt
```

W&B 新增指标：

- `train/clean_rgb_sra_raw`
- `train/clean_rgb_sra_loss`
- `train/clean_rgb_sra_weight`

### 4. 修改文件索引

| 文件 | 相比官方代码的修改 |
| --- | --- |
| [`process_dataset.py`](packages/ltx-trainer/scripts/process_dataset.py) | 识别 `reference_video_1`，用同一个 Video VAE 独立生成 `reference_latents_1/`，并检查第二控制不能脱离第一控制单独存在。 |
| [`flexible.py`](packages/ltx-trainer/src/ltx_trainer/training_strategies/flexible.py) | 保持多 reference 的 YAML 声明顺序；保留 clean RGB target latent 和 target 起始 index；实现 Clean RGB SRA loss。 |
| [`base_strategy.py`](packages/ltx-trainer/src/ltx_trainer/training_strategies/base_strategy.py) | 在 `ModelInputs` 增加 `video_clean_latents` 和 `video_target_start_index`。 |
| [`sra.py`](packages/ltx-trainer/src/ltx_trainer/sra.py) | 新增 Clean RGB SRA projector。 |
| [`trainer.py`](packages/ltx-trainer/src/ltx_trainer/trainer.py) | 创建/加载 SRA head；捕获中间层；叠加 SRA loss；支持独立 LR、checkpoint 和 W&B metrics。 |
| [`validation_runner.py`](packages/ltx-trainer/src/ltx_trainer/validation_runner.py) | 将官方 append 后的 reference 重排到 target 前面，使 validation 与 training 同为 `[Part16 | Depth | target]`。 |
| [`v2v_two_control_ic_lora.yaml`](packages/ltx-trainer/configs/v2v_two_control_ic_lora.yaml) | 新增两个有序 reference 的完整示例和 Clean RGB SRA 参数。 |
| [`tests/`](packages/ltx-trainer/tests/) | 增加双控制预处理、reference 顺序、配置、SRA detach/warmup/mask 和 validation 布局测试。 |

### 5. 与官方保持一致的部分

- 官方 LTX-2 checkpoint 加载、Gemma text embedding、Video VAE 和 RoPE/position 生成逻辑。
- 官方 LoRA target modules、PEFT 包装、optimizer、gradient accumulation 和 checkpoint 主流程。
- 官方 RGB flow-matching target、timestep sampler 和主 loss。
- reference token 仍使用官方 IC-LoRA 的 clean-token conditioning 机制。

示例配置的 `model.load_checkpoint` 当前为 `null`，因此默认是官方 base model 加新初始化的 LoRA，
**不会自动加载某个官方 IC-LoRA adapter**。如果实验要求从官方 IC-LoRA 权重初始化，需要在配置中
显式填写兼容 checkpoint，并在实验记录中注明。

### 6. 配置、启动与对照

先修改示例中的模型路径、数据路径、validation 视频、分辨率、steps 和 W&B：

```bash
cd packages/ltx-trainer
uv run accelerate launch scripts/train.py configs/v2v_two_control_ic_lora.yaml
```

示例 YAML 是代码接口说明，不包含集群私有路径或实验数据。分辨率并未硬编码在训练实现中：
训练分辨率由预计算 latent 决定，validation 分辨率由 `validation.video_dims` 决定。

查看本分支相对官方基线的全部代码差异：

```bash
git diff 9377758..main -- packages/ltx-trainer
git log --oneline 9377758..main
```

当前相关回归测试共 7 个，最后一次执行结果为 `7 passed`。
