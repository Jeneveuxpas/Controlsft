# Training Configs

Example training configurations for the LTX-2 trainer. Each file is a ready-to-run config for one training mode,
expressed through the unified **flexible** strategy (`name: "flexible"`). Pick the one closest to your use case and
adjust paths, dataset, and hyperparameters.

> 📖 For more information about using each training mode, see [Training Modes Guide](../docs/training-modes.md).

## Training Modes

| Mode                  | Video     | Audio     | Conditions          | Config |
|-----------------------|-----------|-----------|---------------------|--------|
| **T2V**               | Generated | Generated | —                   | [`t2v_lora.yaml`](./t2v_lora.yaml), [`t2v_lora_low_vram.yaml`](./t2v_lora_low_vram.yaml) (low VRAM) |
| **I2V**               | Generated | Generated | `first_frame`       | [`i2v_lora.yaml`](./i2v_lora.yaml) |
| **Video Extension**   | Generated | Generated | `prefix`/`suffix`   | [`video_extend_lora.yaml`](./video_extend_lora.yaml) (forward), [`video_suffix_lora.yaml`](./video_suffix_lora.yaml) (backward) |
| **V2V IC-LoRA**       | Generated | —         | `reference`         | [`v2v_ic_lora.yaml`](./v2v_ic_lora.yaml) |
| **Part16 V2V Full Baseline** | Generated | —  | `reference`         | [`v2v_part16_full_baseline.yaml`](./v2v_part16_full_baseline.yaml) |
| **Part16 Stage-1 SRA Ablation** | Generated | — | `reference` | [`baseline`](./ablations/part16_stage1_baseline.yaml), [`MLP-3`](./ablations/part16_stage1_sra_mlp3_layer16.yaml), [`MLP-5`](./ablations/part16_stage1_sra_mlp5_layer16.yaml), [`MLP-8`](./ablations/part16_stage1_sra_mlp8_layer16.yaml) |
| **Part16 Stage-2 Ablation** | Generated | — | `reference` / — | [`layer-8`](./ablations/part16_stage2_sra_mlp5_layer8.yaml), [`layer-24`](./ablations/part16_stage2_sra_mlp5_layer24.yaml), [`layer-16 cosine`](./ablations/part16_stage2_sra_mlp5_layer16_cosine.yaml), [`pure SFT`](./ablations/part16_stage2_pure_sft.yaml) |
| **Part16 Teacher/Student Distillation** | Generated | — | Teacher-only `reference` | [`A: 24→24 direct`](./ablations/part16_stage2_distill_a_l24_direct_same.yaml), [`B: 24→24 MLP`](./ablations/part16_stage2_distill_b_l24_mlp_same.yaml), [`C: 16→34`](./ablations/part16_stage2_distill_c_l16_l34_mlp_same.yaml), [`D: cleaner teacher`](./ablations/part16_stage2_distill_d_l16_l34_mlp_teacher_cleaner.yaml), [`E: Self-Flow`](./ablations/part16_stage2_distill_e_l16_l34_mlp_self_flow.yaml) |
| **A2V**               | Generated | Frozen    | —                   | [`a2v_lora.yaml`](./a2v_lora.yaml) |
| **V2A (Foley)**       | Frozen    | Generated | —                   | [`v2a_lora.yaml`](./v2a_lora.yaml) |
| **Video Inpainting**  | Generated | —         | `mask`              | [`video_inpainting_lora.yaml`](./video_inpainting_lora.yaml) |
| **Video Outpainting** | Generated | —         | `spatial_crop`      | [`video_outpainting_lora.yaml`](./video_outpainting_lora.yaml) |
| **T2A**               | —         | Generated | —                   | [`t2a_lora.yaml`](./t2a_lora.yaml) |
| **Audio Extension**   | —         | Generated | `prefix`/`suffix`   | [`audio_extend_lora.yaml`](./audio_extend_lora.yaml) (forward), [`audio_suffix_lora.yaml`](./audio_suffix_lora.yaml) (backward) |
| **Audio Inpainting**  | —         | Generated | `mask`              | [`audio_inpainting_lora.yaml`](./audio_inpainting_lora.yaml) |
| **A2A IC-LoRA**       | —         | Generated | `reference`         | [`a2a_ic_lora.yaml`](./a2a_ic_lora.yaml) |
| **AV2AV IC-LoRA**     | Generated | Generated | `reference` (both)  | [`av2av_ic_lora.yaml`](./av2av_ic_lora.yaml) |

The [`accelerate/`](./accelerate) directory holds the Accelerate launch configs (FSDP, DDP) for multi-GPU training.
The [`ablations/`](./ablations) directory holds matched experiment configs. The Part16 stage-1 set uses 1-based SRA
layer 16, 1000 optimizer steps, effective global batch size 128, and full FSDP optimizer checkpoints.
The stage-2 set fixes the projector at five linear layers and compares transformer layers 8 and 24, cosine
alignment at layer 16, and pure SFT without reference control or x0 alignment. All stage-2 configs keep the
optimizer, data, seed, global batch size, and checkpoint policy fixed.

The teacher/student A-E set initializes an unconditioned full-model student from the fixed Part16 condition teacher.
A and B use same-layer L1 alignment at block 24 and differ only by the student MLP projector. C-E use block 16→34
cosine alignment: C shares one timestep, D gives the teacher `min(t, s)`, and E additionally assigns timestep `s`
to an expected 10% of student video tokens. B→C changes both layer selection and loss type, so it is not a pure
layer-only comparison. All `/workspace/...` paths are environment-specific templates.
