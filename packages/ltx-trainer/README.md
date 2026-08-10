# LTX-2 Trainer

This package provides tools and scripts for training and fine-tuning
Lightricks' **LTX-2** audio-video generation model. It supports LoRA training, full
fine-tuning, and a flexible conditioning framework covering text-to-video, text-to-audio, image-to-video,
video extension, audio extension, video inpainting, audio inpainting, video outpainting, IC-LoRA for video, audio, and joint
audio-video references, audio-to-video, and video-to-audio. This fork also includes Clean RGB SRA and frozen
condition-teacher → unconditioned-student representation distillation for Part16 research.

---

## 📖 Documentation

All detailed guides and technical documentation are in the [docs](./docs/) directory:

- [⚡ Quick Start Guide](docs/quick-start.md)
- [🎬 Dataset Preparation](docs/dataset-preparation.md)
- [🛠️ Training Modes](docs/training-modes.md)
- [⚙️ Configuration Reference](docs/configuration-reference.md)
- [🚀 Training Guide](docs/training-guide.md)
- [🧪 Inference Guide](../ltx-pipelines/README.md)
- [🔧 Utility Scripts](docs/utility-scripts.md)
- [🧩 Custom Training Strategies](docs/custom-training-strategies.md)
- [📚 LTX-Core Documentation](../ltx-core/README.md)
- [🛡️ Troubleshooting Guide](docs/troubleshooting.md)

### 🤖 Agent-Assisted Training

Use the [`train-model`](../../.claude/skills/train-model/SKILL.md) repository skill for an end-to-end guided run:
it probes your data and hardware, chooses the matching training mode, prepares/preprocesses the dataset, launches
training, and monitors the job while using the docs above as the source of truth.

---

## 🔧 Requirements

- **LTX-2 Model Checkpoint** - Local `.safetensors` file
- **Gemma Text Encoder** - Local Gemma model directory (required for LTX-2)
- **Linux with CUDA** - CUDA 13+ recommended for optimal performance
- **Nvidia GPU with 80GB+ VRAM** - Recommended for the standard config. For GPUs with 32GB VRAM (e.g., RTX 5090),
  use the [low VRAM config](configs/t2v_lora_low_vram.yaml) which enables INT8 quantization and other
  memory optimizations

### Full-model Part16 inference

Use the trainer-side script for a full fine-tuning checkpoint conditioned on one Part16 video. The official
checkpoint provides model metadata and VAE components; the trained checkpoint replaces the transformer weights:

```bash
uv run python scripts/infer_part16_control.py \
  --base-checkpoint /path/to/ltx-2.3-22b-dev.safetensors \
  --trained-checkpoint /path/to/model_weights_step_01000.safetensors \
  --gemma-root /path/to/gemma \
  --control-video /path/to/part16.mp4 \
  --prompt "A person follows the Part16 control motion." \
  --width 768 --height 512 --num-frames 121 \
  --output-path outputs/part16_result.mp4
```

The trainer-side inference loaders ignore training-only Clean RGB SRA and representation-distillation heads. Match
output dimensions and reference scale factors to training; add `--include-control` for a side-by-side
control/generated video. Distilled students are trained without reference tokens and should normally be evaluated
as unconditioned-student/T2V models rather than through the Part16 control interface.

### Part16 research extensions

The implementation and experiment matrix are documented in the repository [README](../../README.md). The six
teacher/student configs are indexed in [configs/README.md](configs/README.md), and every student is initialized from
the condition-teacher checkpoint while the frozen teacher remains explicit and separate. The video-only full-tuning
path supports shared, independently sorted high/low, SRA interval, and unordered token-wise `t/s` pairing.

---

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help:

- **Share Your Work**: If you've trained interesting LoRAs or achieved cool results, please share them with the
  community.
- **Report Issues**: Found a bug or have a suggestion? Open an issue on GitHub.
- **Submit PRs**: Help improve the codebase with bug fixes or general improvements.
- **Feature Requests**: Have ideas for new features? Let us know through GitHub issues.

---

## 💬 Join the Community

Have questions, want to share your results, or need real-time help?

Join our [community Discord server](https://discord.gg/ltxplatform) to connect with other users and the development
team!

- Get troubleshooting help
- Share your training results and workflows
- Collaborate on new ideas and features
- Stay up to date with announcements and updates

We look forward to seeing you there!

---

Happy training! 🎉
