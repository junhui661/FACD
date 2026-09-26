# FACD

Official implementation of **FACD**, a two-stage framework for one-step image super-resolution based on flow-matching distillation.



## Method

FACD consists of two stages:

- **Stage 1 — Frequency-Aware Distillation.** Trains the student with frequency-aware consistency distillation and perceptual supervision.
- **Stage 2 — Projected Adversarial Refinement.** Refines the Stage 1 model with adversarial and perceptual objectives.

## Installation

```bash
python -m pip install -r requirements.txt
```



## Data

The default configurations use DIV2K with the following layout:

```text
dataset/
|-- DIV2K_train_HR/
|   `-- *.png
`-- DIV2K_valid_HR/
    `-- *.png
```

Custom locations can be specified through `dataset.train_path` and `dataset.val_path`.

## Pre-trained Models

Set the checkpoint fields in the configuration files or pass them through `--overrides`:

| Model | Placeholder |
| --- | --- |
| Teacher | `path/to/teacher_checkpoint.pth` |
| Stage 1 | `path/to/stage1_checkpoint.pth` |
| Stage 2 | `path/to/stage2_checkpoint.pth` |



## Training

### Stage 1: Frequency-Aware Distillation

```bash
torchrun --nproc_per_node=1 distill_fm_v5.py \
  --opt configs/dis_fm_DIV2K_facd_v5k.yml \
  --overrides \
  train.resume_from='' \
  train.pre_train_model='path/to/teacher_checkpoint.pth'
```

To resume training, set `train.resume_from` to an experiment directory containing `checkpoints-meta/checkpoint.pth`.

### Stage 2: Projected Adversarial Refinement

```bash
torchrun --nproc_per_node=1 distill_fm_v6.py \
  --opt configs/dis_fm_DIV2K_facd_v6b.yml \
  --overrides \
  train.pre_train_model='path/to/stage1_checkpoint.pth'
```

## Evaluation

```bash
python sample_fm.py \
  --opt configs/dis_fm_DIV2K_facd_v6b.yml \
  --overrides \
  sample.pre_train_model='path/to/stage2_checkpoint.pth' \
  dataset.val_path='./dataset/DIV2K_valid_HR' \
  sample.num_sample=100 \
  sample.psnr_batch_size=1 \
  sample.use_one_step=True \
  sample.one_step_t=0.001
```

## Repository Structure

```text
FACD/
|-- basicsr/              # Data and image-processing utilities
|-- configs/              # Stage 1 and Stage 2 configurations
|-- fm/                   # Flow matching, sampling, and FACD losses
|-- models/               # Network, discriminator, and EMA modules
|-- distill_fm_v5.py      # Stage 1 training
|-- distill_fm_v6.py      # Stage 2 training
`-- sample_fm.py          # Evaluation
```

## Citation

```bibtex
TODO: To be released.
```

## Acknowledgements

This repository builds upon OFTSR and BasicSR. We thank the authors of the respective projects for making their code available.

## License

This project is released under the [Apache License 2.0](LICENSE).
