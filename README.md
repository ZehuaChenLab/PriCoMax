# PriCoMax

PyTorch implementation of **A Prior Is Not Data: Priors as Generated and Consumed States for Joint Low-Light Restoration**.

PriCoMax corrects degraded illumination priors and couples them with image restoration through PRSR, HAPFA, and CRU. RC-PSC calibrates the illumination prior, while the load-bearing output head uses the corrected illumination in image formation. The model is registered in the code as `PriCoDiffMax`.

## Abstract

An illumination prior can fail a low-light restoration network twice: before it arrives, and after. With a fixed analytic Retinex operator, reflectance extracted from jointly noisy, blurred, and underexposed inputs carries a 13.2% mean per-image relative error and 14.3× the high-frequency energy of ground-truth-derived reflectance. The observation alone does not uniquely determine a clean prior state, and end-to-end supervision guarantees neither its calibration nor the use of corrected guidance. With inputs and weights fixed, replacing pre-correction priors by their corrected counterparts changes illumination by 55% but peak signal-to-noise ratio by only -0.0032 dB. We therefore generate the prior through illumination-level supervision and consume its corrected illumination through explicit relighting. Retinex-Consistent Prior State Correction applies bounded, reliability-gated updates against a target-derived reference, cutting absolute illumination-level error by 36% and re-deriving reflectance algebraically. Load-bearing output formation multiplies self-normalized content by the selected illumination, with regularized residual compensation; under the same fixed-weight protocol, a separately trained load-bearing model shows a +0.4853 dB difference. The complete model reaches 28.2358 dB peak signal-to-noise ratio on LOLBlur with 5.74M parameters. Under mixed-task training, a shared checkpoint also performs competitively on low-light enhancement alone across three benchmarks.

## Installation

Use Python 3.10 or newer and a compatible PyTorch/torchvision installation for your CUDA environment. Install the remaining dependencies:

```bash
python -m pip install numpy Pillow PyYAML tqdm pytorch-msssim lpips
```

Run the commands below from the repository root. Perceptual-loss backbones may download pretrained weights on first use.

## Data and Checkpoints

Obtain datasets from their original providers. Set `data.root` in the selected YAML configuration to your local dataset directory. For mixed-task training, update the roots under `data.datasets` instead.

For LOLBlur, use this directory structure; scene subdirectories are supported:

```text
data/LOLBlur/
├── train/
│   ├── low_blur_noise/
│   └── high_sharp_scaled/
└── test/
    ├── low_blur_noise/
    └── high_sharp_scaled/
```

The examples below use a trained checkpoint at `checkpoints/b9_best_model.pth`. Replace this path with your checkpoint. Dataset images and checkpoints must be supplied separately.

## Inference

```bash
python scripts/infer_pricodiff_lolblur.py --config configs/train_lolblur_pricomax_lbhead_s.yaml --checkpoint checkpoints/b9_best_model.pth --input data/LOLBlur/test/low_blur_noise --restored-root results/restored --device cuda:0
```

Restored images retain the input directory structure. Use a checkpoint matching the selected model configuration.

## Evaluation

After updating the dataset path in the configuration:

```bash
python scripts/validate.py --config configs/train_lolblur_pricomax_lbhead_s.yaml --checkpoint checkpoints/b9_best_model.pth --split test --output results/evaluation --device cuda:0
```

For comparisons with paper results, match the checkpoint, preprocessing, color space, and metric implementation used in the corresponding experiment.

## Training

The full LOLBlur configuration uses warm-start training. Before running it, supply the checkpoints specified by `train.init_checkpoint` and `train.pretrained_darkir`, and update their paths in the YAML file.

```bash
python scripts/train.py --config configs/train_lolblur_pricomax_lbhead_s.yaml --output experiments/lolblur_pricomax_lbhead_s --device cuda:0
```

Configurations for component ablations and mixed-task training are available in `configs/`. The recorded LOLBlur recipe uses the `test` split for validation and checkpoint selection; take this into account when interpreting evaluation results.

## Repository Structure

- `models/`: restoration networks and prior-processing modules.
- `datasets/`: paired-image loading and mixed-task data handling.
- `losses/`: training losses and evaluation metrics.
- `configs/`: training and ablation configurations.
- `scripts/`: training, inference, evaluation, and prior diagnostics.
- `utils/`: checkpoint, image, and training utilities.
