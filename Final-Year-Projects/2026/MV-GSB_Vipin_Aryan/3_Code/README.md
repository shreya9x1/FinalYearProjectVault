# FYP 3.0 - MV-GSB Pipeline

This folder contains a new implementation for RGB-to-IR translation using a MambaVision-guided Schrodinger Bridge (MV-GSB), independent from `FYP2.0`.

## Dataset
- LLVIP path (default): `C:\Project\FYP2.0\LLVIP`
- Expected structure:
  - `visible/train/*.jpg`, `visible/test/*.jpg`
  - `infrared/train/*.jpg`, `infrared/test/*.jpg`

## Core Design
- Generator backbone: lightweight MambaVision-style hybrid stub (local CNN + global token mixer).
- Pre-processor: RAWMamba-style low-light enhancement block.
- Oracle 1: DINOv2 semantic guidance stub (frozen feature encoder interface).
- Oracle 2: RF-DETR task guidance stub (frozen detector feature interface).
- 3-stage curriculum:
  - Stage 1 (1-50): structural warmup
  - Stage 2 (51-150): semantic anchoring
  - Stage 3 (151-300): task-aware refinement

## Quick Start
```bash
pip install -r requirements.txt
python train.py
```

Stage-specific runs:
```bash
# Stage 1 only (epochs 1-50)
python train.py --stage 1

# Resume from checkpoint and continue stage 2
python train.py --stage 2 --resume outputs/mvgsb_epoch_050.pt

# Full run with explicit dataset path override
python train.py --stage all --dataset-root "C:\Project\FYP2.0\LLVIP"
```

Optional semantic guidance modes:
```bash
# Use cached DINO features in cache/dinov2/<split>/*.pt (default behavior if cache exists)
python scripts/precompute_dino_features.py --split train

# Try real DINOv2 backbone (falls back to stub if model download/load fails)
python train.py --use-real-dino
```

## Files
- `config.py`: training and stage config
- `dataset_llvip.py`: paired LLVIP dataset loader
- `models/bridge.py`: guided Schrodinger bridge model
- `losses/objective.py`: composite roadmap losses
- `train.py`: staged training loop
- `scripts/precompute_dino_features.py`: optional oracle cache generation

## Training Logs
- Legacy metrics: `outputs/training_metrics.csv`.
- Stabilized diagnostics metrics: `outputs/training_metrics_v2.csv`.
- New columns: `epoch, stage, total, noise, edge, semantic, task, entropy, enh_mean, enh_std, pred_noise_std`.
- Every 5 epochs, training saves 2 visual progress comparisons with on-image metrics to:
  - `outputs/progress_viz/epoch_XXX/*.png`
  - Metrics shown directly on image: `PSNR`, `SSIM`, `Canny-IoU`, `LPIPS` (or `N/A` if LPIPS is unavailable).

## Stage-1 Progress Evaluation
- Script: `scripts/evaluate_stage1_progress.py`
- Purpose: evaluate checkpoints (e.g., `1,5,10,...,40`) with:
  - 5-column visuals per sample: Input RGB, Enhanced, `z_t` at `t=0.5`, Prediction, Ground Truth
  - metrics: PSNR, SSIM, Canny-IoU, LPIPS (optional)

Example:
```bash
python scripts/evaluate_stage1_progress.py --epochs "1,5,10,15,20,25,30,35,40" --split test --max-images 32 --num-workers 0 --use-lpips
```

Outputs:
- `outputs/eval_stage1/summary.csv`
- `outputs/eval_stage1/epoch_XXX/metrics_per_image.csv`
- `outputs/eval_stage1/epoch_XXX/visuals/*.png`

## Notes
This code is intentionally modular so you can replace stubs with official checkpoints:
- Swap `HybridMambaVisionBackbone` with official `nvlabs/mambavision` model.
- Replace oracle stubs with actual frozen DINOv2 and RF-DETR modules.
- Extend stage 3 dataloader to include M3FD once available locally.
