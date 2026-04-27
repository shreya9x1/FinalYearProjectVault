# Dataset Description – MV-GSB (FYP 3.0)

## Overview

This project uses the **LLVIP (Low-Light Visible-Infrared Paired) dataset** for training and evaluating the RGB-to-Infrared translation model.

LLVIP is a benchmark dataset designed for low-light conditions, providing **paired visible (RGB) and infrared (IR) images** captured in real-world environments.

---

## Dataset Source

* Official Website: https://bupt-ai-cz.github.io/LLVIP/

---

## Dataset Structure

The dataset must be organized in the following directory format:

```
LLVIP/
├── visible/
│   ├── train/
│   └── test/
│
└── infrared/
    ├── train/
    └── test/
```

### File Format

* Image type: `.jpg`
* Pairing: Each RGB image has a corresponding IR image with the same filename.

---

## Dataset Statistics

* Training pairs: ~12,000+
* Testing pairs: ~3,400+
* Total pairs: ~15,000+
* Resolution used in this project: **512 × 512**

---

## How It Is Used in This Project

The dataset is used for **paired image-to-image translation** under low-light conditions:

* **Input**: Low-light RGB image
* **Target**: Corresponding IR image

### Processing Pipeline

1. RGB image is enhanced using a low-light enhancement module.
2. Enhanced RGB is converted to grayscale for bridge formulation.
3. A stochastic bridge is constructed between RGB and IR domains.
4. The model learns to reconstruct IR images from the bridge representation.

---

## Important Notes

* The dataset is **not included** in this repository due to its large size.
* Users must **download and place it manually** in the correct directory structure.
* Update dataset path in training command if required:

```bash
python train.py --dataset-root "path_to_LLVIP"
```

---

## Reproducibility

To reproduce results:

1. Download LLVIP dataset from the official link.
2. Arrange files exactly as described above.
3. Ensure correct dataset path is provided during training.

---

## Why LLVIP?

This dataset is specifically chosen because:

* It focuses on **extreme low-light scenarios**
* Provides **aligned RGB–IR pairs**
* Supports evaluation of both:

  * Image quality (PSNR, SSIM)
  * Task performance (object detection readiness)

---

## Summary

The LLVIP dataset enables this project to:

* Learn cross-modal mapping from RGB to IR
* Evaluate robustness in low-light conditions
* Demonstrate improvements in downstream perception tasks

---
