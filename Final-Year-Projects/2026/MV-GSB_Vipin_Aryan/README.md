# 🚀 MV-GSB: MambaVision-Guided Schrödinger Bridge

### Task-Aware RGB-to-IR Translation for Low-Light Surveillance

---

## 📌 Overview

Low-light environments severely degrade RGB image quality, leading to poor performance in downstream tasks like object detection. Infrared (IR) imaging is robust in such conditions, but thermal sensors are expensive and not always available.

<img width="1219" height="586" alt="image" src="https://github.com/user-attachments/assets/39cb1651-3652-440b-9d6e-3b68d0d6d7dc" />



This project introduces **MV-GSB**, a novel framework that learns to translate low-light RGB images into IR-like representations while **preserving structure, semantics, and detector-relevant features**.

---

## 🎯 Key Idea

Instead of direct RGB → IR mapping, this work models the problem as a **stochastic transport process** using a **Schrödinger Bridge**, combined with:

* Low-light enhancement
* Hybrid MambaVision backbone
* Semantic + task-aware guidance
* Stage-wise training strategy

---

## 🧠 Core Contributions

* ✅ **Schrödinger Bridge-based Translation**
  Learns a probabilistic mapping between RGB and IR domains instead of deterministic regression.

* ✅ **Hybrid MambaVision Backbone**
  Combines:

  * Local CNN features (fine details)
  * Global token mixing (long-range context)

* ✅ **Three-Stage Curriculum Training**

  * **Stage 1:** Structural Warmup
  * **Stage 2:** Semantic Anchoring (DINOv2)
  * **Stage 3:** Task-Aware Refinement (RF-DETR)

* ✅ **Multi-Objective Loss Design**

  * Noise loss (core bridge learning)
  * Edge loss (structure preservation)
  * Semantic loss (feature alignment)
  * Task loss (detection readiness)

* ✅ **Task-Oriented Evaluation**
  Focus on both:

  * Image quality (PSNR, SSIM, LPIPS)
  * Detection performance (mAP)

---

## 🏗️ Pipeline Overview

<img width="2125" height="875" alt="mv_gsb_architecture (1)" src="https://github.com/user-attachments/assets/73e96924-1c95-449a-8b33-7dfe03926156" />


---

## MV-GSB Framework

The MV-GSB (MambaVision-Guided Schrödinger Bridge) framework is a generative approach for translating RGB images to infrared (IR) using a diffusion-based bridging mechanism.

It first enhances input images using a RAWMamba pre-processing module, then constructs a stochastic bridge between RGB and IR domains using a Schrödinger Bridge formulation. Features are extracted through a hybrid backbone combining CNN, Mamba, and Transformer blocks for efficient and rich representation learning.

The model further uses task-aware guidance (semantic and detection oracles) to ensure that generated IR images are both visually accurate and useful for downstream tasks. Training is optimized using a combination of losses for noise prediction, edge preservation, semantic alignment, and task performance.

<img width="2912" height="1440" alt="arch_overview (1)" src="https://github.com/user-attachments/assets/07dbe2a0-5ca4-44f4-964d-7dcf9737c599" />


---

## 📊 Results

Final evaluation on LLVIP dataset:

| Metric  | Stage 1 | Stage 3 (Final) |
| ------- | ------: | --------------: |
| PSNR    |   23.57 |       **24.17** |
| SSIM    |   0.583 |       **0.608** |
| LPIPS ↓ |   0.561 |       **0.484** |
| mAP@0.5 |   0.015 |      **0.965+** |
| FPS     |     ~27 |             ~27 |

<img width="1657" height="847" alt="image" src="https://github.com/user-attachments/assets/f964af14-9e01-4218-ab9f-2a80cc2c9b8c" />

📌 Key Insight:

> Detection performance improves **~6×**, proving that visual quality alone is not enough — task-aware learning is critical.

---

## 📂 Project Structure

```id="zq8mcv"
MV-GSB/
├── 1_Report/              # Thesis document
├── 2_Presentation/        # Final presentation
├── 3_Code/                # Model + training pipeline
├── 4_Dataset/             # Dataset instructions
├── 5_Methodology/         # Mathematical formulation
├── 6_Results/             # Outputs and evaluation
└── README.md
```

---

## 📦 Dataset

* Dataset used: **LLVIP (Low-Light Visible-Infrared Paired)**
* Link: https://bupt-ai-cz.github.io/LLVIP/

Structure:

```id="s0c4zd"
visible/train/
visible/test/
infrared/train/
infrared/test/
```

---

## ⚙️ Installation

```bash id="zq1n7a"
pip install -r requirements.txt
```

---

## 🚀 Training

### Full Training

```bash id="jz7c8m"
python train.py
```

### Stage-wise Training

```bash id="0g0r7x"
python train.py --stage 1
python train.py --stage 2 --resume outputs/mvgsb_epoch_050.pt
python train.py --stage all
```

---

## 🧪 Evaluation

```bash id="qz2r4m"
python evaluate.py
```

Outputs:

* Quantitative metrics (CSV/JSON)
* Qualitative results (images)
* Stage-wise comparisons

---

## 🧩 Key Components

* `models/bridge.py` → Core Schrödinger Bridge logic
* `models/mv_generator.py` → Hybrid MambaVision backbone
* `models/enhancement.py` → Low-light enhancer
* `models/guidance.py` → Semantic + task oracles
* `losses/` → Multi-objective loss functions

---

## 💡 Why This Matters

Traditional models optimize only image similarity.

This project shows:

> ✔ A model can look visually correct but still fail in detection
> ✔ Task-aware supervision significantly improves real-world utility

---

## ⚠️ Limitations

* Limited to LLVIP dataset domain
* Extreme weather conditions not fully handled
* Requires paired RGB–IR data

---

## 🔮 Future Work

* Extend to multi-dataset generalization
* Real-time deployment optimization
* Integrate with detection pipelines (YOLO, DETR)
* Use larger MambaVision backbones

---

## 👨‍💻 Authors

* **Vipin Tomer**
* **Aryan Kumar**
  
**Under the guidance of:** [Dr. Shounak Chakraborty](https://github.com/shounakc6)

Department of CSE
IIITDM Kurnool

---

## ⭐ Final Note

This project is not just about image translation —
it is about **making machine perception work in the real world under extreme conditions**.
