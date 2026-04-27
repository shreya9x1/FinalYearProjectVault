# MV-GSB: MambaVision-Guided Schrodinger Bridge for RGB-to-IR Translation

## 1. Novelty (Core Contributions)

### 1.1 Problem Setting
**Low-Light RGB-to-Infrared Translation with Guidance**

Given low-illumination RGB images where visible details are severely degraded, we learn a stochastic transport mapping to produce thermal IR-like output that preserves structure, semantics, and task-relevant features for object detection.

### 1.2 Key Novelties

#### **1. MambaVision-Guided Schrodinger Bridge (MV-GSB)**
- **Classical Diffusion Problem**: Standard diffusion models require hundreds of forward steps for inference.
- **Our Bridge Solution**: We use a Schrodinger Bridge framework that learns a minimal **one-step direct mapping** from RGB latent space to IR target space with principled stochastic interpolation.
- **Why This Matters**: Enables fast, stable, low-light-aware translation without iterative denoising.

$$\text{Bridge Time Path: } z_t = \alpha(t) \cdot x_{ir} + (1-\alpha(t)) \cdot x_{rgb} + \sigma(t) \cdot \epsilon$$

#### **2. Hybrid MambaVision Backbone**
- **Classical CNNs**: Limited receptive field; inefficient at capturing long-range dependencies.
- **Our Architecture**: Combines:
  - **Local Conv Blocks**: Fine-grained structural details (edges, textures).
  - **Token Mixer Layers**: Global depthwise operations for efficient long-range context.
  - **Result**: Lightweight, efficient, and effective for translating structure under extreme illumination degradation.

#### **3. Three-Stage Curriculum Guidance**
- **Problem**: Applying semantic + task guidance from epoch 1 causes training instability and gradient explosions in low-light domains.
- **Our Solution**: Progressive multi-stage curriculum:
  - **Stage 1 (Structural Warmup)**: Learn basic RGB↔IR mapping without guidance (epochs 1–41).
  - **Stage 2 (Semantic Anchoring)**: Activate frozen DINOv2 semantic guidance (epochs 42–150).
  - **Stage 3 (Task-Aware Refinement)**: Add RF-DETR task-level detection guidance (epochs 151–300).
- **Benefit**: Stable training with 6x improvement in detection accuracy (mAP@0.5: 0.015 → 0.097).

$$\text{Stage Scales: } s_{sem}(t) = \begin{cases} 0 & \text{Stage 1} \\ \lambda_{sem} & \text{Stages 2,3} \end{cases}, \quad s_{task}(t) = \begin{cases} 0 & \text{Stages 1,2} \\ \lambda_{task} & \text{Stage 3} \end{cases}$$

#### **4. Low-Light Enhancement Front-End**
- **Classical Preprocessing**: Fixed gamma correction; inadequate for LLVIP extreme darkness.
- **Our RAWMamba Enhancer**: 
  - Zero-reference illumination correction (no reference IR needed during enhancement).
  - Adaptive channel normalization based on raw signal statistics.
  - Prepares RGB for safe bridge mixing.

#### **5. Composite Multi-Objective Loss**
- **Single-Task Limitation**: MSE loss alone produces blurry, detail-less outputs.
- **Our Synergistic Objective**:
  - **Structural (Edge) Loss** ($L_{edge}$): Preserves boundaries on 3 LoG scales.
  - **Semantic Loss** ($L_{sem}$): Enforces DINOv2 feature alignment (content consistency).
  - **Task Loss** ($L_{task}$): Aligns RF-DETR features (detection readiness).
  - **Entropy Regularization** ($L_{entropy}$): Prevents mode collapse; ensures distribution coverage.

$$\mathcal{L}_{total} = w_n \mathcal{L}_{noise} + w_e \mathcal{L}_{edge} + s_{sem} \mathcal{L}_{sem} + s_{task} \mathcal{L}_{task} + w_k \mathcal{L}_{entropy}$$

---

## 2. Mathematical Formulations

### 2.1 Bridge State Construction

#### **Timeline Scheduling**
Let $t \in [0, 1]$ be the normalized bridge time parameter.

**Linear Alpha Schedule (Interpolation Weight)**:
$$\alpha(t) = 1 - t$$

**Parabolic Sigma Schedule (Noise Injection)**:
$$\sigma(t; \sigma_{max}) = \sigma_{max} \cdot 4 \cdot t \cdot (1-t)$$

where $\sigma_{max} = 0.25$ controls maximum noise amplitude. The parabolic form ensures:
- At $t=0$ and $t=1$: $\sigma(t) = 0$ (deterministic endpoints).
- At $t=0.5$: $\sigma(0.5) = \sigma_{max}$ (maximum uncertainty).

#### **Bridge State (Stochastic Interpolation)**
$$z_t = \alpha(t) \cdot x_{ir} + (1 - \alpha(t)) \cdot x_{rgb\_gray} + \sigma(t) \cdot \epsilon$$

where:
- $x_{ir} \in [0, 1]^{C \times H \times W}$ is the target IR image.
- $x_{rgb\_gray} = 0.2989 \cdot R + 0.5870 \cdot G + 0.1140 \cdot B$ (ITU grayscale from enhanced RGB).
- $\epsilon \sim \mathcal{N}(0, I)$ is standard Gaussian noise.
- $z_t$ is a **deterministic interpolation** of the two modalities with controlled stochastic perturbation.

**Clamping & Numerical Safety**:
$$t \gets \text{clamp}(t, t_{eps}, 1 - t_{eps}), \quad t_{eps} = 10^{-4}$$
$$\alpha \gets \text{clamp}(\alpha, \epsilon_{clamp}), \quad \epsilon_{clamp} = 10^{-4}$$

---

### 2.2 Model Architecture: Hybrid MambaVision Backbone

#### **Input**: Bridge State Tensor
$$\mathbf{B}_{in} = [\mathbf{z}_t, \mathbf{x}_{rgb\_gray}, \alpha_{map}, \sigma_{map}] \in \mathbb{R}^{B \times 4 \times H \times W}$$

where:
- $\mathbf{z}_t$: 1-channel bridge state.
- $\mathbf{x}_{rgb\_gray}$: 1-channel enhanced grayscale.
- $\alpha_{map}$: 1-channel scalar $\alpha(t)$ broadcast to full spatial resolution.
- $\sigma_{map}$: 1-channel scalar $\sigma(t)$ broadcast to full spatial resolution.

#### **Architecture Layers**

**1. Stem Block (Context Initialization)**
$$\mathbf{h}_0 = \text{GELU}\left(\text{Conv}_{3 \times 3}^{64}(\mathbf{B}_{in})\right)$$
$$\mathbf{h}_0 = \text{GELU}\left(\text{Conv}_{3 \times 3}^{64}(\mathbf{h}_0)\right)$$

Maps 4-channel bridge state to 64-channel feature space.

**2. Local Block (Fine-Grained Spatial Context)**
$$\mathbf{h}_{local} = \mathbf{h}_0 + \text{GELU}\left(\text{Conv}_{3 \times 3}^{64}\left(\text{GELU}\left(\text{Conv}_{3 \times 3}^{64}(\mathbf{h}_0)\right)\right)\right)$$

Residual convolution for local structural feature extraction.

**3. Global Block (Token Mixer for Long-Range Dependencies)**
$$\mathbf{m}_{in} = \text{Conv}_{1 \times 1}^{64}(\mathbf{h}_{local})$$
$$\mathbf{m}_{conv} = \text{Conv}_{7 \times 7, \text{groups}=64}^{64}(\mathbf{m}_{in}) \quad \text{(depthwise)}$$
$$\mathbf{m}_{out} = \text{Conv}_{1 \times 1}^{64}(\mathbf{m}_{conv})$$
$$\mathbf{h}_{global} = \mathbf{h}_{local} + \mathbf{m}_{out}$$

Depthwise separable convolution (7×7 kernel) captures global context efficiently without dense matrix operations.

**4. Head (Noise Prediction)**
$$\hat{\epsilon} = \text{Conv}_{1 \times 1}^{1}(\mathbf{h}_{global})$$

Single-channel output predicting the injected noise $\epsilon$.

#### **Full Forward Pass**
$$\hat{\epsilon} = \text{Backbone}(\mathbf{B}_{in})$$

---

### 2.3 IR Recovery (Reverse Bridge Equation)

Given the predicted noise $\hat{\epsilon}$ from the backbone, recover the estimated IR image:

$$\hat{x}_{ir} = \frac{z_t - (1 - \alpha) \cdot x_{rgb\_gray} - \sigma \cdot \hat{\epsilon}}{\alpha}$$

**Numerical Safety**:
- Clamp denominator: $\alpha \gets \text{clamp}(\alpha, 10^{-4})$
- Clamp output: $\hat{x}_{ir} \gets \text{clamp}(\hat{x}_{ir}, 0, 1)$

This rearrangement solves the bridge equation linearly for the target sample:
$$z_t = \alpha \cdot x_{ir} + (1-\alpha) \cdot x_{rgb\_gray} + \sigma \cdot \epsilon$$

Rearranging:
$$\alpha \cdot x_{ir} = z_t - (1-\alpha) \cdot x_{rgb\_gray} - \sigma \cdot \epsilon$$
$$x_{ir} = \frac{z_t - (1-\alpha) \cdot x_{rgb\_gray} - \sigma \cdot \epsilon}{\alpha}$$

---

### 2.4 Composite Loss Function

#### **2.4.1 Noise Prediction Loss**
$$\mathcal{L}_{noise} = \mathbb{E}_{t, x_{rgb}, x_{ir}} \left[ \left\| \hat{\epsilon} - \epsilon \right\|_2^2 \right]$$

MSE between predicted and actual injected noise.

#### **2.4.2 Edge-Aware Structural Loss**
$$\mathcal{L}_{edge} = \sum_{k=1}^{3} \mathbb{E}_{x_{ir}} \left[ \left\| \nabla^2_{LoG_k}(\hat{x}_{ir}) - \nabla^2_{LoG_k}(x_{ir}) \right\|_1 \right]$$

where $\nabla^2_{LoG_k}$ is the Laplacian-of-Gaussian operator at scale $k$.

**LoG Scales Used**:
- Scale 1: Kernel $5 \times 5$, $\sigma = 0.8$
- Scale 2: Kernel $7 \times 7$, $\sigma = 1.2$
- Scale 3: Kernel $9 \times 9$, $\sigma = 1.6$

L1 loss ensures edge preservation without washing out fine details.

#### **2.4.3 Semantic Consistency Loss**
$$\mathcal{L}_{sem} = \left\| \phi_{DINOv2}(x_{rgb\_enh}) - \phi_{DINOv2}(\hat{x}_{ir} \times 3) \right\|_2^2$$

where:
- $\phi_{DINOv2}$: Frozen DINOv2-ViT-L/14 feature extractor (no gradients).
- $\hat{x}_{ir} \times 3$: Repeat grayscale IR to 3 channels for DINO input.
- **Goal**: Ensure semantic consistency between enhanced RGB and predicted IR.

**Stage-Controlled Scale**:
$$\mathcal{L}_{sem}^{\text{active}} = s_{sem} \cdot \mathcal{L}_{sem}$$

where $s_{sem} = 0$ in Stage 1, $s_{sem} > 0$ in Stages 2–3.

#### **2.4.4 Task-Level Detection Loss**
$$\mathcal{L}_{task} = \left\| \phi_{RF\text{-}DETR}(\hat{x}_{ir}) - \phi_{RF\text{-}DETR}(x_{ir}) \right\|_2^2$$

where:
- $\phi_{RF\text{-}DETR}$: Frozen RF-DETR backbone feature extractor.
- **Goal**: Align internal detection features so predicted IR is ready for downstream detection.

**Stage-Controlled Scale**:
$$\mathcal{L}_{task}^{\text{active}} = s_{task} \cdot \mathcal{L}_{task}$$

where $s_{task} = 0$ in Stages 1–2, $s_{task} > 0$ in Stage 3.

#### **2.4.5 Entropy/Mode Coverage Loss**
$$\mathcal{L}_{entropy} = \text{KL}\left( \mathcal{N}(\mu(\hat{x}_{ir}), \sigma(\hat{x}_{ir})) \,\big\|\, \mathcal{N}(0, I) \right)$$

where:
- $\mu(\hat{x}_{ir}) = \text{mean}(\hat{x}_{ir})$ over spatial dimensions.
- $\sigma(\hat{x}_{ir}) = \text{std}(\hat{x}_{ir}) + \epsilon_{var}$, with variance clamping $\epsilon_{var} = 10^{-5}$.

**Interpretation**: Regularizes the output distribution to be close to unit Gaussian, preventing mode collapse and degenerate outputs.

#### **2.4.6 Total Composite Loss**
$$\boxed{\mathcal{L}_{total} = w_n \cdot \mathcal{L}_{noise} + w_e \cdot \mathcal{L}_{edge} + s_{sem} \cdot \mathcal{L}_{sem} + s_{task} \cdot \mathcal{L}_{task} + w_k \cdot \mathcal{L}_{entropy}}$$

**Default Weights** (empirically tuned):
- $w_n = 1.0$ (primary structural signal)
- $w_e = 0.5$ (balance with noise loss)
- $s_{sem}$: curriculum-dependent (0 or 0.1)
- $s_{task}$: curriculum-dependent (0 or 0.2)
- $w_k = 0.01$ (weak regularization)

---

### 2.5 Bridge Time Sampling Strategy

During training, bridge time $t$ is sampled for each batch:

$$t \sim \mathcal{U}(0, 1) \quad \text{Uniform sampling}$$

or

$$t \sim \text{Beta}(3, 3) \quad \text{Beta distribution (concentrates on mid-range for higher variance)}$$

**Biological Interpretation**: 
- Mid-range times ($t \approx 0.5$) have maximum stochasticity (variance $\sigma(0.5) = \sigma_{max}$).
- Early times ($t \to 0$) are deterministic (endpoint on target side).
- Late times ($t \to 1$) are deterministic (endpoint on source side).

---

## 3. Training Pipeline

### 3.1 Three-Stage Curriculum

| Stage | Epochs | $s_{sem}$ | $s_{task}$ | Objective | Key Milestones |
|-------|--------|-----------|-----------|-----------|---|
| **1: Structural Warmup** | 1–41 | 0.0 | 0.0 | MSE noise + edge | Learn basic RGB→IR structural mapping |
| **2: Semantic Anchoring** | 42–150 | 0.1 | 0.0 | + DINO guidance | Enforce content/semantic consistency |
| **3: Task-Aware Refinement** | 151–300 | 0.1 | 0.2 | + detection features | Align with downstream detection |

### 3.2 Optimization Details

**Optimizer**: Adam with learning rate $\eta = 1 \times 10^{-5}$

**Mixed Precision**: FP16 forward, FP32 accumulation (AMP with GradScaler)

**Gradient Clipping**: $\|\nabla \theta\|_2 \leq 1.0$ per step

**Finite-Loss Guard**: Skip batch if loss is NaN or Inf

**Batch Size**: 8 per GPU; gradient accumulation over 4 steps → effective 32

---

## 4. Quantitative Results

### 4.1 Full Test Split Evaluation (3463 Images)

| Epoch | Stage | **PSNR (↑)** | **SSIM (↑)** | **LPIPS (↓)** | **mAP@0.5 (↑)** | **mAP@0.5:0.95 (↑)** |
|-------|-------|---|---|---|---|---|
| 50 | Stage 1 | 23.58 | 0.584 | 0.562 | 0.015 | 0.010 |
| 150 | Stage 2 | 24.00 | 0.595 | 0.512 | 0.053 | 0.025 |
| **300** | **Stage 3** | **24.18** | **0.609** | **0.485** | **0.097** | **0.045** |

**Stage-Wise Gains**:
- PSNR: +0.60 (50→300)
- SSIM: +0.025
- LPIPS: -0.077 (better perceptual quality)
- **mAP@0.5: +0.082 (6.5x improvement)**
- **mAP@0.5:0.95: +0.035 (4.5x improvement)**

---

## 5. Qualitative Insights

### 5.1 RGB-IR Relationship in Low-Light

**Physical Model**:
$$I_{rgb} = R \cdot L + \text{Noise}$$
$$I_{ir} \approx T + e$$

where:
- $R$: reflectance (intrinsic surface properties)
- $L$: illumination (highly variable in low light)
- $T$: thermal emission (robust, stable)
- $e$: sensor noise

**Learning Challenge**: The model must disentangle $R$ from $L$ in the RGB channel, then map feature to the stable thermal domain.

**Our Solution**: Bridge interpolation learns a **smooth probabilistic path** from RGB latent to IR target, avoiding the need for pixel-level color inversion.

---

## 6. Architectural Rationale

### 6.1 Why Hybrid Mamba + Vision?

| Component | Classical Approach | Mamba Benefit | Our Hybrid Approach |
|-----------|-------------------|--------------|-----|
| **Receptive Field** | Conv: Limited to kernel size | Attention: Heavy compute | Depthwise (7×7) + local convs |
| **Memory** | Full feature maps | None (sequential state) | Dynamic route based on $s_{sem}$, $s_{task}$ |
| **Efficiency** | Fixed compute | Adaptive | 64-dim intermediate, fast inference |
| **Inductive Bias** | Locality | None | Locality + global token mixing |

### 6.2 Why Frozen Oracles?

- **No Task Data**: We lack ground-truth detection labels for the entire training set; freezing avoids label leakage.
- **Stability**: Frozen features act as stable targets; unfrozen adaption would require adversarial correction.
- **Efficiency**: Single forward pass per oracle; no backprop overhead.

---

## 7. Implementation Highlights

### 7.1 Code Structure
```
FYP_3.0/
├── models/
│   ├── enhancement.py          # RAWMambaEnhancer (low-light preprocessing)
│   ├── mv_generator.py         # HybridMambaVisionBackbone + TokenMixer
│   ├── guidance.py             # Frozen DINOv2 + RF-DETR oracles
│   └── bridge.py               # MVGSBModel (main high-level API)
├── losses/
│   ├── structural.py           # Laplacian-of-Gaussian edge loss
│   └── objective.py            # Composite loss + entropy KL
├── train.py                    # 3-stage curriculum training loop
├── evaluate.py                 # PSNR/SSIM/LPIPS/mAP evaluation
└── dataset_llvip.py            # LLVIP paired data loader
```

### 7.2 Key Safe-Guards

1. **Denominator Clamping**: Prevents division by near-zero $\alpha$.
2. **Variance Clamping**: Entropy computation stable over 300 epochs.
3. **Gradient Clipping**: Avoids gradient explosion from frozen oracle perturbing high-loss regions.
4. **Finite-Loss Check**: Batch skipped if loss contains NaN/Inf.

---

## 8. Future Work & Extensions

1. **Full MambaVision-B Integration**: Replace placeholder with official state-space backbone.
2. **Cross-Dataset Validation**: Test on CVC-14, M3FD datasets beyond LLVIP.
3. **Real-Time Inference**: Deploy on edge devices (NVIDIA Jetson).
4. **Conditional Generation**: Control output temperature/emissivity via conditioning token.
5. **Hybrid Training**: Joint RGB→IR + detection end-to-end (no frozen oracles).

---

## 9. References & Inspirations

- **Schrodinger Bridges**: [Pavon et al., 2021] optimal transport with stochastic interpolation
- **DINOv2**: [Oquab et al., 2023] self-supervised vision features
- **Mamba**: [Gu & Dao, 2023] efficient sequence modeling with state-space networks
- **LLVIP Dataset**: [Jia et al., 2021] large-scale low-light visible-IR paired benchmark

---

## 10. Citation

If you use this work, please cite:

```bibtex
@article{fyp3_2026,
  title={MV-GSB: MambaVision-Guided Schrodinger Bridge for RGB-to-IR Translation in Low-Light Environments},
  author={Your Name},
  year={2026},
  journal={arXiv preprint}
}
```

---

**Last Updated**: March 27, 2026  
**Status**: ✅ Complete and Operational (300 epochs, 3463-sample test set validation)
