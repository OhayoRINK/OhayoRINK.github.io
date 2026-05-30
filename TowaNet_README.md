# TowaNet 👾

> **Topology-Oriented Wireless Autoencoder for CSI Feedback Compression in Massive MIMO Systems**

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Key Features](#key-features)
- [Key Improvements over DPAS](#key-improvements-over-dpas)
- [Inherited from DPAS](#inherited-from-dpas)
- [Inherited from PekoraNet](#inherited-from-pekoranet)
- [Model Specs](#model-specs)
- [Quick Start](#quick-start)
- [Experimental Results](#experimental-results)
- [Future Work](#future-work)
- [Citation](#citation)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Overview

**TowaNet** (Topology-Oriented Wireless Autoencoder) is a deep learning-based CSI (Channel State Information) feedback compression model for Massive MIMO FDD systems.

TowaNet fully inherits the core structure of DPAS-CsiNet (DualPathBlock, AFG, ProgressiveFeedbackLoss) while introducing a **stride=2 spatial compression path** in the Encoder and replacing the decoder's SRB with **TowaRefineBlock**, motivated by the decoder-centric reconstruction philosophy demonstrated in both DPAS-CsiNet and PekoraNet.

```
Input CSI (2×32×32)
       │
  [ TowaEncoder ]
  stage1: DualPath(2→16, 32×32)
  stage2: DualPath(16→32, 32×32)
       │ aux_proj → Auxiliary codeword (ProgressiveFeedbackLoss용)
  compress: stride=2 conv (32×32 → 16×16) ★
  stage3: DualPath(32→32, 16×16)
  FC: 8192 → M bits
       │
  Codeword (M bits, [0,1])
       │
  [ TowaDecoder ]
  FC: M → 8192
  reshape: (32, 16, 16)
  upsample: bilinear 16×16 → 32×32
  refine: DualPath → RefineBlock×2 → DualPath(→2ch)
  Sigmoid
       │
Reconstructed CSI (2×32×32)
```

---

## Architecture

### DPAS vs TowaNet Structure

| Component | DPAS-CsiNet | TowaNet |
|---|---|---|
| **Encoder channels** | 2→16→16 (fixed) | 2→16→32 (expanded at stage2) |
| **Spatial resolution** | 32×32 maintained | stride=2 → 16×16 compressed ★ |
| **FC input dim** | 16×32×32 = **16,384** | 32×16×16 = **8,192** (50% reduction) |
| **Decoder start** | FC→2×32×32 direct reshape | FC→32×16×16 → bilinear upsample ★ |
| **Refinement block** | SpectralRefinementBlock (FFT conv) | TowaRefineBlock (spatial residual + SE + AFG) |
| **SE in DualPathBlock** | included | **removed** (overlap with AFG) |
| **Parameters** | ~9M | ~8.5M |

### Encoder

```
Input (B, 2, 32, 32)
    │
[stage1] TowaDualPathBlock(2 → 16)    → (B, 16, 32, 32)
    │
[stage2] TowaDualPathBlock(16 → 32)   → (B, 32, 32, 32)
    │                 └──► aux_proj (AdaptiveAvgPool → FC) → Aux codeword
    │
[compress] Conv2d stride=2            → (B, 32, 16, 16)  ★
    │
[stage3] TowaDualPathBlock(32 → 32)   → (B, 32, 16, 16)
    │
[FC] Linear(8192 → M) + Sigmoid       → (B, M)
```

DPAS encodes a 16,384-dim flattened vector directly into the FC layer.
TowaNet applies `stride=2` conv first to reduce spatial resolution to 16×16, cutting the FC input dimension in half while learning spatially hierarchical features during compression — an approach inspired by PekoraNet's hierarchical downsampling encoder design.

### Decoder

```
Codeword (B, M)
    │
[FC] Linear(M → 8192)                 → (B, 8192)
    │
[reshape]                             → (B, 32, 16, 16)
    │
[upsample] bilinear 16×16 → 32×32    → (B, 32, 32, 32)  ★
    + Conv2d(32→32) + BN + LReLU
    │
[refine]
  TowaDualPathBlock(32 → 32)
  TowaRefineBlock(32)
  TowaRefineBlock(32)
  TowaDualPathBlock(32 → 2)           → (B, 2, 32, 32)
    │
[Sigmoid]                             → Reconstructed CSI (B, 2, 32, 32)
```

The latent upsample approach — starting from a 32-channel 16×16 representation before bilinear upsampling — is inspired by PekoraNet's ConvTranspose-based learned upsampling decoder, which demonstrated that richer intermediate representations improve spatial reconstruction quality.

### Building Blocks

#### TowaDualPathBlock

Two parallel paths capture different physical characteristics of the channel:

- **Path A** — `3×3 Conv`: spatial (antenna-delay) correlation
- **Path B** — `1×W Conv`: delay-domain horizontal correlation
- **Fuse** — `1×1 Conv`: channel integration after concat
- **AFG** — Adaptive Frequency Gating: FFT magnitude-based channel weighting

> SE is removed from DualPathBlock (present in DPAS). AFG already performs channel-wise importance weighting via FFT magnitude, making SE redundant. SE is reserved exclusively for TowaRefineBlock.

#### TowaRefineBlock

Replaces DPAS's SpectralRefinementBlock (SRB):

```
Input x
   │
[Conv3×3 → BN → LReLU → Conv3×3 → BN]  ← 2-layer spatial residual
   │
[SEBlock]  ← channel attention
   │
residual: x + SE(conv(x))
   │
[AFG]  ← frequency-domain refinement
   │
Output
```

| | SRB (DPAS) | TowaRefineBlock |
|---|---|---|
| Refinement domain | FFT complex conv | spatial residual |
| Channel selection | none | SEBlock |
| Frequency refinement | FFT irfft2 direct | AFG indirect |
| AMP compatibility | complex half precision issue | fully compatible |
| Training stability | learnable alpha required | standard residual |

#### AdaptiveFrequencyGating (AFG)

Identical to DPAS:

```python
x_fft = torch.fft.rfft2(x)             # FFT transform
mag   = x_fft.abs().mean(dim=(-2,-1))  # per-channel mean magnitude
gate  = FC(mag) → Sigmoid              # learnable gate weights
output = x * gate                      # applied in spatial domain
```

---

## Key Features

### 1. Spatial Compression Encoder (stride=2)

DPAS maintains 32×32 resolution through the encoder and passes a 16,384-dim vector to the FC layer. TowaNet downsamples to 16×16 via `stride=2` conv before the FC, reducing the input dimension to 8,192 (50% reduction). This enables selective feature learning during spatial compression rather than naive flattening.

### 2. Channel Expansion (16 → 32)

DPAS fixes channel width at 16 throughout the encoder. TowaNet expands to 32 channels at stage2 for richer feature representation before spatial compression.

### 3. Latent-Space Upsampling Decoder

DPAS Decoder outputs 2×32×32 directly from FC. TowaNet starts from a 32-channel 16×16 latent and recovers spatial resolution via bilinear upsample, providing richer intermediate representation for the refinement blocks.

### 4. TowaRefineBlock (SRB → spatial residual + SE + AFG)

DPAS's SRB performs complex-valued convolution in the FFT domain, which causes AMP (Automatic Mixed Precision) compatibility issues. TowaRefineBlock replaces this with a standard spatial residual + SEBlock + AFG combination, achieving full AMP compatibility without sacrificing refinement quality.

### 5. SE Role Separation

DPAS DualPathBlock includes both SE and AFG. TowaNet removes SE from DualPathBlock (keeping only AFG) and reserves SE exclusively for TowaRefineBlock, establishing clear functional separation between feature extraction and reconstruction refinement stages.

---

## Inherited from DPAS

TowaNet fully inherits the following components from DPAS-CsiNet without modification:

| Component | Description |
|---|---|
| `AdaptiveFrequencyGating (AFG)` | FFT magnitude-based channel gating, identical implementation |
| `DualPathBlock concept` | Path A (3×3 spatial) + Path B (1×W delay-domain) dual path |
| `ProgressiveFeedbackLoss` | Main MSE loss + Auxiliary loss (adjustable aux_weight) |
| `Auxiliary branch` | AdaptiveAvgPool → FC aux codeword from encoder mid-stage |
| `Sigmoid codeword` | Final codeword normalized to [0, 1] |

---

## Inherited from PekoraNet

TowaNet adopts the following design principles from PekoraNet 🐇:

| Component | PekoraNet | TowaNet |
|---|---|---|
| **Hierarchical downsampling encoder** | `Conv2d stride=2` × 2 (32×32 → 16×16 → 8×8) | `stride=2` conv (32×32 → 16×16) ★ |
| **Latent upsample decoder** | `ConvTranspose2d` × 2 (learned upsampling) | `bilinear upsample` 16×16 → 32×32 ★ |
| **Decoder-centric design** | heavier decoder FLOPs than encoder | majority of refinement capacity allocated to BS-side decoder |
| **ResidualBlock in decoder** | single `ResidualBlock` for refinement | `TowaRefineBlock` (spatial residual + SE + AFG) |
| **Asymmetric encoder-decoder** | lightweight UE encoder / heavy BS decoder | identical philosophy |

### Key Adaptations

**Hierarchical spatial compression (Encoder):**  
PekoraNet demonstrates that applying `stride=2` Conv before the FC layer — rather than directly flattening the full-resolution feature map — significantly improves feature learning efficiency. TowaNet inherits this approach with a single `stride=2` compression stage (32×32 → 16×16), combined with channel expansion (16→32) from DPAS-style DualPathBlocks.

**Latent upsample decoder:**  
PekoraNet's ablation study (Ablation B-3) confirms that replacing ConvTranspose-based learned upsampling with plain convolution causes major NMSE degradation (-15.72 dB vs -22.07 dB). This finding directly motivated TowaNet's decoder design: instead of reshaping the FC output directly to the target resolution, TowaNet reconstructs a 32-channel 16×16 latent and applies bilinear upsampling, enabling richer intermediate representations for the subsequent refinement blocks.


---

## Model Specs

| CR | Feedback Bits (M) | Parameters (est.) |
|---|---|---|
| 1/4 | 512 | ~9.5M |
| 1/8 | 256 | ~5.0M |
| 1/16 | 128 | ~2.6M |
| 1/32 | 64 | ~1.4M |

---

## Quick Start

### Requirements

```bash
pip install torch torchvision
```

### Training

```python
from towanet import TowaNet, ProgressiveFeedbackLoss
import torch

model = TowaNet(feedback_bits=512).cuda()
criterion = ProgressiveFeedbackLoss(aux_weight=0.1)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for x in dataloader:
    x = x.cuda()
    recon, aux = model(x)
    loss, ml, al = criterion(recon, x, aux, feedback_bits=512)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### Smoke Test

```bash
python towanet.py
```

Expected output:
```
Device : cuda
Params : 9,xxx,xxx
Forward : x.xxxs (batch=200, no_grad)
Recon : (200, 2, 32, 32)
Loss : x.xxxx (main=x.xxxx, aux=x.xxxx)

Smoke test passed!
```

### Encoder-only Inference (UE side)

```python
model.eval()
with torch.no_grad():
    codeword, _ = model.encoder(csi_input)  # (B, M) — feedback to BS
```

---

## Experimental Results

> Results vary depending on dataset (Indoor/Outdoor), training epochs, and scheduler configuration.

### Indoor NMSE

| Model | CR=1/4 | CR=1/16 | CR=1/32 | CR=1/64 |
|---|---|---|---|---|
| CsiNet (baseline) | -17.36 dB | -8.65 dB | -6.24 dB | -5.84 dB |
| PekoraNet | -22.067 dB | -12.749 dB | -9.199 dB | -7.048 dB |
| DPAS-CsiNet | -24.72 dB | -13.52 dB | -9.35 dB | -6.64 dB |
| **TowaNet** | **-25.52 dB** | **TBD** | **TBD** | **TBD** |
| **TowaNet(cosine)** | **-26.50 dB** | **TBD** | **TBD** | **TBD** |

### Outdoor NMSE

| Model | CR=1/4 | CR=1/16 | CR=1/32 | CR=1/64 |
|---|---|---|---|---|
| CsiNet (baseline) | -8.75 dB | -4.51 dB | -2.81 dB | -1.93 dB |
| PekoraNet | -11.566 dB | -5.169 dB | -3.329 dB | -2.259 dB |
| DPAS-CsiNet | -15.15 dB | -6.56 dB | -4.37 dB | -2.49 dB |
| **TowaNet** | **TBD** | **TBD** | **TBD** | **TBD** |
| **TowaNet(cosine)** | **TBD** | **TBD** | **TBD** | **TBD** |

> NMSE (Normalized Mean Square Error): lower is better.

---

## Future Work

- Quantization-aware training
- Adaptive compression ratio support
- Temporal CSI modeling
- Transformer-free attention integration
- Joint indoor/outdoor multi-scenario training
- Formal NMSE benchmarking against CsiNet, CRNet, DPAS

---

## Citation

If you use TowaNet in your research or projects, please consider citing this repository.

Formal publication information will be added in a future release.

---

## License

MIT License

---

## Acknowledgements

TowaNet is built upon and inspired by the following works:

- **CsiNet** — Wen et al. (2018), the foundational deep learning-based CSI feedback compression framework that established the autoencoder paradigm for massive MIMO CSI compression
- **CRNet** — Multi-resolution CNN-based CSI compression baseline
- **DPAS-CsiNet** (Dual-Path Adaptive Spectral CsiNet) — The core architectural foundation of TowaNet. The DualPathBlock (Path A spatial + Path B delay-domain), AdaptiveFrequencyGating (AFG), and ProgressiveFeedbackLoss are directly inherited from this work. The decoder-centric refinement strategy using SpectralRefinementBlock also motivated TowaNet's decoder design
- **PekoraNet** 🐇 — The hierarchical downsampling encoder (stride=2 spatial compression) and decoder latent upsample approach in TowaNet are directly inspired by PekoraNet's hierarchical CNN compression and ConvTranspose-based learned upsampling decoder design
