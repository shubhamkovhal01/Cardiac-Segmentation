# Cardiac Left Atrium Segmentation

3D left atrium segmentation from cardiac MRI using SegResNet, trained on the 
Medical Segmentation Decathlon Task02_Heart dataset.

**Live demo:** [HuggingFace Space](https://huggingface.co/spaces/SSKO2119/cardiac-segmentation)  
**Model weights:** [HuggingFace Hub](https://huggingface.co/SSKO2119/cardiac-segmentation-segresnet)

---

## Results

| Patient | Dice | HD95 (mm) |
|---------|------|-----------|
| 1 | 0.910 | 3.74 |
| 2 | 0.873 | 36.40 |
| 3 | 0.885 | 8.80 |
| 4 | 0.890 | 4.36 |
| **Mean** | **0.890** | **13.33** |

Patient 2's elevated HD95 is a diagnosed, specific failure mode — not an unexplained 
anomaly. The model's prediction correctly identified the atrium but truncated 6 slices 
(9mm) at the cranial extent, driven by this patient's atrium spanning 69 of the 72 
maximum observed slices in the training set — close to the boundary of the 80-slice 
patch used during training. Connected component analysis confirmed a single contiguous 
prediction (no spurious blobs), isolating the cause to boundary truncation rather than 
false positives.

---

## Dataset

**Medical Segmentation Decathlon — Task02_Heart**
- Modality: LGE-MRI (Late Gadolinium Enhancement)
- Target: Left atrium (binary segmentation)
- 20 labelled training volumes, 10 unlabelled test volumes
- All volumes: 320×320 in-plane, Z depth 90–130 slices
- Uniform voxel spacing: (1.25, 1.25, 1.37) mm across all patients

Split: 16 train / 4 val (random shuffle, seed=42)

---

## Architecture

**SegResNet** (Myronenko, 2018) — a residual encoder-decoder network designed 
specifically for 3D medical image segmentation.

Encoder: 128³ → 64³ → 32³ → 16³  (residual blocks, channel doubling)
Decoder: 16³ → 32³ → 64³ → 128³  (transpose convolutions + skip connections)

Key parameters:
- `spatial_dims=3` — full 3D convolutions
- `in_channels=1` — single MRI channel
- `out_channels=2` — background + left atrium
- `init_filters=32`, `blocks_down=(1,2,2,4)`, `blocks_up=(1,1,1)`
- `dropout_prob=0.2` — regularisation for small dataset

Skip connections preserve fine boundary detail that the encoder's downsampling 
discards — critical for precise atrial wall delineation relevant to ablation planning.

---

## Preprocessing Pipeline

All transforms applied identically at training and inference time:

| Step | Transform | Reasoning |
|------|-----------|-----------|
| Load | `LoadImaged` | NIfTI → tensor |
| Channel | `EnsureChannelFirstd` | Add channel dimension |
| Resample | `Spacingd(1.5, 1.5, 1.5)` | Standardise voxel size across patients |
| Orient | `Orientationd(RAS)` | Consistent anatomical orientation |
| Crop | `CropForegroundd` | Remove empty margins, reduce memory |
| Normalise | `ScaleIntensityRangePercentilesd(0.5, 99.5)` | Percentile clipping handles MRI intensity variation |

**Patch size selection:** Empirical profiling showed the atrium spans 49–72 Z-slices 
across training patients after foreground cropping. Patch Z size set to 80 — 8-slice 
margin above the maximum observed atrium extent — providing full atrial coverage while 
fitting within all patients' cropped volume sizes.

**Spacing standardisation** makes the model's spatial priors transferable to new 
patients: the same physical patch size (in mm) is maintained regardless of native 
scanner resolution.

---

## Training

- **Loss:** `DiceCELoss` — Dice component addresses class imbalance 
  (atrium ≈ 2–3% of total volume), CE component provides stable early gradients
- **Optimiser:** Adam, lr=1e-4, weight_decay=1e-5
- **Patch sampling:** `RandCropByPosNegLabeld`, pos:neg=1:1, 2 patches per volume
- **Augmentation:** Random affine (rotation ±20°, scale ±10%), random flips 
  (all axes), Gaussian noise — all applied with `prob=0.5`
- **Batch size:** 2 volumes × 2 patches = 4 effective patches per step
- **Epochs:** 100, validation every 5 epochs
- **Validation:** Full-volume sliding window inference (128×128×80 windows, 
  overlap handled by MONAI)
- **Checkpointing:** Best validation Dice saved

**Training hardware:** Google Colab Pro, A100 GPU

---

## Inference

At inference time, predictions are mapped back to original patient space via 
MONAI's `Invertd` — reversing resampling, orientation, and cropping operations 
using the MetaTensor transform history. This ensures the output segmentation mask 
aligns precisely with the original NIfTI file's coordinate system, suitable for 
clinical overlay or downstream processing.

---

## Repository Structure
cardiac-segmentation/
├── notebooks/
│   └── 01_data_exploration.ipynb   # EDA, transforms, training, evaluation
├── src/
│   ├── transforms.py               # Preprocessing + inverse transform pipeline
│   ├── model.py                    # SegResNet definition
│   └── inference.py                # Full predict() function
├── app/
│   ├── app.py                      # Gradio demo
│   └── requirements.txt
└── README.md

---

## Clinical Context

Left atrium segmentation supports atrial fibrillation ablation planning — precise 
atrial geometry informs catheter placement during pulmonary vein isolation procedures. 
HD95 is therefore a clinically meaningful metric alongside Dice: boundary precision 
matters for ablation target definition, not just volumetric overlap.

This project is a direct extension of dissertation work on proxy token compression 
for cardiac MRI vision-language models (MViT encoder, ACDC dataset), exploring 
the CNN segmentation baseline against which transformer-based architectures can 
be compared.

---

## Limitations

- Validation set of n=4 produces high-variance Dice estimates across epochs; 
  5-fold cross-validation would give more statistically reliable performance bounds
- Fixed 80-slice patch Z dimension may under-represent patients with larger-than-typical 
  atrial extent (confirmed in one val case, Patient 2)
- Connected component post-processing evaluated but found unnecessary for this model — 
  all predictions formed single contiguous regions with no spurious blobs
- Inference on CPU (HuggingFace free tier) takes approximately 2–3 minutes per volume

---

## References

- Myronenko, A. (2018). 3D MRI Brain Tumor Segmentation Using Autoencoder 
  Regularization. *BrainLes Workshop, MICCAI.*
- Simpson, A.L. et al. (2019). A large annotated medical image dataset for the 
  development and evaluation of segmentation algorithms. *arXiv:1902.09063.*
- MONAI Consortium. Project MONAI. https://monai.io
