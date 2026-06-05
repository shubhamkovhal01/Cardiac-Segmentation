# Cardiac Segmentation — Left Atrium (MSD Task02)

3D left atrium segmentation from cardiac MRI using MONAI, 
trained on the Medical Segmentation Decathlon Task02_Heart dataset 
and deployed on Hugging Face Spaces.

---

## Clinical Motivation

Accurate left atrium segmentation is a key step in diagnosing and 
planning treatment for **atrial fibrillation (AF)** — one of the most 
common cardiac arrhythmias. Manual segmentation is time-consuming 
and subject to inter-observer variability. This project automates 
that process using a deep learning pipeline.

---

## Dataset

| Property | Detail |
|---|---|
| Source | Medical Segmentation Decathlon — Task02_Heart |
| Modality | MRI (single modality) |
| Target | Left atrium (binary segmentation) |
| Training cases | 20 patients |
| Test cases | 10 patients |
| Format | NIfTI (.nii.gz) |
| Challenge | Small dataset with large anatomical variability |

---
## Dataset Exploration

All 20 training volumes share consistent voxel spacing (1.25 × 1.25 × 1.37 mm).
The only variation is in the z-axis (slice depth), ranging from 90 to 130 slices.

| Property | Value |
|---|---|
| In-plane resolution | 320 × 320 |
| Voxel spacing (x, y) | 1.25 mm |
| Voxel spacing (z) | 1.37 mm |
| Slice depth range | 90 – 130 slices |
| Intensity range | ~0 – 2200 (raw MRI units) |
| Spacing consistency | Identical across all 20 patients |

The consistent spacing means **no resampling is required** across patients — 
a preprocessing step that can be skipped for this dataset.

## Augmentation Strategy

The small training set (20 cases) makes augmentation essential to 
prevent overfitting and simulate real-world variability in patient 
positioning and scanner settings:

- **Rotation** — patients are not identically positioned across scans
- **Scaling** — accounts for anatomical size differences
- **Elastic deformation** — simulates soft tissue variability
- **Gamma / contrast augmentation** — simulates scanner variability
- **Gaussian noise** — improves robustness to acquisition noise
- **Foreground-aware patch sampling** — ensures the atrium is 
  present in training patches (critical for small structures)

---

## Stack

| Component | Tool |
|---|---|
| Segmentation framework | MONAI |
| Training environment | Google Colab Pro (A100 GPU) |
| Deployment | Hugging Face Spaces |
| Language | Python |

---

## Repository Structure
cardiac-segmentation/
├── notebooks/       # Colab training notebooks
├── src/             # Dataloaders, transforms, model code
├── app/             # Hugging Face Spaces deployment
└── README.md

---

## Results

*Training in progress — results will be updated here.*

| Metric | Score |
|---|---|
| Dice (Left Atrium) | TBD |
| Hausdorff Distance | TBD |

---

## Reference

Antonelli et al., *The Medical Segmentation Decathlon*, 
Nature Communications, 2022. 
https://doi.org/10.1038/s41467-022-30695-9