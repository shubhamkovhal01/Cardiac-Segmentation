import sys
import os

# add src/ to path — works both locally and in HuggingFace Space
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import torch
from huggingface_hub import hf_hub_download

from inference import predict

# download weights from HF Hub on startup
WEIGHTS_PATH = hf_hub_download(
    repo_id="SSKO2119/cardiac-segmentation-segresnet",
    filename="best_metric_model.pth",
)

def segment(nifti_file):
    image_path = nifti_file.name
    mask = predict(image_path, WEIGHTS_PATH)

    img = nib.load(image_path).get_fdata()

    z_slices_with_pred = np.where(mask.sum(axis=(0, 1)) > 0)[0]
    if len(z_slices_with_pred) == 0:
        z_indices = [img.shape[2] // 2]
    else:
        z_indices = z_slices_with_pred[
            np.linspace(0, len(z_slices_with_pred) - 1, 3, dtype=int)
        ]

    fig, axes = plt.subplots(1, len(z_indices), figsize=(5 * len(z_indices), 5))
    if len(z_indices) == 1:
        axes = [axes]

    for ax, z in zip(axes, z_indices):
        ax.imshow(img[:, :, z], cmap='gray', origin='lower')
        ax.imshow(mask[:, :, z], cmap='hot', alpha=0.4, origin='lower')
        ax.set_title(f'Slice z={z}')
        ax.axis('off')

    plt.tight_layout()
    return fig


demo = gr.Interface(
    fn=segment,
    inputs=gr.File(label="Upload cardiac MRI (.nii.gz)"),
    outputs=gr.Plot(label="Left atrium segmentation"),
    title="Cardiac Left Atrium Segmentation",
    description="Upload a cardiac MRI NIfTI file (.nii.gz). The model segments the left atrium using a SegResNet trained on the Medical Segmentation Decathlon Task02_Heart dataset (mean Dice: 0.890).",
    examples=[],
)

if __name__ == "__main__":
    demo.launch()
