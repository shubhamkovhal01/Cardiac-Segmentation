from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    Orientationd,
    CropForegroundd,
    ScaleIntensityRangePercentilesd,
    Invertd,
)


def get_inference_transforms():
    """
    Returns the deterministic preprocessing pipeline used for inference.
    Must match the shared_transforms used during training exactly.
    """
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Spacingd(keys=["image"], pixdim=(1.5, 1.5, 1.5), mode="bilinear"),
        Orientationd(keys=["image"], axcodes="RAS"),
        CropForegroundd(keys=["image"], source_key="image"),
        ScaleIntensityRangePercentilesd(
            keys=["image"], lower=0.5, upper=99.5, b_min=0.0, b_max=1.0, clip=True
        ),
    ])


def get_post_transforms(pre_transforms):
    """
    Returns a transform that inverts preprocessing to map predictions
    back to original patient coordinate space.
    """
    return Compose([
        Invertd(
            keys="pred",
            transform=pre_transforms,
            orig_keys="image",
            nearest_interp=True,
        ),
    ])
