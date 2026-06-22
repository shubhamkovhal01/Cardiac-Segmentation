import torch
from monai.networks.nets import SegResNet


def build_model(device=None):
    """
    Returns the SegResNet architecture used for left atrium segmentation.
    Weights must be loaded separately via model.load_state_dict(...).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SegResNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        init_filters=32,
        blocks_down=(1, 2, 2, 4),
        blocks_up=(1, 1, 1),
        dropout_prob=0.2,
    ).to(device)

    return model
