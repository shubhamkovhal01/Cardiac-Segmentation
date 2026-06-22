import torch
from monai.inferers import sliding_window_inference

from transforms import get_inference_transforms
from model import build_model


def predict(image_path, weights_path, device=None):
    """
    Run left atrium segmentation inference on a single NIfTI file.

    Args:
        image_path: path to input .nii.gz file
        weights_path: path to trained model .pth weights
        device: torch device, auto-detected if None

    Returns:
        pred_mask: numpy array, discrete segmentation mask (0=background, 1=atrium)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # preprocess
    transforms = get_inference_transforms()
    data = transforms({"image": image_path})
    input_tensor = data["image"].unsqueeze(0).to(device)  # add batch dimension

    # load model
    model = build_model(device=device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    # inference
    with torch.no_grad():
        output = sliding_window_inference(input_tensor, (128, 128, 80), 4, model)
        pred_mask = torch.argmax(output, dim=1)[0].cpu().numpy()

    return pred_mask
