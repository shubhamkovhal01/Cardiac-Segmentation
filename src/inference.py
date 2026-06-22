import torch
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscreted
from monai.data import decollate_batch

from transforms import get_inference_transforms, get_post_transforms
from model import build_model


def predict(image_path, weights_path, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pre_transforms = get_inference_transforms()
    data = pre_transforms({"image": image_path})
    input_tensor = data["image"].unsqueeze(0).to(device)

    model = build_model(device=device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    with torch.no_grad():
        output = sliding_window_inference(input_tensor, (128, 128, 80), 4, model)

    # attach raw model output as "pred" in the data dict, so Invertd can map it back
    data["pred"] = output[0]

    post_transforms = get_post_transforms(pre_transforms)
    discrete = AsDiscreted(keys="pred", argmax=True)

    data = discrete(data)
    data = post_transforms(data)

    pred_mask = data["pred"][0].cpu().numpy()  # back in original space now

    return pred_mask
