from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from PIL import Image, ImageOps
from tqdm import tqdm

from src.dl.infer import get_transforms
from src.dl.train import prepare_model
from src.ptypes import img_size, num_labels


def compute_gradcam(model, target_layer, img, target_class=None):
    model.eval()

    # Hook the target layer
    def save_gradient(module, grad_input, grad_output):
        nonlocal gradient
        gradient = grad_output[0]

    def forward_hook(module, input, output):
        nonlocal activations
        activations = output

    activations = None
    gradient = None
    handle_forward = target_layer.register_forward_hook(forward_hook)
    handle_backward = target_layer.register_full_backward_hook(save_gradient)

    # Forward pass
    output = model(img)
    _, predicted = torch.max(output, 1)

    # If no specific class is targeted, use the predicted class
    if target_class is None:
        target_class = predicted.item()

    # Backward pass
    model.zero_grad()
    class_score = output[0, target_class]
    class_score.backward()

    # Compute weights
    pooled_gradients = torch.mean(gradient, dim=[0, 2, 3])
    for i in range(pooled_gradients.shape[0]):
        activations[:, i, :, :] *= pooled_gradients[i]

    heatmap = torch.sum(activations, dim=1).squeeze()
    heatmap = F.relu(heatmap)  # ReLU removes negative values
    heatmap /= torch.max(heatmap)

    # Remove the hooks after use
    handle_forward.remove()
    handle_backward.remove()

    return heatmap


def vis_heatmap(img_pil, heatmap, output_path):
    # Resize heatmap to match the original image size
    heatmap_resized = np.array(
        Image.fromarray(heatmap.detach().numpy()).resize(img_pil.size, Image.BILINEAR)
    )

    # Create a colormap
    colormap = plt.get_cmap("jet")
    heatmap_colored = colormap(heatmap_resized)

    # Convert heatmap to PIL image
    heatmap_pil = Image.fromarray((heatmap_colored[:, :, :3] * 255).astype(np.uint8))

    # Blend with original image
    blended_image = Image.blend(img_pil.convert("RGBA"), heatmap_pil.convert("RGBA"), alpha=0.4)

    # Save the resulting image
    blended_image.save(output_path)


def vis_gradcam(model, folder_to_run, output_path, target_layer, device):
    for class_folder in folder_to_run.iterdir():
        if class_folder.is_dir():
            print("Processing", class_folder.name)
            output_class_path = output_path / class_folder.name
            output_class_path.mkdir(parents=True, exist_ok=True)
            img_paths = [x for x in Path(class_folder).glob("*.jpg")]

            for img_path in tqdm(img_paths):
                if img_path.is_file():
                    img_pil = Image.open(img_path)
                    img_pil = ImageOps.exif_transpose(img_pil)

                    img_tensor = get_transforms(img_pil).unsqueeze(0).to(device)
                    img_tensor.requires_grad_()

                    heatmap = compute_gradcam(model, target_layer, img_tensor)
                    vis_heatmap(img_pil, heatmap, output_class_path / f"{img_path.stem}.png")


@hydra.main(version_base=None, config_path="../../", config_name="config")
def main(cfg: DictConfig) -> None:
    folder_to_run = Path(cfg.export.path_to_data)
    output_path = Path(cfg.export.vis_path)
    model = prepare_model(
        Path(cfg.train.path_to_save) / "model.pt",
        num_labels,
        cfg.train.device,
    ).to("cpu")
    target_layer = model.features[8][0]

    vis_gradcam(model, folder_to_run, output_path, target_layer, "cpu")


if __name__ == "__main__":
    main()
