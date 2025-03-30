import numpy as np
import matplotlib.pyplot as plt
from diffusers import DDPMScheduler
from torch.utils.data import DataLoader, Subset

from gray2rgb_diffusion.configs.config import config
from gray2rgb_diffusion.data.dataset import CIFAR10GrayToColor
from gray2rgb_diffusion.inference.infer import infer
from gray2rgb_diffusion.models.model import get_model
from gray2rgb_diffusion.models.save_model import load_model
from gray2rgb_diffusion.utils.transforms import get_transforms

# Get transforms
transform, reverse_transform = get_transforms()

# Load test dataset
test_dataset = CIFAR10GrayToColor(root="./data", train=False, transform=transform)

if config["do_subset_test"]:
    print("Here")
    test_dataset = Subset(test_dataset, range(config["test_subset_size"]))

test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=True)

# Initialize model and scheduler
model = get_model(config)
scheduler = DDPMScheduler(num_train_timesteps=config["num_train_timesteps"], beta_schedule="linear")

# Load pre-trained model
model = load_model(model, f"{config['save_dir']}/colorization_epoch.pth", config["device"])

# Perform inference
samples, gray_images, original_images = infer(model, test_loader, scheduler, config)

print("Samples:")
print(f"Min: {np.min(samples)}, Max: {np.max(samples)}")

print("Original Images:")
print(f"Min: {np.min(original_images)}, Max: {np.max(original_images)}")

# Visualize results
def plot_results(samples, gray_images, original_images, reverse_transform):
    """
    Plot grayscale, predicted colorized, and original color images horizontally.

    Args:
    - samples (list of tensors): Predicted colorized images.
    - gray_images (list of tensors): Grayscale input images.
    - original_images (list of tensors): Original color images.
    - reverse_transform (callable): Transform to convert tensors back to image format.
    """
    num_images = len(samples)
    fig, axs = plt.subplots(3, num_images, figsize=(15, 5))  # Arrange images horizontally

    for idx in range(num_images):
        # Grayscale image
        gray_img = reverse_transform(gray_images[idx]).cpu().numpy()
        axs[0, idx].imshow(gray_img, cmap='gray')
        axs[0, idx].set_title("Grayscale Image")
        axs[0, idx].axis('off')

        # Predicted colorized image
        pred_img = reverse_transform(samples[idx]).cpu().numpy()
        axs[1, idx].imshow(pred_img)
        axs[1, idx].set_title("Predicted Colorized Image")
        axs[1, idx].axis('off')

        # Original color image
        orig_img = reverse_transform(original_images[idx]).cpu().numpy()
        axs[2, idx].imshow(orig_img)
        axs[2, idx].set_title("Original Image")
        axs[2, idx].axis('off')

    plt.tight_layout()
    plt.show()

# Example Usage
plot_results(samples[:8], gray_images[:8], original_images[:8], reverse_transform)
