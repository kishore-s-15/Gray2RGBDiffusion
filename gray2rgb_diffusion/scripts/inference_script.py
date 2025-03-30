import torch
import matplotlib.pyplot as plt
from diffusers import DDPMScheduler
from torch.utils.data import DataLoader, Subset
from gray2rgb_diffusion.configs.config import config
from gray2rgb_diffusion.data.dataset import CIFAR10GrayToColor
from gray2rgb_diffusion.inference.infer import infer
from gray2rgb_diffusion.models.model import get_model
from gray2rgb_diffusion.models.save_model import load_model
from gray2rgb_diffusion.utils.transforms import get_transforms

"""
CIFAR-10 Image Colorization Script using Diffusion Models

This script evaluates a pretrained diffusion model for colorizing grayscale CIFAR-10 images.
It loads a trained model, runs inference on test data, and visualizes the results.

The process involves:
1. Loading the test dataset and relevant transforms
2. Initializing the UNet model and diffusion scheduler
3. Loading a pretrained model checkpoint
4. Running inference to generate colorized images
5. Visualizing and comparing results with original images
"""

# Get transforms
transform, reverse_transform, eval_transform = get_transforms()

# Load test dataset
test_dataset = CIFAR10GrayToColor(root="./data", train=False, transform=transform)
if config["do_subset_test"]:
    """
    Optionally use a subset of test data for faster evaluation.
    This is controlled by the 'do_subset_test' and 'test_subset_size' config parameters.
    """
    test_dataset = Subset(test_dataset, range(config["test_subset_size"]))
test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=True)

# Initialize model and scheduler
model = get_model(config)
scheduler = DDPMScheduler(num_train_timesteps=config["num_train_timesteps"], beta_schedule="linear")

# Load pre-trained model
model = load_model(model, f"{config['save_dir']}/colorization_epoch.pth", config["device"])

# Perform inference
samples, gray_images, original_images = infer(model, test_loader, scheduler, config)
eval_samples = torch.stack([eval_transform(img) for img in samples])
eval_original_images = torch.stack([eval_transform(img) for img in original_images])

# Visualize results
def plot_results(samples, gray_images, original_images, reverse_transform):
    """
    Plot grayscale, predicted colorized, and original color images side by side.
    
    This function creates a visualization grid where each column contains:
    - Top row: The grayscale input image
    - Middle row: The model's colorized prediction
    - Bottom row: The original color image (ground truth)
    
    Args:
        samples (list of tensors): Predicted colorized images from the model.
        gray_images (list of tensors): Grayscale input images.
        original_images (list of tensors): Original color images (ground truth).
        reverse_transform (callable): Transform to convert tensors back to displayable image format.
        
    Returns:
        None: This function displays the plot but does not return any values.
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