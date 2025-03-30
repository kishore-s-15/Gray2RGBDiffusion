from gray2rgb_diffusion.data.dataset import CIFAR10GrayToColor
from gray2rgb_diffusion.models.model import get_model
from gray2rgb_diffusion.training.train import train_model
from gray2rgb_diffusion.configs.config import config
from gray2rgb_diffusion.utils.transforms import get_transforms
from torch.utils.data import DataLoader, Subset

"""
Image Colorization Training Script

This script sets up and initiates the training process for a diffusion-based
grayscale-to-RGB colorization model on the CIFAR-10 dataset.

The workflow includes:
1. Loading appropriate image transforms
2. Setting up the training dataset with optional subset selection
3. Initializing the UNet model architecture
4. Running the training process with specified configuration
"""

# Load transforms for data preprocessing
transform, _, _ = get_transforms()

# Set up training dataset
train_dataset = CIFAR10GrayToColor(root="./data", train=True, transform=transform)

# Optionally use a subset of training data
if config["do_subset_train"]:
    """
    If configured, use only a subset of the training data.
    This is useful for debugging or faster iteration during development.
    The subset size is controlled by the 'train_subset_size' config parameter.
    """
    train_dataset = Subset(train_dataset, range(config["train_subset_size"]))

# Create data loader with batching and shuffling
train_loader = DataLoader(
    train_dataset, 
    batch_size=config["batch_size"], 
    shuffle=True
)

# Initialize the UNet model based on configuration
model = get_model(config)

# Run the training process
train_model(model, train_loader, config)