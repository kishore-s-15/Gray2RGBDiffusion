import torch
from tqdm import tqdm

def infer(model, test_loader, scheduler, config):
    """
    Perform inference with a diffusion model to generate colorized images from grayscale inputs.
    
    This function runs the reverse diffusion process starting from random noise,
    conditioning on grayscale images to produce colorized versions. The process
    follows the standard diffusion model sampling procedure, iteratively denoising
    the images across multiple timesteps.
    
    Args:
        model (torch.nn.Module): The trained diffusion model to use for inference.
        test_loader (torch.utils.data.DataLoader): DataLoader containing test data
            pairs of (grayscale_image, original_color_image).
        scheduler (DiffusionScheduler): The noise scheduler that controls the 
            diffusion process parameters.
        config (dict): Configuration dictionary containing inference settings.
            Required keys:
                - device (str or torch.device): Device to run inference on.
                - image_size (int): Size of the input/output images.
    
    Returns:
        tuple: A tuple containing:
            - list: Generated/colorized image samples (torch.Tensor objects on CPU).
            - list: Input grayscale images (torch.Tensor objects on CPU).
            - list: Original color images for comparison (torch.Tensor objects on CPU).
    """
    model.eval()
    gray_images, original_images, samples = [], [], []
    device = config["device"]
    
    with torch.no_grad():
        for gray, original in tqdm(test_loader, desc="Inference Progress"):
            gray, original = gray.to(device), original.to(device)
            
            # Start from pure noise
            batch_size = gray.size(0)
            noisy_images = torch.randn(
                (batch_size, 3, config["image_size"], config["image_size"])
            ).to(device)
            
            # Perform denoising with tqdm for reverse diffusion steps
            for t in tqdm(
                range(scheduler.config.num_train_timesteps - 1, -1, -1),
                desc="Reverse Diffusion",
                total=scheduler.config.num_train_timesteps
            ):
                timesteps = torch.full((batch_size,), t, device=device, dtype=torch.long)
                x_t = torch.cat((noisy_images, gray), dim=1)
                noise_pred = model(x_t, timesteps).sample
                noisy_images = scheduler.step(noise_pred, t, noisy_images).prev_sample
            
            samples.extend(noisy_images.cpu())  # Move to CPU for visualization/storage
            gray_images.extend(gray.cpu())
            original_images.extend(original.cpu())
    
    return samples, gray_images, original_images