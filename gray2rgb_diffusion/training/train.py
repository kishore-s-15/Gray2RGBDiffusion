import os
import torch
from diffusers import DDPMScheduler
from torch.nn import MSELoss
from torch.optim import AdamW
from tqdm import tqdm

def train_model(model, train_loader, config):
    """
    Train a diffusion model for grayscale to RGB image colorization.
    
    This function implements the full training loop for a diffusion-based colorization model.
    It sets up the diffusion scheduler, optimizer, and loss function, then runs the training
    process for the specified number of epochs. During training, the model learns to predict
    noise added to color images, conditioned on their grayscale versions.
    
    Args:
        model (torch.nn.Module): The UNet model to be trained.
        train_loader (torch.utils.data.DataLoader): DataLoader providing batches of
            (grayscale_image, color_image) pairs for training.
        config (dict): Configuration dictionary containing training parameters.
            Required keys:
                - num_train_timesteps (int): Number of timesteps in diffusion process.
                - beta_schedule (str): Schedule for noise variance (e.g., 'linear').
                - learning_rate (float): Learning rate for optimizer.
                - device (str or torch.device): Device to run training on.
                - num_epochs (int): Number of training epochs.
                - save_dir (str): Directory to save model checkpoints.
    
    Returns:
        None: The function saves the trained model to disk but does not return any values.
    
    Note:
        - The model is saved after each epoch, overwriting the previous checkpoint.
        - The training uses gradient clipping with a max norm of 1.0 for stability.
        - The grayscale image is concatenated with the noisy color image as conditioning.
    """
    # Prepare training components
    scheduler = DDPMScheduler(
        num_train_timesteps=config["num_train_timesteps"],
        beta_schedule=config["beta_schedule"]
    )
    optimizer = AdamW(model.parameters(), lr=config["learning_rate"])
    criterion = MSELoss()
    save_dir = config["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    
    # Training loop
    for epoch in range(config["num_epochs"]):
        model.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['num_epochs']}")
        losses = 0
        
        for batch in progress_bar:
            gray, color = batch
            gray, color = gray.to(config["device"]), color.to(config["device"])
            
            # Add noise
            noise = torch.randn_like(color)
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (gray.size(0),)).long().to(
                config["device"])
            noisy_images = scheduler.add_noise(color, noise, timesteps)
            
            # Concatenate grayscale and noisy color images
            x_t = torch.cat((noisy_images, gray), dim=1)
            
            # Predict noise
            noise_pred = model(x_t, timesteps).sample
            
            # Compute loss
            loss = criterion(noise_pred, noise)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            losses += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({"loss": losses})
        
        # Save model after each epoch
        torch.save(model.state_dict(), os.path.join(save_dir, f"colorization_epoch.pth"))
    
    print(f"Training completed. Models are saved in {save_dir}")