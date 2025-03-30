from diffusers import UNet2DModel

def get_model(config):
    """
    Create and initialize a UNet2D model for image generation or transformation tasks.
    
    This function instantiates a UNet2DModel from the diffusers library with 
    configuration parameters specified in the config dictionary. The model is
    suitable for diffusion-based image generation tasks such as image colorization.
    
    Args:
        config (dict): Configuration dictionary containing model parameters.
            Required keys:
                - image_size (int): Size of input/output images (assumed square).
                - block_out_channels (list or tuple): Number of output channels 
                  for each U-Net block.
                - device (str or torch.device): Device to load the model on.
    
    Returns:
        UNet2DModel: Initialized UNet2D model placed on the specified device.
        
    Notes:
        - The model is configured with 4 input channels (typically 3 for RGB noise 
          and 1 for grayscale conditioning) and 3 output channels (for RGB output).
        - The model uses 4 layers per block for increased capacity.
    """
    model = UNet2DModel(
        sample_size=config["image_size"],
        in_channels=4,  # 3 channels for noisy image + 1 for grayscale condition
        out_channels=3,  # Predict noise for RGB channels
        layers_per_block=4,
        block_out_channels=config["block_out_channels"],
    )
    return model.to(config["device"])