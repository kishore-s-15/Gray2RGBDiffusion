import torchvision.transforms as transforms

def get_transforms():
    """
    Create and return image transformation pipelines for the diffusion colorization model.
    
    This function defines three different transform pipelines:
    1. A normalization transform for model input (-1 to 1 range)
    2. A reverse transform to convert model outputs back to viewable images
    3. An evaluation transform for processing model outputs for metrics calculation
    
    Returns:
        tuple: A tuple containing three transform pipelines:
            - transform (torchvision.transforms.Compose): Input normalization transform
              that converts images to tensors and scales values to [-1, 1] range.
            - reverse_transform (torchvision.transforms.Compose): Output denormalization 
              transform that converts model outputs back to displayable RGB images.
            - eval_transform (torchvision.transforms.Compose): Evaluation transform
              that normalizes outputs to [0, 1] range for metric calculation.
              
    Notes:
        - The input transform scales pixel values from [0, 1] to [-1, 1] as expected
          by many diffusion models.
        - The reverse transform converts tensors back to a format suitable for 
          displaying or saving as images (byte format with values in [0, 255]).
        - The eval transform performs normalization for evaluation metrics but
          doesn't convert to byte format or change the tensor dimensions.
    """
    # Input transform: Convert to tensor and normalize to [-1, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),            # Convert PIL image to tensor, scales to [0, 1]
        transforms.Lambda(lambda x: (x * 2) - 1),  # Scale from [0, 1] to [-1, 1]
    ])
    
    # Reverse transform: Convert model output to displayable image format
    reverse_transform = transforms.Compose([
        transforms.Lambda(lambda x: x.clamp(-1, 1)),       # Clamp values to valid range
        transforms.Lambda(lambda x: (x + 1) / 2),          # Scale from [-1, 1] to [0, 1]
        transforms.Lambda(lambda x: x * 255),              # Scale to [0, 255]
        transforms.Lambda(lambda x: x.permute(1, 2, 0)),   # Change from CxHxW to HxWxC
        transforms.Lambda(lambda x: x.byte()),             # Convert to byte tensor
    ])
    
    # Evaluation transform: Normalize outputs for metric calculation
    eval_transform = transforms.Compose([
        transforms.Lambda(lambda x: x.clamp(-1, 1)),       # Clamp values to valid range
        transforms.Lambda(lambda x: (x + 1) / 2)           # Scale from [-1, 1] to [0, 1]
    ])
    
    return transform, reverse_transform, eval_transform