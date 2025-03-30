import torch
import torch.nn.functional as F
from torchvision.models import inception_v3
from scipy.linalg import sqrtm
import numpy as np

def calculate_fid(mu1, sigma1, mu2, sigma2):
    """
    Compute the Fréchet Inception Distance (FID) between two sets of features.
    
    The FID measures the similarity between two distributions of images.
    Lower FID values indicate more similar distributions, suggesting the
    generated images are more similar to real images.
    
    Args:
        mu1 (numpy.ndarray): Mean of the first feature distribution.
        sigma1 (numpy.ndarray): Covariance matrix of the first feature distribution.
        mu2 (numpy.ndarray): Mean of the second feature distribution.
        sigma2 (numpy.ndarray): Covariance matrix of the second feature distribution.
        
    Returns:
        float: The calculated FID score.
    """
    diff = mu1 - mu2
    covmean = sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid

def get_inception_activations(images, model, device):
    """
    Extract InceptionV3 feature activations for given images.
    
    This function processes a batch of images through the InceptionV3 model
    and returns the activations from the final layer.
    
    Args:
        images (torch.Tensor): Batch of images to extract features from.
        model (torch.nn.Module): Pre-trained InceptionV3 model with modified
                                final layer to return features.
        device (torch.device): Device to run the model on (CPU or GPU).
        
    Returns:
        numpy.ndarray: Extracted feature activations of shape (batch_size, feature_dim).
    """
    model.eval()
    images = images.to(device)
    images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
    with torch.no_grad():
        activations = model(images)
    return activations.cpu().numpy()

def inception_score(images, model, device, splits=10):
    """
    Compute the Inception Score (IS) for a batch of generated images.
    
    The Inception Score measures both the quality and diversity of generated images.
    Higher scores indicate better quality and diversity.
    
    Args:
        images (torch.Tensor): Batch of images to evaluate.
        model (torch.nn.Module): Pre-trained InceptionV3 model.
        device (torch.device): Device to run the model on (CPU or GPU).
        splits (int, optional): Number of splits to use when computing score.
                               Default is 10.
        
    Returns:
        tuple: A tuple containing:
            - float: Mean Inception Score across all splits.
            - float: Standard deviation of Inception Scores across all splits.
    """
    model.eval()
    images = images.to(device)
    images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
    with torch.no_grad():
        preds = model(images)
    preds = F.softmax(preds, dim=1).cpu().numpy()
    
    split_scores = []
    for k in np.array_split(preds, splits):
        kl_div = k * (np.log(k) - np.log(np.mean(k, axis=0, keepdims=True)))
        split_scores.append(np.exp(np.mean(np.sum(kl_div, axis=1))))
    
    return np.mean(split_scores), np.std(split_scores)

def compute_fid_and_is(original_images, generated_images, device='cuda'):
    """
    Compute both FID and Inception Score given batches of original and generated images.
    
    This function serves as a wrapper to calculate both metrics in a single pass
    through the InceptionV3 model.
    
    Args:
        original_images (torch.Tensor): Batch of real/original images.
        generated_images (torch.Tensor): Batch of generated/synthetic images.
        device (str, optional): Device to run computation on. Default is 'cuda'.
                               Falls back to 'cpu' if CUDA is not available.
    
    Returns:
        tuple: A tuple containing:
            - float: Fréchet Inception Distance (FID) score.
            - tuple: Inception Score (IS) as (mean, std).
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Load and modify InceptionV3 model
    inception = inception_v3(pretrained=True, transform_input=False).to(device)
    inception.fc = torch.nn.Identity()
    
    # Extract features and compute statistics
    orig_acts = get_inception_activations(original_images, inception, device)
    gen_acts = get_inception_activations(generated_images, inception, device)
    
    mu1, sigma1 = orig_acts.mean(axis=0), np.cov(orig_acts, rowvar=False)
    mu2, sigma2 = gen_acts.mean(axis=0), np.cov(gen_acts, rowvar=False)
    
    fid_score = calculate_fid(mu1, sigma1, mu2, sigma2)
    is_score, is_std = inception_score(generated_images, inception, device)
    
    return fid_score, (is_score, is_std)