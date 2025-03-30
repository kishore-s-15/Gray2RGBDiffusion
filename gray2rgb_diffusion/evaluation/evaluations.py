import torch
import torch.nn.functional as F
from torchvision.models import inception_v3
from scipy.linalg import sqrtm
import numpy as np

def calculate_fid(mu1, sigma1, mu2, sigma2):
    """Compute the Fréchet Inception Distance (FID)."""
    diff = mu1 - mu2
    covmean = sqrtm(sigma1 @ sigma2)
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid

def get_inception_activations(images, model, device):
    """Extract InceptionV3 activations for given images."""
    model.eval()
    images = torch.stack(images).to(device)
    images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
    with torch.no_grad():
        activations = model(images)
    return activations.cpu().numpy()

def inception_score(images, model, device, splits=10):
    """Compute the Inception Score (IS)."""
    model.eval()
    images = torch.stack(images).to(device)
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
    """Compute both FID and IS scores given lists of original and generated images."""
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    inception = inception_v3(pretrained=True, transform_input=False).to(device)
    inception.fc = torch.nn.Identity()
    
    orig_acts = get_inception_activations(original_images, inception, device)
    gen_acts = get_inception_activations(generated_images, inception, device)
    
    mu1, sigma1 = orig_acts.mean(axis=0), np.cov(orig_acts, rowvar=False)
    mu2, sigma2 = gen_acts.mean(axis=0), np.cov(gen_acts, rowvar=False)
    
    fid_score = calculate_fid(mu1, sigma1, mu2, sigma2)
    is_score, is_std = inception_score(generated_images, inception, device)
    
    return fid_score, (is_score, is_std)