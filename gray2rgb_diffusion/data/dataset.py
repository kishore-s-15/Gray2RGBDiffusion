from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class CIFAR10GrayToColor(Dataset):
    """
    A PyTorch Dataset that converts CIFAR10 RGB images to grayscale.
    
    This dataset pairs grayscale versions of CIFAR10 images with their original
    color versions, enabling tasks such as image colorization.
    
    Attributes:
        dataset (torchvision.datasets.CIFAR10): The original CIFAR10 dataset.
        transform (callable, optional): Optional transform to be applied to both
            grayscale and color images.
    """
    
    def __init__(self, root, train=True, transform=None):
        """
        Initialize the CIFAR10GrayToColor dataset.
        
        Args:
            root (str): Root directory where the CIFAR10 dataset is stored or 
                        will be downloaded to.
            train (bool, optional): If True, creates dataset from training set, 
                                   otherwise from test set. Default is True.
            transform (callable, optional): A function/transform that takes in a 
                                           PIL image and returns a transformed version.
        """
        self.dataset = CIFAR10(root=root, train=train, download=True)
        self.transform = transform
        
    def __len__(self):
        """
        Return the number of images in the dataset.
        
        Returns:
            int: The length of the dataset.
        """
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """
        Get the grayscale and color versions of an image at the specified index.
        
        Args:
            idx (int): Index of the image to retrieve.
            
        Returns:
            tuple: (grayscale_image, color_image) where both images have been
                  processed by the same transform if one was specified.
        """
        image, _ = self.dataset[idx]
        gray_image = Image.fromarray(np.array(image)).convert('L')
        
        if self.transform:
            gray_image = self.transform(gray_image)
            image = self.transform(image)
            
        return gray_image, image