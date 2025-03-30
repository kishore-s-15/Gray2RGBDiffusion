import torch

def save_model(model, path="model.pth"):
    """
    Save a PyTorch model's state dictionary to a file.
    
    This function saves only the model's parameters (state_dict) rather than
    the entire model, which is the recommended way to save PyTorch models.
    
    Args:
        model (torch.nn.Module): The PyTorch model to save.
        path (str, optional): File path where the model will be saved. 
                             Defaults to "model.pth".
                             
    Returns:
        None
    """
    torch.save(model.state_dict(), path)


def load_model(model, path, device):
    """
    Load a previously saved state dictionary into a PyTorch model.
    
    This function loads the model parameters from a file, transfers the model
    to the specified device, and sets it to evaluation mode.
    
    Args:
        model (torch.nn.Module): The model architecture to load parameters into.
        path (str): File path where the model state dictionary is stored.
        device (str or torch.device): Device to load the model on (e.g., 'cuda', 'cpu').
        
    Returns:
        torch.nn.Module: The loaded model, placed on the specified device
                        and set to evaluation mode.
    """
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model loaded from {path}")
    return model