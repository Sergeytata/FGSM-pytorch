import torch
import torch.nn as nn
from typing import Tuple

class FGSM:
    def __init__(self, model: nn.Module, epsilon: float = 0.05):
        self.model = model
        self.device = model.parameters().__next__().device
        self.epsilon = epsilon
        self.criterion = nn.CrossEntropyLoss()
    
    def generate(self, image: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Enable gradients for input image
        image.requires_grad = True
        
        # forward pass
        output = self.model(image.to(self.device))
        
        # Calculate loss
        loss = self.criterion(output, target.to(self.device))
        
        # backward pass
        loss.backward()
        
        # Generate perturbation
        grads_sign = torch.sign(image.grad.data)
        grads_sign.requires_grad = False
        perturbation = self.epsilon * grads_sign
        
        # Disable gradients
        image.requires_grad = False
        # Create adversarial image
        adversarial_image = image + perturbation
        
        return adversarial_image, perturbation
