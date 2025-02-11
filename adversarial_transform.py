import torch
import torch.nn as nn
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class FPGSMConfig:
    epsilon: float = 0.01
    steps: int = 5
    beta: float = 0.5

class FGSM:
    def __init__(self, model: nn.Module, config: FPGSMConfig):
        self.model = model
        self.device = model.parameters().__next__().device
        self.epsilon = config.epsilon
        self.steps = config.steps
        self.beta = config.beta
        self.criterion = nn.CrossEntropyLoss()
    
    def generate(self, image: torch.Tensor, target: torch.Tensor, targeted = False) -> Tuple[torch.Tensor, torch.Tensor]:
        epsilon = self.epsilon
        perturbation = None
        for _ in range(self.steps):
            # enable gradients
            image.requires_grad = True
            # zero gradients
            self.model.zero_grad()

            # forward pass
            output = self.model(image.to(self.device))
            
            # Calculate loss
            loss = self.criterion(output, target.to(self.device))
            
            # backward pass
            loss.backward()
            
            # Generate perturbation
            grads_sign = torch.sign(image.grad.data)
            
            # Disable gradients
            grads_sign.requires_grad = False
            image.requires_grad = False
            
            # Pertrubate image
            perturbation = epsilon * grads_sign
            image = image - perturbation if targeted else image + perturbation

            # Decrease epsilon
            epsilon = epsilon * self.beta 
        
        return image, perturbation
