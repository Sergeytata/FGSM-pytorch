import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from typing import Tuple

class FGSM:
    def __init__(self, model: nn.Module, epsilon: float = 0.007, device=torch.device("cpu")):
        self.model = model
        self.epsilon = epsilon
        self.criterion = nn.CrossEntropyLoss()
        self.device = device
    
    def generate(self, image: torch.Tensor, target: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Enable gradients for input image
        image.requires_grad = True
        
        # forward pass
        output = self.model(image)
        
        # Calculate loss
        loss = self.criterion(output, torch.tensor([target]).to(device))
        
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

def create_adversarial_transform(model: nn.Module, epsilon: float = 0.007):
    fgsm = FGSM(model, epsilon)
    
    def transform(image: torch.Tensor, target: int) -> torch.Tensor:
        adversarial_image, _ = fgsm.generate(image.unsqueeze(0), target)
        return adversarial_image.squeeze(0)
    
    return transform


class AdversarialDataset(Dataset):
    def __init__(self, base_dataset, model, epsilon=0.007):
        self.base_dataset = base_dataset
        self.fgsm = FGSM(model, epsilon)
        
    def __len__(self):
        return len(self.base_dataset)
        
    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        adversarial_image, _ = self.fgsm.generate(image.unsqueeze(0), label)
        return adversarial_image.squeeze(0), label


if __name__ == "__main__":
    from model_inference import setup_model, ImageNetValidationDataset

    # Setup model
    device = torch.device("cpu")
    model = setup_model(device)

    # Setup validation data
    import os
    IMAGENET_1K_VAL_DIR = os.environ["IMAGENET_1K_VAL_DIR"]

    # Set up dataset and loader
    batch_size = 1

    # Create adversarial transform
    epsilon = 0.05

    adversarial_transform = create_adversarial_transform(model, epsilon)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_dataset = ImageNetValidationDataset(
        IMAGENET_1K_VAL_DIR, 
        transform=transform
        )

    # Create adversarial dataset
    adversarial_dataset = AdversarialDataset(val_dataset, model, epsilon)
    
    
    # Predict on single image from dataset
    
    # Visualize image and adversarial image
    import matplotlib.pyplot as plt
    # img_idx = 5000 # cannon - bright light image
    img_idx = 30000 # turtle

    image, label = val_dataset[img_idx]
    adversarial_image, _ = adversarial_dataset[img_idx]
    # reverse normalization to visualise images for perceptual difference
    image_viz = image * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    adversarial_image_viz = adversarial_image * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)

    image_viz = torch.clamp(image_viz, 0, 1).permute(1, 2, 0)
    adversarial_image_viz = torch.clamp(adversarial_image_viz, 0, 1).permute(1, 2, 0)



    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image_viz)
    axes[0].set_title("Original Image")
    axes[1].imshow(adversarial_image_viz)
    axes[1].set_title("Adversarial Image")
    plt.show()


    # Predict on image and adversarial image against label
    with torch.no_grad():
        outputs = model(image.unsqueeze(0))
        _, predicted = torch.max(outputs, 1)   
        print(f"Original Image Prediction: {predicted.item()}") 
    
        outputs = model(adversarial_image.unsqueeze(0))
        _, predicted = torch.max(outputs, 1)
        print(f"Adversarial Image Prediction: {predicted.item()}")

        print(f"Original Image Label: {label}")

