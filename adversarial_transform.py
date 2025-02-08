import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from typing import Tuple

class FGSM:
    def __init__(self, model: nn.Module, epsilon: float = 0.05):
        self.model = model
        self.device = model.parameters().__next__().device
        self.epsilon = epsilon
        self.criterion = nn.CrossEntropyLoss()
    
    def generate(self, image: torch.Tensor, target: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Enable gradients for input image
        image.requires_grad = True
        
        # forward pass
        output = self.model(image.to(self.device))
        
        # Calculate loss
        loss = self.criterion(output, torch.tensor([target]).to(self.device))
        
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


class AdversarialDataset(Dataset):
    def __init__(self, base_dataset, model, epsilon=0.05):
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
    # device = torch.device("cuda")
    model = setup_model(device)

    # Setup validation data
    import os
    IMAGENET_1K_VAL_DIR = os.environ["IMAGENET_1K_VAL_DIR"]

    # Set up dataset and loader
    batch_size = 1

    # Create adversarial transform
    epsilon = 0.05

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
    
    # Experiment with a single image
    
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

    # move images to the same device as model
    image = image.to(device)
    adversarial_image = adversarial_image.to(device)

    # Predict on image and adversarial image against label
    with torch.no_grad():
        outputs = model(image.unsqueeze(0))
        _, predicted = torch.max(outputs, 1)   
        print(f"Original Image Prediction: {predicted.item()}") 
    
        outputs = model(adversarial_image.unsqueeze(0))
        _, predicted = torch.max(outputs, 1)
        print(f"Adversarial Image Prediction: {predicted.item()}")

        print(f"Original Image Label: {label}")

