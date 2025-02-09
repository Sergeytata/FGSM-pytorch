import torch
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt

from adversarial_transform import FGSM

def main():
    # Setup model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    weights = ResNet18_Weights(ResNet18_Weights.DEFAULT)
    model = resnet18(weights=weights)
    model.eval()
    model = model.to(device)

    # FGSM
    epsilon = 0.05
    fgsm = FGSM(model, epsilon)


    # load data/panda.JPEG
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image = Image.open("data/panda.JPEG").convert('RGB')
    label = 388 # panda label
    image = transform(image)

    # Create adversarial image
    adversarial_image, _ = fgsm.generate(image.unsqueeze(0), torch.tensor([label]))

    # squeeze batch dimension
    image = image.squeeze(0)
    adversarial_image = adversarial_image.squeeze(0)


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

    os.makedirs("output", exist_ok=True)
    fig.savefig("output/result.png")

    return

if __name__ == "__main__":
    main()