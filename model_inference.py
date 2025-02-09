import torch
import torchvision.transforms as transforms
from torchvision.models import (
    resnet18, ResNet18_Weights,
    resnet50, ResNet50_Weights,
    convnext_tiny, ConvNeXt_Tiny_Weights,
    convnext_small, ConvNeXt_Small_Weights
)
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from PIL import Image
from typing import Tuple
import json
from tqdm import tqdm
import os

from adversarial_transform import FGSM


class ImageNetValidationDataset(Dataset):
    def __init__(self, root_dir: Path, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.images = list(self.root_dir.glob('**/*.JPEG'))
        
        # Load class mapping
        # obtained from here - https://github.com/raghakot/keras-vis/blob/master/resources/imagenet_class_index.json
        with open('imagenet_class_index.json', 'r') as f:
            class_idx = json.load(f)
            # Convert from {idx: [wnid, label]} to {wnid: idx}
            self.class_to_idx = {v[0]: int(k) for k, v in class_idx.items()}
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, int]:
        img_path = self.images[idx]
        # Get class from parent directory name
        class_dir = img_path.parent.name
        label = self.class_to_idx[class_dir]
        
        # Load and transform image
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
            
        return image, label

def setup_model(model_name="resnet18", device=torch.device('cpu')):
    model_configs = {
        'resnet18': (resnet18, ResNet18_Weights.DEFAULT),
        'resnet50': (resnet50, ResNet50_Weights.DEFAULT),
        'convnext_tiny': (convnext_tiny, ConvNeXt_Tiny_Weights.DEFAULT),
        'convnext_small': (convnext_small, ConvNeXt_Small_Weights.DEFAULT)
    }
    
    if model_name not in model_configs:
        raise ValueError(f"Model {model_name} not supported. Choose from {list(model_configs.keys())}")
    
    model_fn, weights = model_configs[model_name]
    model = model_fn(weights=weights)
    model.eval()
    return model.to(device)

def setup_validation_data(val_dir, batch_size=32):
    # Standard ImageNet transforms
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Load ImageNet validation dataset with torchvision
    # NOTE: This requires .tar file downloaded from ImageNet website.
    #       This requires an approval I did not have time to wait for.
    # val_dataset = torchvision.datasets.ImageNet(
    #     root=val_dir,
    #     split='val',
    #     transform=transform,
    # )

    # Use custom dataset with validation set downloaded from Kaggle - https://www.kaggle.com/datasets/titericz/imagenet1k-val?resource=download
    val_dataset = ImageNetValidationDataset(
        val_dir, 
        transform=transform
        )
        
    # Create DataLoader
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    return val_dataset, val_loader


def predict(model, val_loader):
    device = model.parameters().__next__().device
    predictions_top1 = []
    predictions_top5 = []
    true_labels = []
    
    for images, labels in tqdm(val_loader):
        with torch.no_grad():
            images = images.to(device)
            
            outputs = model(images)
            _, predicted_top1 = torch.max(outputs, 1)
            _, predicted_top5 = torch.topk(outputs, k=5, dim=1)
            
            predictions_top1.extend(predicted_top1.cpu().numpy())
            predictions_top5.extend(predicted_top5.cpu().numpy())
            true_labels.extend(labels.numpy())
    
    return predictions_top1, predictions_top5, true_labels

def predict_adversarial(model, val_loader, fgsm):
    device = model.parameters().__next__().device
    fgsm_device = fgsm.model.parameters().__next__().device
    if device != fgsm_device:
        print("[Warning]: Model and FGSM model are not on the same device!")
    predictions_top1 = []
    predictions_top5 = []
    true_labels = []
    
    for images, labels in tqdm(val_loader):
        images = images.to(fgsm_device)
        labels = labels.to(fgsm_device)
        adversarail_images, _ = fgsm.generate(images, labels)
        with torch.no_grad():
            adversarail_images = adversarail_images.to(device)

            outputs = model(adversarail_images)
            _, predicted_top1 = torch.max(outputs, 1)
            _, predicted_top5 = torch.topk(outputs, k=5, dim=1)
            
            predictions_top1.extend(predicted_top1.cpu().numpy())
            predictions_top5.extend(predicted_top5.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    return predictions_top1, predictions_top5, true_labels

def calculate_accuracy(predictions_top1, predictions_top5, true_labels):
    # acc@1
    acc_top1 = sum(p == t for p, t in zip(predictions_top1, true_labels)) / len(true_labels)
    # acc@5
    acc_top5 = sum(t in p for p, t in zip(predictions_top5, true_labels)) / len(true_labels)
    return acc_top1, acc_top5
    

def main():
    # Set up model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = setup_model("resnet18", device)
    
    # Set ImageNet validation directory
    IMAGENET_1K_VAL_DIR = os.environ["IMAGENET_1K_VAL_DIR"]

    # Set up dataset and dataloader
    batch_size = 32
    val_dataset, val_loader = setup_validation_data(IMAGENET_1K_VAL_DIR, batch_size=batch_size) # Update environment variable with path to ImageNet validation data

    # Verify dataset size
    assert len(val_dataset) == 50000, "Validation dataset should have 50,000 images"

    # FGSM
    model_fgsm = setup_model("resnet18", device)
    fgsm = FGSM(model_fgsm, epsilon=0.05)

    # Benchmark original dataset
    predictions_top1, predictions_top5, true_labels = predict(model, val_loader)
    accuracy_top1, accuracy_top5 = calculate_accuracy(predictions_top1, predictions_top5, true_labels)

    # Benchmark adversarial dataset
    predictions_top1_adver, predictions_top5_adver, true_labels_adver = predict_adversarial(model, val_loader, fgsm)
    accuracy_top1_adver, accuracy_top5_adver = calculate_accuracy(predictions_top1_adver, predictions_top5_adver, true_labels_adver)
    
    print(f"Original acc@1   : {accuracy_top1:.4f}")
    print(f"Original acc@5   : {accuracy_top5:.4f}")
    print(f"Adversarial acc@1: {accuracy_top1_adver:.4f}")
    print(f"Adversarial acc@5: {accuracy_top5_adver:.4f}")


if __name__ == "__main__":
    main()
