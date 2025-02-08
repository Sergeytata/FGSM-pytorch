import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from PIL import Image
from typing import Tuple
import json
from tqdm import tqdm
import os

from adversarial_transform import AdversarialDataset


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

def setup_model(device=torch.device('cpu')):
    # Load pretrained ResNet18
    weights = ResNet18_Weights(ResNet18_Weights.DEFAULT)
    model = resnet18(weights=weights)
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


def setup_validation_adversarial_data(val_dataset, model, epsilon=0.05, batch_size=32):
    # Create adversarial dataset
    adversarial_dataset = AdversarialDataset(
        val_dataset, 
        model, 
        epsilon)
        
    # Create DataLoader
    val_loader = DataLoader(
        adversarial_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    return adversarial_dataset, val_loader

def predict(model, val_loader):
    predictions = []
    true_labels = []
    
    for images, labels in tqdm(val_loader):
        with torch.no_grad():
            if torch.cuda.is_available():
                images = images.cuda()
            
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.numpy())
    
    return predictions, true_labels

def main():
    # Set up model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    model = setup_model(device)
    
    # Set ImageNet validation directory
    IMAGENET_1K_VAL_DIR = os.environ["IMAGENET_1K_VAL_DIR"]

    # Set up data loader
    batch_size = 32
    val_dataset, val_loader = setup_validation_data(IMAGENET_1K_VAL_DIR, batch_size=batch_size) # Update environment variable with path to ImageNet validation data


    # NOTE: Turns out we cannot use cuda with default torch.multiprocessing start.
    # mp.set_start_method('spawn', force=True) is required at the beginning of the script.
    # However, this spawns as many models as there are workers defined in DataLoader.
    model_cpu = setup_model(torch.device('cuda'))
    val_adver_dataset, val_adver_loader = setup_validation_adversarial_data(val_dataset, model_cpu, epsilon=0.05, batch_size=batch_size)
    print(val_adver_dataset.fgsm.device)
    # Verify dataset size
    assert len(val_dataset) == 50000, "Validation dataset should have 50,000 images"
    
    # Make predictions on original validation data
    # predictions, true_labels = predict(model, val_loader)

    # Make predictions on adversarial validation data
    predictions, true_labels = predict(model, val_adver_loader)
    
    # Calculate accuracy
    accuracy = sum(p == t for p, t in zip(predictions, true_labels)) / len(predictions)
    print(f"Validation Accuracy: {accuracy:.4f}")

    # resnet18 validation acc@1: 0.6976
    # [Verified] PyTorch states 69.758% - https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html

if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    main()
