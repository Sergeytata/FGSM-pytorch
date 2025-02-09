# ML Challenge

## Setup

```bash
# Clone the repository
git clone <repository-url>

# Install dependencies
pip install -r requirements.txt
```

## Project Structure
adverarial_transform.py contains adversarial transformation.
model_inference.py contains imagenet benchmarking.

## Results
| Exp. \ Model   | ResNet18 | ResNet50 | ConvNeXt Tiny |
|----------------|----------|----------|---------------|
| acc@1          | 69.76%   | -----    | -----         |
| acc@5          | 89.08%   | -----    | -----         |
| acc@1 + FGSN   |  1.24%   | -----    | -----         |
| acc@5 + FGSN   | 16.54%   | -----    | -----         |
