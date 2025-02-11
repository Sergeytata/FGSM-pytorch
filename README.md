# Fast Gradient Sign Method (FGSM) Adversarial Transformation
This project implement Fast Gradient Sign Method (FGSM) adversarial transformation using PyTorch. FGSM is a simple yet effective method to generate adversarial examples. The idea is to perturb the input image in the direction of the gradient of the loss with respect to the input image. The perturbation is calculated as:

```math
x_{adv} = x + \epsilon * sign(\nabla_x J(\theta, x, y))
```

where:
- $x$: input image
- $x_{adv}$: adversarial image
- $\epsilon$: perturbation magnitude
- $J$: loss function
- $\theta$: model parameters
- $y$: target label

The project also includes a benchmarking of the adversarial transformation on ImageNet-1k-val dataset using ResNet18, ResNet50, and ConvNeXt Tiny models.


## Setup

```bash
# Clone the repository
git clone https://github.com/Sergeytata/FGSM-pytorch

# Install dependencies
pip install -r requirements.txt
```

## Usage
The project comes with two examples to show how to use the adversarial transformation.

single_image_example.py - adversarial transformation of a single image. A good starting point to understand how the adversarial transformation works and to evaluate the perceptual quality of the transformed image.
```bash
# single image is a self-contained example and does not require any additional setup.
python single_image_example.py
```


model_inference.py - adversarial transformation of the ImageNet-1k-val dataset. This example illustrates how to integrate the adversarial transformation into the evaluation pipeline.
```bash
# model_inference.py requires the ImageNet-1k-val dataset to be downloaded and extracted.

# Set IMAGENET_1K_VAL_DIR environment variable to the val directory of your ImageNet dataset.
export IMAGENET_1K_VAL_DIR=%/path/to/imagenet%/val

python model_inference.py
```


### Integration
The project is designed as an integration with evaluation or training pipelines. adverarial_transform.py contains all the necessary classes to enable adversarial transformation of an image. 

```python
# %Your project imports%
from adversarial_transform import FGSM, FPGSMConfig


# % Your project code%

# Create FGSM object
adv_target = 301 # attack target
config = FPGSMConfig(epsilon=0.01, steps=7, beta=0.5)
fgsm = FGSM(model_fgsm, config=config)


# Training or evaluation pipeline
fgsm_device = fgsm.model.parameters().__next__().device

# Training/Evaluation loop example
for images, labels in tqdm(val_loader):
    images = images.to(fgsm_device)
    adv_labels = torch.full_like(labels, adv_target).to(fgsm_device)
    adv_images, _ = fgsm.generate(images, adv_labels)

    # %Your training loop in here%


    # fgsm requires grads for perturbation calculation
    # torch.no_grad() is used after adversarial transformation.
    with torch.no_grad():
        # %Your evaluation loop in here%

```


## Project Structure
adverarial_transform.py - adversarial transformation.
single_image_example.py - single image adversarial transformation.
model_inference.py - ImageNet-1k-val benchmarking.

## Results

I use resnet18, resnet50, and ConvNeXt Tiny models to benchmark untargeted and targeted adversarial attacks on ImageNet-1k-val dataset.

### Untargeted
steps = 1 \
beta = 1.0 \
epsilon = 0.05

Additionally, I explore extrapolation of resnet18 as an FGSM model to measure the effectiveness of the method on other unknown models from the same family and not.
The results are shown in the table below:

|      Experiment \ Model      | ResNet18 | ResNet50 | ConvNeXt Tiny |
|------------------------------|----------|----------|---------------|
| acc@1                        |  69.76%  |  80.34%  |    82.13%     |
| acc@5                        |  89.08%  |  95.13%  |    95.95%     |
| acc@1 + FGSM (resnet18)      |   1.24%  |  68.87%  |    73.49%     |
| acc@5 + FGSM (resnet18)      |  16.54%  |  90.45%  |    92.44%     |
| acc@1 + FGSM (convnext_tiny) |  ------  |  ------  |    34.07%     |
| acc@5 + FGSM (convnext_tiny) |  ------  |  ------  |    57.69%     |

### Targeted
Targeted attack uses smaller epsilon value to preserve image quality, presumably because of higher gradients calculated with respect to the target from a different class. Additionally, I explore the effectiveness of multi-step perturbations to increase the effectiveness of the algorithm.

epsilon = 0.01 \
beta = 0.5 \
target = 301 (ladybug)


|        Experiment \ Model        | ResNet18 | ResNet50 | ConvNeXt Tiny |
|----------------------------------|----------|----------|---------------|
| acc@1 + FGSM (resnet18, steps=1) |   1.85%  |  ------  |    ------     |
| acc@5 + FGSM (resnet18, steps=1) |   5.71%  |  ------  |    ------     |
| acc@1 + FGSM (resnet18, steps=3) |  33.57%  |  ------  |    ------     |
| acc@5 + FGSM (resnet18, steps=3) |  50.44%  |  ------  |    ------     |
| acc@1 + FGSM (resnet18, steps=7) |  53.64%  |  ------  |    ------     |
| acc@5 + FGSM (resnet18, steps=7) |  69.74%  |  ------  |    ------     |


acc@1 and acc@5 are calculated for the attack target.

## Conclusion
The adversarial transformation is effective in reducing the accuracy of the models if the weights are aviailable. The adversarial transformation is more effective on older models with ReLU activation functions than newer models with more complex activation functions. 

The adversarial transformation is not effective if FGSM model is different from the model being attacked. There is not enough evidence to suggest that the adversarial transformation is effective on models from the same family.

Targeted attacks are more difficult and require multiple perturbations to be effective. Higher number of steps increases the effectiveness of the attack, but also require epsilon scaling to maintain the perceptual quality of the image.

## Future Work
- Use FLIP as a perceptual quality metric.
- Attempt to train epsilon and beta as parameters using FLIP loss as a perceptual quality loss.
- Attempt to generalise the adversarial transformation to attack unknown models.


## References
- [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572)
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)
- [FLIP: A Difference Evaluator for Alternating Images](https://research.nvidia.com/sites/default/files/node/3260/FLIP_Paper.pdf)
