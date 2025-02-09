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
git clone

# Install dependencies
pip install -r requirements.txt
```

## Project Structure
adverarial_transform.py - adversarial transformation.
single_image_example.py - single image adversarial transformation.
model_inference.py - ImageNet-1k-val benchmarking.

## Results
I use resnet18, resnet50, and ConvNeXt Tiny models to benchmark the adversarial transformation on ImageNet-1k-val dataset with epsilon set to 0.05. 

Additionally, I explore extrapolation of resnet18 as an FGSM model to measure the effectiveness of theis method on other models from the same family and not.
The results are shown in the table below:


|      Experiment \ Model      | ResNet18 | ResNet50 | ConvNeXt Tiny |
|------------------------------|----------|----------|---------------|
| acc@1                        |  69.76%  |  80.34%  |    82.13%     |
| acc@5                        |  89.08%  |  95.13%  |    95.95%     |
| acc@1 + FGSM (resnet18)      |   1.24%  |  68.87%  |    73.49%     |
| acc@5 + FGSM (resnet18)      |  16.54%  |  90.45%  |    92.44%     |
| acc@1 + FGSM (convnext_tiny) |  ------  |  ------  |    34.07%     |
| acc@5 + FGSM (convnext_tiny) |  ------  |  ------  |    57.69%     |

## Conclusion
The adversarial transformation is effective in reducing the accuracy of the models if the weights are aviailable. The adversarial transformation is more effective on older models with ReLU activation functions than newer models with more complex activation functions. 

The adversarial transformation is not effective if FGSM model is different from the model being attacked. There is not enough evidence to suggest that the adversarial transformation is effective on models from the same family.

## Future Work
- Use FLIP evaluate perceptual quality of images.
- Attempt to train epsilon as a parameter using FLIP loss as a perceptual quality loss.
- Attempt to generalise the adversarial transformation to attack unknown models.

## References
- [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572)
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)
- [FLIP: A Difference Evaluator for Alternating Images](https://research.nvidia.com/sites/default/files/node/3260/FLIP_Paper.pdf)
