# Image Classification Pytorch
Reimplement image classification models for practice. Train them for CIFAR-10 and CIFAR-100 dataset using PyTorch.

## Quick Start

```bash
git clone https://github.com/omihub777/image-classification-pytorch.git
cd image-classification-pytorch/
sh setup.sh

python main.py --api-key [YOUR API KEY OF COMET.ML]
```
* [comet.ml](https://www.comet.ml/): Logger for this repository.(You need to register to get your own api key.)

## Dataset
* CIFAR-10
* CIFAR-100

## Version
* Python: 3.6.9


## Results(Accuracy)

|Models|CIFAR-10|CIFAR-100|
|:--:|:--:|:--:|
|AllConvC|-|-|
|[ResNet18](https://arxiv.org/abs/1512.03385)|-|-|
|[PreAct18](https://arxiv.org/abs/1603.05027)|-|-|
|[PreAct34](https://arxiv.org/abs/1603.05027)|-|-|
|[PreActSE18](https://arxiv.org/abs/1709.01507)|-|-|
|[PreActSE34](https://arxiv.org/abs/1709.01507)|-|-|
|MobV1|-|-|
|MobV2|-|-|
|MNasNet|-|-|
|EfficientNetB0|-|-|


## Hyperparameters
|Params|Values|
|:--|:--:|
|Epochs| 200|
|Batch Size| 128|
|Learning Rate| 0.1|
|LR Milestones| [100, 150]|
|LR Decay Rate| 0.1|
|Weight Decay| 0.0001|

## Models
* ResNet[[He, K.(CVPR'16)]](https://arxiv.org/abs/1512.03385)
* PreAct-ResNet[[He, K.(ECCV'16)]](https://arxiv.org/abs/1603.05027)
* PreAct-ResNet+SEModule[[Hu, J.(CVPR'18)]](https://arxiv.org/abs/1709.01507)
