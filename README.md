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

|Models|CIFAR-10|CIFAR-100|#Params|
|:--:|:--:|:--:|:--:|
|[AllCNNC](https://arxiv.org/abs/1412.6806)|93.640|72.040|1.4M|
|[ResNet18](https://arxiv.org/abs/1512.03385)|94.465|74.465|11.2M|
|[PreAct18](https://arxiv.org/abs/1603.05027)|94.575|75.415|11.2M|
|[PreAct34](https://arxiv.org/abs/1603.05027)|95.010|75.715|21.3M|
|[PreAct50](https://arxiv.org/abs/1603.05027)|-|-|23.5M|
|[SEPreAct18](https://arxiv.org/abs/1709.01507)|94.685|76.135|11.3M|
|[SEPreAct34](https://arxiv.org/abs/1709.01507)|95.000|76.095|21.4M|
|[SEPreAct50](https://arxiv.org/abs/1709.01507)|-|-|26.0M|
|MobV1|-|-|-|
|MobV2|-|-|-|
|MNasNet|-|-|-|
|EfficientNetB0|-|-|-|

* Did experiments 2 times and report the averaged best accuracy.

## Hyperparameters
|Params|Values|
|:--|:--:|
|Epochs| 200|
|Batch Size| 128|
|Learning Rate| 0.1|
|LR Milestones| [100, 150]|
|LR Decay Rate| 0.1|
|Weight Decay| 1e-4|

* PreAct50/SEPreAct50 are trained with BS:64 and LR:0.05 (follow linear scaling rule.)

## Models
* All-CNN-C(+BN)[[Springenberg, J.(ICLRW'15)]](https://arxiv.org/abs/1412.6806)
* ResNet[[He, K.(CVPR'16)]](https://arxiv.org/abs/1512.03385)
* PreAct-ResNet[[He, K.(ECCV'16)]](https://arxiv.org/abs/1603.05027)
* PreAct-ResNet+SEModule[[Hu, J.(CVPR'18)]](https://arxiv.org/abs/1709.01507)

## Miscellaneous
* Spatial dimensions of tensors right before GAP should be 4x4 rather than 2x2.
