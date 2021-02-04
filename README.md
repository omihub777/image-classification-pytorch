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
|[AllCNNC](https://arxiv.org/abs/1412.6806)|-|-|1.4M|
|[ResNet18](https://arxiv.org/abs/1512.03385)|-|-|11.2M|
|[PreAct18](https://arxiv.org/abs/1603.05027)|92.995|69.125|11.2M|
|[PreAct34](https://arxiv.org/abs/1603.05027)|-|-|-|
|[PreActSE18](https://arxiv.org/abs/1709.01507)|92.980|68.725|11.3M|
|[PreActSE34](https://arxiv.org/abs/1709.01507)|-|-|
|MobV1|-|-|-|
|MobV2|-|-|-|
|MNasNet|-|-|-|
|EfficientNetB0|-|-|-|


## Hyperparameters
|Params|Values|
|:--|:--:|
|Epochs| 200|
|Batch Size| 128|
|Learning Rate| 0.1|
|LR Milestones| [100, 150]|
|LR Decay Rate| 0.1|
|Weight Decay| 1e-4|

## Models
* All-CNN-C[[Springenberg, J.(ICLRW'15)]](https://arxiv.org/abs/1412.6806)
* ResNet[[He, K.(CVPR'16)]](https://arxiv.org/abs/1512.03385)
* PreAct-ResNet[[He, K.(ECCV'16)]](https://arxiv.org/abs/1603.05027)
* PreAct-ResNet+SEModule[[Hu, J.(CVPR'18)]](https://arxiv.org/abs/1709.01507)

## Miscellaneous
* Shape of tensors right before GAP should be $4\times 4$ rather than $2\times 2$. (I think.)
