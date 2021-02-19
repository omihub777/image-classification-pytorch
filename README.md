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
* PyTorch: 1.7.1
* PyTorch-Lightning: 1.1.6
* Comet-ml: 3.3.1


## Results(Accuracy)

|Models|CIFAR-10|CIFAR-100|#Params|
|:--:|:--:|:--:|:--:|
|[AllCNNC](https://arxiv.org/abs/1412.6806)|93.640|72.040|1.4M|
|[VGG16](https://arxiv.org/abs/1409.1556)|-|-|-|
|[ResNet18](https://arxiv.org/abs/1512.03385)|94.465|74.465|11.2M|
|[PreAct18](https://arxiv.org/abs/1603.05027)|94.575|75.415|11.2M|
|[PreAct34](https://arxiv.org/abs/1603.05027)|95.010|75.715|21.3M|
|[PreAct50](https://arxiv.org/abs/1603.05027)|94.970|76.685|23.5M|
|[WRN22-8](https://arxiv.org/abs/1605.07146)|95.080|77.775|17.2M|
|[PreAct-ResNeXt50](https://arxiv.org/abs/1611.05431)|94.950|78.315|23.0M|
|DenseNet|-|-|-|
|[SEPreAct18](https://arxiv.org/abs/1709.01507)|94.685|76.135|11.3M|
|[SEPreAct34](https://arxiv.org/abs/1709.01507)|95.000|76.095|21.4M|
|[SEPreAct50](https://arxiv.org/abs/1709.01507)|95.075|77.615|26.0M|
|[SEWRN22-8](https://arxiv.org/abs/1709.01507)|**95.530**|*77.830*|17.3M|
|[SEPreAct-ResNeXt50](https://arxiv.org/abs/1709.01507)|*95.160*|**78.520**|25.5M|
|[MobV1](https://arxiv.org/abs/1704.04861)|94.000|75.250|3.2M|
|[MobV2](https://arxiv.org/abs/1801.04381)|94.085|75.425|2.2M|
|[MobV3](https://arxiv.org/abs/1905.02244)|93.980|75.430|4.2M|
|MNasNet|-|-|-|
|EfficientNetB0|-|-|-|

* Did experiments 2 times and report the averaged best accuracy.
* Apply "Random Crop + Horizontal Flip + Normalization" as Data Augmentation.
* **bold** and *italic* indicates **1st** and *2nd* place.

## Hyperparameters
|Params|Values|
|:--|:--:|
|Epochs| 200|
|Batch Size| 128|
|Learning Rate| 0.1|
|LR Milestones| [100, 150]|
|LR Decay Rate| 0.1|
|Weight Decay| 1e-4|

* For the fair comparison, all experiments are done under the same experimental settings.


## Models
* All-CNN-C(+BN)[[Springenberg, J.(ICLRW'15)]](https://arxiv.org/abs/1412.6806)
* VGG16(+BN)[Simonyan, K.(ICLR'15)](https://arxiv.org/abs/1409.1556)
* ResNet[[He, K.(CVPR'16)]](https://arxiv.org/abs/1512.03385)
* PreAct-ResNet[[He, K.(ECCV'16)]](https://arxiv.org/abs/1603.05027)
* WideResNet[[Zagoruyko, S.(BMVC'16)]](https://arxiv.org/abs/1605.07146)
* ResNeXt[[Xie, S.(CVPR'17)]](https://arxiv.org/abs/1611.05431)
* SENet[[Hu, J.(CVPR'18)]](https://arxiv.org/abs/1709.01507)
* MobileNetV1[[Howard, A.(2017)]](https://arxiv.org/abs/1704.04861)
* MobileNetV2[[Sandler, M.(CVPR'18)]](https://arxiv.org/abs/1801.04381)
* MobileNetV3[[Howard, A.(ICCV'19)]](https://arxiv.org/abs/1905.02244)

## Miscellaneous
* Spatial dimensions of tensors right before GAP should be 4x4 rather than 2x2.
* No bias term used for Squeeze-and-Excitation Module.
* Use dropout right after activation.(in WRN, at least.)
    * If apply dropout right after conv, the performance degrades by 1-2% in CIFAR-100
