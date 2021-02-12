import torchvision
import torchvision.transforms as transforms


def get_model(args):
    if args.model_name == 'preact18':
        from model.preact18 import PreAct18
        net = PreAct18(in_c=args.in_c, num_classes=args.num_classes)
    elif args.model_name=='sepreact18':
        from model.sepreact18 import SEPreAct18
        net = SEPreAct18(in_c=args.in_c, num_classes=args.num_classes, r=16)
    elif args.model_name=='resnet18':
        from model.resnet18 import ResNet18
        net = ResNet18(in_c=args.in_c, num_classes=args.num_classes)
    elif args.model_name == 'preact34':
        from model.preact34 import PreAct34
        net = PreAct34(in_c=args.in_c, num_classes=args.num_classes)
    elif args.model_name=='sepreact34':
        from model.sepreact34 import SEPreAct34
        net = SEPreAct34(in_c=args.in_c, num_classes=args.num_classes, r=16)
    elif args.model_name == 'preact50':
        from model.preact50 import PreAct50
        net = PreAct50(in_c=args.in_c, num_classes=args.num_classes)
    elif args.model_name=='sepreact50':
        from model.sepreact50 import SEPreAct50
        net = SEPreAct50(in_c=args.in_c, num_classes=args.num_classes, r=16)
    elif args.model_name=='allcnnc':
        from model.allcnnc import AllCNNC
        net = AllCNNC(in_c=args.in_c, num_classes=args.num_classes)
    elif args.model_name=='mobv1':
        from model.mobv1 import MobileNetV1
        net = MobileNetV1(in_c=args.in_c, num_classes=args.num_classes)    
    elif args.model_name=='mobv2':
        from model.mobv2 import MobileNetV2
        net = MobileNetV2(in_c=args.in_c, num_classes=args.num_classes)    
    elif args.model_name=='mobv3':
        from model.mobv3 import MobileNetV3
        net = MobileNetV3(in_c=args.in_c, num_classes=args.num_classes)
    elif args.model_name=='wrn':
        from model.wrn import WideResNet
        net = WideResNet(in_c=args.in_c, num_classes=args.num_classes, l=args.l, widen=args.widen)
    elif args.model_name=='sewrn':
        from model.wrn import WideResNet
        net = WideResNet(in_c=args.in_c, num_classes=args.num_classes, l=args.l, widen=args.widen, se=True, r=16)
    elif args.model_name=='preact_resnext50':
        from model.preact_resnext50 import PreActResNeXt50
        net = PreActResNeXt50(in_c=args.in_c, num_classes=args.num_classes, se=False)
    elif args.model_name=='sepreact_resnext50':
        from model.preact_resnext50 import PreActResNeXt50
        net = PreActResNeXt50(in_c=args.in_c, num_classes=args.num_classes, se=True, r=16)
    else:
        raise NotImplementedError(f"{model_name} is not implemented yet...")

    return net

def get_transform(args):
    train_transform = []
    test_transform = []
    if args.dataset == 'c10' or args.dataset=='c100':
        train_transform.append(transforms.RandomCrop(size=args.size, padding=args.padding))
        train_transform.append(transforms.RandomHorizontalFlip())

    train_transform.append(transforms.ToTensor())
    train_transform.append(transforms.Normalize(mean=args.mean, std=args.std))
    test_transform.append(transforms.ToTensor())
    test_transform.append(transforms.Normalize(mean=args.mean, std=args.std))

    train_transform = transforms.Compose(train_transform)
    test_transform = transforms.Compose(test_transform)

    return train_transform, test_transform
    

def get_dataset(args):
    root = "data"
    if args.dataset == "c10":
        args.num_classes=10
        args.size = 32
        args.padding = 4
        args.mean, args.std = [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]
        train_transform, test_transform = get_transform(args)
        train_ds = torchvision.datasets.CIFAR10(root, train=True, transform=train_transform, download=True)
        test_ds = torchvision.datasets.CIFAR10(root, train=False, transform=test_transform, download=True)

    elif args.dataset == "c100":
        args.num_classes=100
        args.size = 32
        args.padding = 4
        args.mean, args.std = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
        train_transform, test_transform = get_transform(args)
        train_ds = torchvision.datasets.CIFAR100(root, train=True, transform=train_transform, download=True)
        test_ds = torchvision.datasets.CIFAR100(root, train=False, transform=test_transform, download=True)

    else:
        raise NotImplementedError(f"{args.dataset} is not implemented yet.")
    
    return train_ds, test_ds

def get_experiment_name(args):
    if "wrn" in args.model_name:
        model_name = f"{args.model_name}{args.l}_{args.widen}"
    else:
        model_name = args.model_name
    experiment_name = f"{model_name}_{args.dataset}"
    return experiment_name