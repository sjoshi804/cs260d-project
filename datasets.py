from torchvision import datasets, transforms

def get_dataset(dataset: str):
    if dataset == "mnist":
        return datasets.MNIST("~/data", train=True, download=True), datasets.MNIST("~/data", train=False, download=True)
    elif dataset == "cifar10":
        # Image Preprocessing
        normalize = transforms.Normalize(
            mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
            std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

        # Setup train transforms
        train_transform = transforms.Compose([])
        train_transform.transforms.append(transforms.RandomCrop(32, padding=4))
        train_transform.transforms.append(transforms.RandomHorizontalFlip())
        train_transform.transforms.append(transforms.ToTensor())
        train_transform.transforms.append(normalize)

        # Setup test transforms
        test_transform = transforms.Compose([transforms.ToTensor(), normalize])
        return datasets.CIFAR10("~/data", train=True, transform=train_transform, download=True), datasets.CIFAR10("~/data", train=False, transform=test_transform, download=True)
    else:
        raise NotImplementedError(f"{dataset} not supported yet.")