import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

def get_dataloaders(batch_size=32):
    transform = T.Compose([
        T.Resize((64, 64)),
        T.ToTensor()
    ])

    train_dataset = ImageFolder(
        "tiny-imagenet/tiny-imagenet-200/train",
        transform=transform
    )

    val_dataset = ImageFolder(
        "tiny-imagenet/tiny-imagenet-200/val",
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader