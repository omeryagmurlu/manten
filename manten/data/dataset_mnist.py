import torchvision
from torch.utils.data import Dataset
from torchvision import transforms

DEFAULT_TRANSFORM = transforms.Compose(
    [
        # transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: 2 * (x - 0.5)),  # to -1, 1
    ]
)


class MNISTDataset(Dataset):
    def __init__(self, training=False):
        self.transform = DEFAULT_TRANSFORM
        self.dataset = torchvision.datasets.MNIST(
            root="~/mnist/", train=training, download=True
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        image = x
        label = y
        if self.transform:
            image = self.transform(image)
        return {
            "image": image,
            "label": label,
        }

    def get_dataset_info(self):
        return {}
