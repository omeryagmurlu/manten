from datasets import load_dataset
from torch.utils.data import Dataset
from torchvision import transforms

DEFAULT_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: 2 * (x - 0.5)),  # to -1, 1
    ]
)


class MNISTDataset(Dataset):
    def __init__(self, training=False):
        self.transform = DEFAULT_TRANSFORM
        self.dataset = load_dataset("mnist", split="train" if training else "test")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        label = item["label"]
        if self.transform:
            image = self.transform(image)
        return {
            "image": image,
            "label": label,
        }
