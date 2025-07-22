from .base import *
from datasets import load_dataset
from PIL import Image

class Food101(BaseDataset):
    def __init__(self, root, mode='train', transform=None):
        self.root = root
        self.mode = mode
        self.transform = transform
        self.ys, self.I, self.images = [], [], []

        if mode == 'train':
            self.classes = list(range(0, 10))
        else:
            self.classes = list(range(99, 102))  # hoặc giữ là 0–100 nếu muốn test đủ class

        # Load dữ liệu từ Hugging Face
        dataset = load_dataset("food101", split=mode)

        index = 0
        for item in dataset:
            label = item['label']
            if label in self.classes:
                self.ys.append(label)
                self.I.append(index)
                self.images.append(item['image'])  # giữ dạng PIL.Image
                index += 1

        # Gọi constructor cha sau khi gán đủ thông tin
        BaseDataset.__init__(self, self.root, self.mode, self.transform)

    def __getitem__(self, index):
        image = self.images[index]
        if image.mode != 'RGB':
            image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.ys[index]
        return image, label, index
