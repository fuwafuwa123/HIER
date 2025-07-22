from .base import *
from datasets import load_dataset
from PIL import Image
import io

class Food101(BaseDataset):
    def __init__(self, root, mode='train', transform=None):
        self.root = root
        self.mode = mode
        self.transform = transform
        self.ys, self.I, self.images = [], [], []

        # Tùy chọn số class
        if mode == 'train':
            self.classes = list(range(0, 20))
        else:
            self.classes = list(range(99, 102))

        # Tải dataset mà không decode sẵn ảnh để tránh UnicodeDecodeError
        dataset = load_dataset("food101", split=mode, streaming=False).with_format("python", decode=False)

        index = 0
        for i, item in enumerate(dataset):
            label = item['label']
            if label not in self.classes:
                continue

            try:
                img_bytes = item['image']['bytes']
                image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                _ = image.size  # test lỗi ảnh
            except Exception as e:
                print(f"[Skip] Error reading image at index {i}: {e}")
                continue

            self.ys.append(label)
            self.I.append(index)
            self.images.append(image)
            index += 1

        BaseDataset.__init__(self, self.root, self.mode, self.transform)

    def __getitem__(self, index):
        image = self.images[index]
        try:
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"[Warning] Error processing image at index {index}: {e}")
            return self.__getitem__((index + 1) % len(self))

        label = self.ys[index]
        return image, label, index
