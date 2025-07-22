from .base import *
from datasets import load_dataset
from PIL import Image
import io

class Food101(BaseDataset):
    def __init__(self, root, mode='train', transform=None):
        self.root = root
        self.mode = mode
        self.transform = transform
        self.ys, self.I, self.im_paths = [], [], []

        if mode == 'train':
            self.classes = list(range(0, 20))
        else:
            self.classes = list(range(99, 102))

        dataset = load_dataset("ethz/food101", split=mode, streaming=True, **{"decode": False})

        index = 0
        for i, item in enumerate(dataset):
            label = item['label']
            if label not in self.classes:
                continue

            try:
                img_bytes = item['image']['bytes']
                image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            except Exception as e:
                print(f"[Skip] Error reading image at index {i}: {e}")
                continue

            self.ys.append(label)
            self.I.append(index)
            self.im_paths.append(image)  
            index += 1

        BaseDataset.__init__(self, self.root, self.mode, self.transform)
