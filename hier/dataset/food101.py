from .base import *
from datasets import load_dataset
from PIL import Image
import io

class Food101(BaseDataset):
    def __init__(self, root, mode='train', transform=None):
        super().__init__(root, mode, transform)
        self.ys, self.I, self.im_paths = [], [], []

        self.classes = list(range(0, 20)) if mode == 'train' else list(range(99, 102))

        # ✅ KHÔNG dùng decode=False – Hugging Face tự decode thành PIL.Image
        dataset = load_dataset("ethz/food101", split=mode, streaming=True)

        index = 0
        for i, item in enumerate(dataset):
            label = item['label']
            if label not in self.classes:
                continue

            try:
                image = item['image']  # đã là PIL.Image
                _ = image.size  # kiểm tra ảnh có lỗi không
            except Exception as e:
                print(f"[Skip] Error reading image at index {i}: {e}")
                continue

            self.ys.append(label)
            self.I.append(index)
            self.im_paths.append(image)  # không cần decode thủ công
            index += 1
