from .base import *
from datasets import load_dataset
from PIL import Image

class Food101(BaseDataset):
    def __init__(self, root, mode='train', transform=None):
        self.root = root
        self.mode = mode
        self.transform = transform
        self.ys, self.I, self.images = [], [], []

        # Tùy chọn số class
        if mode == 'train':
            self.classes = list(range(0, 20))  # ví dụ chỉ lấy 20 class đầu
        else:
            self.classes = list(range(99, 102))  # ví dụ test một vài class cuối

        dataset = load_dataset("food101", split=mode)

        index = 0
        for i, item in enumerate(dataset):
            label = item['label']
            if label in self.classes:
                try:
                    image = item['image']
                    _ = image.size  # thử truy cập để phát hiện ảnh lỗi
                except Exception as e:
                    print(f"[Skip] Error reading image at index {i}: {e}")
                    continue
                self.ys.append(label)
                self.I.append(index)  # hoặc I.append(i) nếu bạn cần giữ index gốc
                self.images.append(image)
                index += 1

        BaseDataset.__init__(self, self.root, self.mode, self.transform)

    def __getitem__(self, index):
        image = self.images[index]
        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"[Warning] Error processing image at index {index}: {e}")
            # fallback: return blank image hoặc sample khác
            return self.__getitem__((index + 1) % len(self))

        label = self.ys[index]
        return image, label, index
