from .base import *
from PIL import Image
from collections import defaultdict
import random

class Food101(BaseDataset):
    def __init__(self, root, mode='train', transform=None, limit_per_class=150, num_classes=50):
        self.root = root + '/food41'
        self.mode = mode
        self.transform = transform
        self.limit_per_class = limit_per_class
        self.num_classes = num_classes

        # Tạo class_to_id - chỉ lấy 50 classes đầu tiên
        class_names = sorted(os.listdir(self.root + '/images'))[:self.num_classes]
        self.class_to_id = {name: idx for idx, name in enumerate(class_names)}

        # Khởi tạo
        self.ys = []
        self.I = []
        self.im_paths = []

        # Gọi BaseDataset
        BaseDataset.__init__(self, self.root, self.mode, self.transform)

        # Đọc metadata
        meta_file = 'train.txt' if self.mode == 'train' else 'test.txt'
        meta_path = self.root + '/meta/meta/' + meta_file

        # Tạo bộ nhớ tạm để giới hạn số lượng ảnh/class
        class_to_images = defaultdict(list)

        with open(meta_path) as metadata:
            for i, line in enumerate(metadata):
                path = line.strip() + '.jpg'
                class_name = path.split('/')[0]
                
                # Chỉ xử lý các class được chọn
                if class_name in self.class_to_id:
                    class_id = self.class_to_id[class_name]
                    full_path = self.root + '/images/' + path
                    class_to_images[class_id].append((class_id, i, full_path))

        # Lọc tối đa limit_per_class ảnh cho mỗi class
        for class_id, items in class_to_images.items():
            selected = items[:self.limit_per_class]
            for y, idx, img_path in selected:
                self.ys.append(y)
                self.I.append(idx)
                self.im_paths.append(img_path)
