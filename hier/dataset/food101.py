from .base import *
from PIL import Image
from collections import defaultdict
import random
import os
import torchvision

class Food101(BaseDataset):
    def __init__(self, root, mode='train', transform=None, limit_per_class=200, num_classes=50):
        self.root = root + '/food41'
        self.mode = mode
        self.transform = transform
        self.limit_per_class = limit_per_class
        self.num_classes = num_classes

        # Tạo class_to_id - chỉ lấy 50 classes đầu tiên
        class_names = sorted(os.listdir(self.root + '/images'))[:self.num_classes]
        self.class_to_id = {name: idx for idx, name in enumerate(class_names)}

        # Split classes theo mode như CUB
        if self.mode == 'train':
            self.classes = range(0, self.num_classes // 2)  # First 25 classes for train
        elif self.mode == 'eval':
            self.classes = range(self.num_classes // 2, self.num_classes)  # Last 25 classes for eval

        # Khởi tạo
        self.ys = []
        self.I = []
        self.im_paths = []

        # Gọi BaseDataset
        BaseDataset.__init__(self, self.root, self.mode, self.transform)

        # Scan images cho các class được chọn
        images_root = os.path.join(self.root, 'images')
        
        # Tạo bộ nhớ tạm để lưu ảnh theo class
        class_to_images = defaultdict(list)
        
        # Scan tất cả ảnh trong thư mục images cho các class được chọn
        for class_name in class_names:
            class_id = self.class_to_id[class_name]
            if class_id in self.classes:  # Chỉ xử lý class thuộc mode hiện tại
                class_path = os.path.join(images_root, class_name)
                if os.path.exists(class_path):
                    for img_name in os.listdir(class_path):
                        if img_name.endswith('.jpg'):
                            img_path = os.path.join(class_path, img_name)
                            class_to_images[class_id].append((class_id, img_path))

        # Thêm ảnh vào dataset
        for class_id, items in class_to_images.items():
            # Shuffle để đảm bảo random selection
            random.shuffle(items)
            
            # Giới hạn số lượng ảnh per class
            selected_items = items[:self.limit_per_class]
            
            # Thêm vào dataset
            for idx, (y, img_path) in enumerate(selected_items):
                self.ys.append(y)
                self.I.append(len(self.I))  # Sequential index
                self.im_paths.append(img_path)