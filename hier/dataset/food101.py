from .base import *
from PIL import Image


class Food101(BaseDataset):
    def __init__(self, root, mode='train', transform=None):
        self.root = root + '/food41'
        self.mode = mode
        self.transform = transform

        # Tạo class_to_id
        class_names = sorted(os.listdir(self.root + '/images'))
        self.class_to_id = {name: idx for idx, name in enumerate(class_names)}

        # Khởi tạo
        self.ys = []
        self.I = []
        self.im_paths = []

        # Gọi BaseDataset sau khi có self.root, mode, transform
        BaseDataset.__init__(self, self.root, self.mode, self.transform)

        # Đọc metadata
        meta_file = 'train.txt' if self.mode == 'train' else 'test.txt'
        meta_path = self.root + '/meta/meta/' + meta_file

       

        metadata = open(meta_path)
        with open(meta_path) as metadata:
            for i, line in enumerate(metadata):
                path = line.strip() + '.jpg'
                class_name = path.split('/')[0]
                class_id = self.class_to_id[class_name]

                self.ys += [class_id]
                self.I += [i]
                self.im_paths.append(self.root + '/images/' + path)
