from .base import *
from datasets import load_dataset
from PIL import Image
import io
import warnings

class Food101(BaseDataset):
    def __init__(self, root, mode='train', transform=None):
        self.root = root + '/food41'
        self.mode = mode
        self.transform = transform
        if self.mode == 'train':
            self.classes = range(0,100)
        elif self.mode == 'eval':
            self.classes = range(100,200)
        
        BaseDataset.__init__(self, self.root, self.mode, self.transform)
        # Select correct metadata file based on mode
        meta_file = 'train.txt' if self.mode == 'train' else 'test.txt'
        meta_path = os.path.join(self.root, 'meta', 'meta', meta_file)
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")
        metadata = open(meta_path)
        for i, (image_id, class_id, _, path) in enumerate(map(str.split, metadata)):
            if i > 0:
                if int(class_id)-1 in self.classes:
                    self.ys += [int(class_id)-1]
                    self.I += [int(image_id)-1]
                    self.im_paths.append(os.path.join(self.root, 'images', path))