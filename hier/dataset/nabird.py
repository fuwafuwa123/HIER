from .base import *
from PIL import Image

class NABirds(BaseDataset):
    def __init__(self, root, mode, transform=None):
        self.root = root + '/nabirds'
        self.mode = mode
        self.transform = transform
        self.hierarchical_labels = []

        self.hierarchy = load_hierarchy_txt(self.root + '/hierarchy.txt')
        image_paths = load_image_paths(self.root + '/images.txt')
        image_labels = load_image_labels(self.root + '/image_class_labels.txt')

        # Xác định tập class
        all_classes = sorted(set(image_labels.values()))
        all_classes = [c - 1 for c in all_classes]  # convert to 0-based
        num_classes = len(all_classes)
        split = num_classes // 2

        if self.mode == 'train':
            self.classes = set(all_classes[:split])
        elif self.mode == 'eval':
            self.classes = set(all_classes[split:])
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        BaseDataset.__init__(self, self.root, self.mode, self.transform)

        index = 0
        for img_id, rel_path in image_paths.items():
            class_id = image_labels[img_id] - 1  # convert to 0-based
            path = f"{self.root}/images/{rel_path}"

            if class_id in self.classes:
                self.ys.append(class_id)
                self.I.append(index)
                self.im_paths.append(path)
                self.hierarchical_labels.append(self.hierarchy.get(class_id, [class_id]))
                index += 1

    def __getitem__(self, index):
        im = Image.open(self.im_paths[index]).convert("RGB")
        if self.transform:
            im = self.transform(im)
        return im, self.ys[index], self.hierarchical_labels[index]

# --------- Loaders ---------

def load_image_paths(file):
    paths = {}
    with open(file, 'r') as f:
        for line in f:
            img_id, rel_path = line.strip().split()
            paths[img_id] = rel_path
    return paths

def load_image_labels(file):
    labels = {}
    with open(file, 'r') as f:
        for line in f:
            img_id, class_id = line.strip().split()
            labels[img_id] = int(class_id)
    return labels

def load_hierarchy_txt(hierarchy_file):
    parent_map = {}
    with open(hierarchy_file, 'r') as f:
        for line in f:
            child, parent = map(int, line.strip().split())
            parent_map[child] = parent

    hierarchy_map = {}
    for child in parent_map:
        path = []
        current = child
        while current in parent_map and current != 0:
            path.insert(0, current)
            current = parent_map[current]
        if current != 0:
            path.insert(0, current)
        hierarchy_map[child - 1] = [p - 1 for p in path]  # 0-based
    return hierarchy_map
