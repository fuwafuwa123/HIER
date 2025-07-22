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

        # Set class split
        if self.mode == 'train':
            self.classes = range(0, 100)
        elif self.mode == 'eval':
            self.classes = range(100, 200)

        BaseDataset.__init__(self, self.root, self.mode, self.transform)

        index = 0
        for img_id in image_paths.keys():
            path = self.root + '/images/' + image_paths[img_id]
            class_id = image_labels[img_id] - 1  # convert to 0-based

            if class_id in self.classes:
                self.ys.append(class_id)
                self.I.append(index)
                self.im_paths.append(path)

                if class_id in self.hierarchy:
                    self.hierarchical_labels.append(self.hierarchy[class_id])
                else:
                    self.hierarchical_labels.append([-1])
                index += 1

    def __getitem__(self, index):
        im = Image.open(self.im_paths[index]).convert("RGB")
        if self.transform:
            im = self.transform(im)
        label = self.ys[index]
        hierarchy = self.hierarchical_labels[index]
        return im, label, hierarchy


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
        hierarchy_map[child - 1] = [p - 1 for p in path]  # adjust to 0-based
    return hierarchy_map
