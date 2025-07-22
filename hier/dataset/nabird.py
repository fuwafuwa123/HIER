from .base import *
import os
import torchvision

class NABirds(BaseDataset):
    def __init__(self, root, mode, transform=None):
        self.root = os.path.join(root, 'nabirds')
        self.mode = mode
        self.transform = transform

        self.hierarchy = load_hierarchy_txt(os.path.join(self.root, 'hierarchy.txt'))
        self.hierarchical_labels = []

        # Load mapping from class folder name → NABirds class ID
        class_map = load_class_mapping(os.path.join(self.root, 'classes.txt'))

        # Load ImageFolder to get label → folder name mapping
        dataset = torchvision.datasets.ImageFolder(root=os.path.join(self.root, 'images'))
        idx_to_classname = {v: k for k, v in dataset.class_to_idx.items()}

        # Define class ranges
        if self.mode == 'train':
            self.classes = range(0, 100)
        elif self.mode == 'eval':
            self.classes = range(100, 200)

        BaseDataset.__init__(self, self.root, self.mode, self.transform)

        index = 0
        for img_path, label in dataset.imgs:
            class_folder = idx_to_classname[label]
            class_id = class_map[class_folder]  # NABirds class_id (1-based)

            fn = os.path.split(img_path)[1]
            if label in self.classes and fn[:2] != '._':
                self.ys.append(label)
                self.I.append(index)
                self.im_paths.append(os.path.join(self.root, img_path))

                if class_id in self.hierarchy:
                    self.hierarchical_labels.append(self.hierarchy[class_id])
                else:
                    print(f"[WARN] Class ID {class_id} not found in hierarchy")
                    self.hierarchical_labels.append([-1])  # optional fallback

                index += 1
        

def load_hierarchy_txt(hierarchy_file):
    parent_map = {}
    hierarchy_map = {}
    with open(hierarchy_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                child, parent = map(int, parts)
                parent_map[child] = parent

    hierarchy_map = {}
    for child in parent_map.keys():
        path = []
        current = child
        while current in parent_map and current != 0:
            path.insert(0, current)
            current = parent_map[current]
        if current != 0:
            path.insert(0, current)
        hierarchy_map[child] = path
    return hierarchy_map

def load_class_mapping(class_file):
    class_map = {}
    with open(class_file, 'r') as f:
        for line in f:
            id_str, folder = line.strip().split()
            class_map[folder] = int(id_str)
    return class_map

class_map = load_class_mapping(os.path.join(self.root, 'classes.txt'))

dataset = torchvision.datasets.ImageFolder(root=os.path.join(self.root, 'images'))
idx_to_classname = {v: os.path.basename(k) for k, v in dataset.class_to_idx.items()}