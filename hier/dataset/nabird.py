from .base import *

class NABirds(BaseDataset):
    def __init__(self, root, mode, transform = None):
        self.root = root + '/nabirds'
        self.mode = mode
        self.transform = transform
        self.hierarchy = load_hierarchy_txt(os.path.join(self.root, 'hierarchy.txt'))
        self.hierarchical_labels = []
        if self.mode == 'train':
            self.classes = range(0,100)
        elif self.mode == 'eval':
            self.classes = range(100,200)
        
        BaseDataset.__init__(self, self.root, self.mode, self.transform)
        index = 0
        for i in torchvision.datasets.ImageFolder(root = 
                os.path.join(self.root, 'images')).imgs:
            # i[1]: label, i[0]: root
            y = i[1]
            # fn needed for removing non-images starting with `._`
            fn = os.path.split(i[0])[1]
            if y in self.classes and fn[:2] != '._':
                self.ys += [y]
                self.I += [index]
                self.im_paths.append(os.path.join(self.root, i[0]))
                self.hierarchical_labels.append(self.hierarchy[y])  # <-- Add this line
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