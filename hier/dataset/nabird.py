from .base import *

class NABirds(BaseDataset):
    def __init__(self, root, mode, transform = None):
        self.root = root + '/nabirds'
        self.mode = mode
        self.transform = transform
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
                index += 1
    

    