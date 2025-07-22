from .cars import Cars
from .cub import CUBirds
from .SOP import SOP
from .food101 import Food101
from .base import BaseDataset


_type = {
    'cars': Cars,
    'cub': CUBirds,
    'SOP': SOP,
    'Food101': Food101,
}

def load(name, root, mode, transform = None):
    return _type[name](root = root, mode = mode, transform = transform)
    
