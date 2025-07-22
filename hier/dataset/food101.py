from .base import *
from datasets import load_dataset
from PIL import Image
import io

class Food101(BaseDataset):
    def __init__(self, root, mode='train', transform=None):
        super().__init__(root, mode, transform)
        
        # Define classes for train/test
        if mode == 'train':
            self.classes = list(range(0, 20))
        else:
            self.classes = list(range(99, 102))

        # Load dataset with streaming
        dataset = load_dataset("food101", split=mode, streaming=True)
        
        index = 0
        for i, item in enumerate(dataset):
            label = item['label']
            if label not in self.classes:
                continue

            try:
                image = item['image']
                
                # Handle potential image issues
                try:
                    # Convert to RGB if needed
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                        
                    # Verify image is valid by attempting to resize a tiny version
                    image.thumbnail((1, 1))
                    
                except Exception as e:
                    print(f"[Image Process] Error processing image at index {i}: {e}")
                    continue
                    
            except Exception as e:
                print(f"[Skip] Error reading image at index {i}: {e}")
                continue

            self.ys.append(label)
            self.I.append(index)
            self.im_paths.append(image)  # Store PIL Image object directly
            index += 1

        # Verify we have some data
        if len(self.ys) == 0:
            raise RuntimeError(f"No valid images found for mode '{mode}' with classes {self.classes}")