from .base import *
from datasets import load_dataset
from PIL import Image
import io
import warnings

class Food101(BaseDataset):
    def __init__(self, root, mode='train', transform=None):
        super().__init__(root, mode, transform)
        
        self.classes = list(range(0, 20)) if mode == 'train' else list(range(99, 102))
        dataset = load_dataset("food101", split=mode, streaming=True)
        
        warnings.filterwarnings("ignore", category=UserWarning, module="PIL")
        
        index = 0
        for i, item in enumerate(dataset):
            label = item['label']
            if label not in self.classes:
                continue

            try:
                # Get the original image
                image = item['image']
                
                # Create a clean copy by converting to RGB and stripping metadata
                try:
                    # Create new image buffer
                    with io.BytesIO() as buffer:
                        # Save as PNG to avoid JPEG metadata issues
                        image.save(buffer, format='PNG')
                        buffer.seek(0)
                        
                        # Load the clean image
                        clean_image = Image.open(buffer)
                        clean_image.load()
                        
                        # Ensure RGB format
                        if clean_image.mode != 'RGB':
                            clean_image = clean_image.convert('RGB')
                            
                        # Quick validation
                        clean_image.thumbnail((10, 10))
                        
                        self.ys.append(label)
                        self.I.append(index)
                        self.im_paths.append(clean_image)
                        index += 1
                        
                except Exception as e:
                    print(f"[Image Process] Error processing image at index {i}: {e}")
                    continue
                    
            except Exception as e:
                print(f"[Skip] Error reading image at index {i}: {e}")
                continue

        warnings.resetwarnings()
        
        if len(self.ys) == 0:
            raise RuntimeError(f"No valid images found for mode '{mode}' with classes {self.classes}")