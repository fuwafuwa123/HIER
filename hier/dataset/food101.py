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
                # Get the image bytes directly to avoid EXIF processing
                image_bytes = item['image'].tobytes()
                
                # Create a clean image without metadata
                with io.BytesIO(image_bytes) as buffer:
                    try:
                        # Load image while ignoring EXIF data
                        image = Image.open(buffer)
                        image.load()  # Load all pixel data
                        
                        # Convert to RGB if needed
                        if image.mode != 'RGB':
                            image = image.convert('RGB')
                            
                        # Verify image is valid
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

        if len(self.ys) == 0:
            raise RuntimeError(f"No valid images found for mode '{mode}' with classes {self.classes}")