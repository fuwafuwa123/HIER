import torch, math, time, argparse, json, os, sys, random
import utils
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import DataLoader

from dataset import *
from dataset.Inshop import Inshop_Dataset
from models.model import init_model
from utils import *
from helpers import get_emb, evaluate
from tqdm import *

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed) 


parser = argparse.ArgumentParser(description="HIER Evaluation Script")

parser.add_argument('--dataset', required=True, type=str,
                   choices=['CUB', 'Cars', 'SOP', 'Inshop', 'Food101', 'NABirds'],
                   help='Dataset to evaluate on')

parser.add_argument('--data_path', default='/kaggle/input', type=str,
        help='Please specify path to the dataset data.')

parser.add_argument('--resume', required=True,
    help='Path to model checkpoint (required)'
)

parser.add_argument('--visualize', action='store_true',
    help='Enable visualization of top-k similar images'
)
parser.add_argument('--query-index', default=0, type=int,
    help='Index of query image for visualization'
)
parser.add_argument('--top-k', default=5, type=int,
    help='Number of top similar images to visualize'
)
parser.add_argument('--save-viz', default=None,
    help='Path to save visualization image'
)

parser.add_argument('--batch_size', default=90, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')

parser.add_argument('--gpu_id', default=0, type=int, help='GPU ID to use')

args = parser.parse_args()

# Set device
device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() and args.gpu_id >= 0 else 'cpu')
if args.gpu_id >= 0:
    torch.cuda.set_device(args.gpu_id)

# Load checkpoint first to get model configuration
if not os.path.isfile(args.resume):
    print(f"No checkpoint found at {args.resume}")
    sys.exit(1)

print(f"Loading checkpoint from {args.resume}")
try:
    # First try with weights_only=True (PyTorch 2.6+ default)
    checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
except Exception as e:
    print(f"Failed to load with weights_only=True: {e}")
    print("Retrying with weights_only=False (use only if you trust the checkpoint source)...")
    # Fallback to weights_only=False for older checkpoints
    checkpoint = torch.load(args.resume, map_location=device, weights_only=False)

# Extract model configuration from checkpoint
if 'args' in checkpoint:
    checkpoint_args = checkpoint['args']
    print("Loaded model configuration from checkpoint:")
    print(f"  Model: {checkpoint_args.model}")
    print(f"  Embedding dimension: {checkpoint_args.emb}")
    print(f"  Hyperbolic curvature: {checkpoint_args.hyp_c}")
    print(f"  Clip radius: {checkpoint_args.clip_r}")
    print(f"  Use last norm: {checkpoint_args.use_lastnorm}")
    print(f"  BN freeze: {checkpoint_args.bn_freeze}")
    print(f"  Pool: {checkpoint_args.pool}")
    print(f"  Patch size: {checkpoint_args.patch_size}")
    
    # Use checkpoint args for model initialization
    model_args = checkpoint_args
else:
    print("Warning: No 'args' found in checkpoint. Using default configuration.")
    # Create a minimal args object with defaults
    class DefaultArgs:
        def __init__(self):
            self.model = 'resnet50'
            self.emb = 128
            self.hyp_c = 0.1
            self.clip_r = 2.3
            self.use_lastnorm = True
            self.bn_freeze = True
            self.pool = 'token'
            self.patch_size = 16
            self.image_size = 224
            self.resize_size = 256
            self.crop_size = 224
            self.use_fp16 = True
    
    model_args = DefaultArgs()

# Initialize model with configuration from checkpoint
print(f"Initializing model with configuration:")
print(f"  Model type: {model_args.model}")
print(f"  Embedding dimension: {model_args.emb}")
print(f"  Hyperbolic curvature: {model_args.hyp_c}")
print(f"  Use last norm: {model_args.use_lastnorm}")
print(f"  BN freeze: {model_args.bn_freeze}")

model = init_model(model_args)
model = model.to(device)

# Print model structure for debugging
print(f"Model structure:")
print(f"  Model type: {type(model)}")
print(f"  Body type: {type(model.body)}")
print(f"  Last layer type: {type(model.last_layer)}")
print(f"  Total parameters: {sum(p.numel() for p in model.parameters())}")

# Load model weights from checkpoint
print(f"Checkpoint keys: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else 'Not a dict'}")

def try_load_state_dict(model, state_dict, key_name=""):
    """Try to load state dict with various fallback strategies"""
    try:
        # First try strict loading
        model.load_state_dict(state_dict, strict=True)
        print(f"Successfully loaded from {key_name} with strict=True")
        return True
    except Exception as e:
        print(f"Failed to load from {key_name} with strict=True: {e}")
        try:
            # Try with strict=False
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            print(f"Successfully loaded from {key_name} with strict=False")
            if missing_keys:
                print(f"Missing keys: {missing_keys[:5]}..." if len(missing_keys) > 5 else f"Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys: {unexpected_keys[:5]}..." if len(unexpected_keys) > 5 else f"Unexpected keys: {unexpected_keys}")
            return True
        except Exception as e2:
            print(f"Failed to load from {key_name} with strict=False: {e2}")
            return False

def print_state_dict_info(state_dict, name=""):
    """Print information about state dict keys"""
    print(f"State dict {name} info:")
    print(f"  Number of keys: {len(state_dict.keys())}")
    print(f"  First 10 keys: {list(state_dict.keys())[:10]}")
    if len(state_dict.keys()) > 10:
        print(f"  Last 10 keys: {list(state_dict.keys())[-10:]}")

# Try loading from different possible keys
load_success = False
for key in ['student', 'stduent', 'model', 'state_dict']:
    if key in checkpoint:
        print(f"Trying to load from '{key}' key")
        print_state_dict_info(checkpoint[key], key)
        if try_load_state_dict(model, checkpoint[key], key):
            load_success = True
            break

if not load_success:
    # Try to load directly if it's a state dict
    try:
        print("Attempting to load checkpoint directly as state dict")
        if try_load_state_dict(model, checkpoint, "checkpoint"):
            load_success = True
    except Exception as e:
        print(f"Failed to load checkpoint directly: {e}")
        
        # Try to find any key that might contain model weights
        if isinstance(checkpoint, dict):
            for key in checkpoint.keys():
                if key not in ['optimizer', 'epoch', 'args', 'fp16_scaler']:
                    print(f"Trying to load from key '{key}'")
                    if try_load_state_dict(model, checkpoint[key], key):
                        load_success = True
                        break
            else:
                print("Could not find valid model weights in any key")
                raise e
        else:
            raise e

if not load_success:
    print("Failed to load model weights from checkpoint")
    sys.exit(1)

print("Checkpoint loaded successfully")
model.eval()

# Set up data transforms based on model type
if model_args.model.startswith("vit"):
    mean_std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
elif model_args.model == "bn_inception":
    mean_std = (104, 117, 128), (1, 1, 1)
else:
    mean_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

# Dataset mapping
ds_list = {"CUB": CUBirds, "SOP": SOP, "Cars": Cars, "Inshop": Inshop_Dataset, "Food101": Food101, "NABirds": NABirds}
ds_class = ds_list[args.dataset]

# Create evaluation transform
eval_tr = T.Compose([
    T.Resize(model_args.resize_size, interpolation=Image.BICUBIC),
    T.CenterCrop(model_args.crop_size),
    T.ToTensor(),
    T.Normalize(*mean_std),
])

# Load evaluation dataset
dl_query = None
dl_gallery = None
if args.dataset == 'Inshop':
    # For Inshop, we need both query and gallery datasets
    ds_query = ds_class(args.data_path, "query", eval_tr)
    ds_gallery = ds_class(args.data_path, "gallery", eval_tr)
    dl_query = DataLoader(ds_query, batch_size=args.batch_size, shuffle=False, num_workers=4)
    dl_gallery = DataLoader(ds_gallery, batch_size=args.batch_size, shuffle=False, num_workers=4)
    dl_ev = dl_query  # For visualization, use query set
else:
    # For other datasets, use evaluation set
    ds_ev = ds_class(args.data_path, "eval", eval_tr)
    dl_ev = DataLoader(ds_ev, batch_size=args.batch_size, shuffle=False, num_workers=4)

def get_emb_s(ds_type="eval"):
    """Wrapper function for get_emb to match the expected signature"""
    return get_emb(model, ds_class, args.data_path, mean_std, ds_type=ds_type)

def get_embeddings_and_paths(model, dataloader):
    """Extract embeddings and image paths from the dataset"""
    model.eval()
    embeddings = []
    labels = []
    image_paths = []
    
    with torch.no_grad():
        for batch_idx, (x, y, indices) in enumerate(dataloader):
            # Get embeddings
            x = x.to(device)
            m = model(x)
            embeddings.append(m.cpu())
            labels.append(y)
            
            # Get image paths for visualization
            if hasattr(dataloader.dataset, 'im_paths'):
                for idx in indices:
                    if idx < len(dataloader.dataset.im_paths):
                        image_paths.append(dataloader.dataset.im_paths[idx])
                    else:
                        image_paths.append(f"Image_{idx}")
    
    embeddings = torch.cat(embeddings, dim=0)
    labels = torch.cat(labels, dim=0)
    
    return embeddings, labels, image_paths

def find_top_k_similar(query_embedding, gallery_embeddings, k=5):
    """Find top-k most similar embeddings using cosine similarity"""
    # Normalize embeddings
    query_embedding = F.normalize(query_embedding, p=2, dim=1)
    gallery_embeddings = F.normalize(gallery_embeddings, p=2, dim=1)
    
    # Compute cosine similarity
    similarity = F.linear(query_embedding, gallery_embeddings)
    
    # Get top-k indices
    top_k_values, top_k_indices = torch.topk(similarity, k=k, dim=1)
    
    return top_k_values, top_k_indices

def visualize_top_k(query_image_path, top_k_image_paths, top_k_scores, save_path=None):
    """Visualize query image and top-k similar images"""
    total_images = 1 + len(top_k_image_paths)
    fig, axes = plt.subplots(1, total_images, figsize=(4*total_images, 4))
    
    # Handle single subplot case
    if total_images == 1:
        axes = [axes]
    
    # Load and display query image
    query_img = Image.open(query_image_path)
    axes[0].imshow(query_img)
    axes[0].set_title('Query Image', fontsize=12)
    axes[0].axis('off')
    
    # Load and display top-k similar images
    for i, (img_path, score) in enumerate(zip(top_k_image_paths, top_k_scores)):
        img = Image.open(img_path)
        axes[i+1].imshow(img)
        axes[i+1].set_title(f'Top {i+1}\nScore: {score:.3f}', fontsize=12)
        axes[i+1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Visualization saved to {save_path}')
    
    plt.show()

def visualize_similar_images(model, dataloader, query_index=0, top_k=5, save_path=None):
    """Main function to visualize top-k similar images"""
    print(f"Extracting embeddings for visualization...")
    embeddings, labels, image_paths = get_embeddings_and_paths(model, dataloader)
    print(f"Extracted embeddings shape: {embeddings.shape}")
    
    # Select query image
    if query_index >= len(embeddings):
        print(f"Query index {query_index} is out of range. Using index 0.")
        query_index = 0
    
    query_embedding = embeddings[query_index:query_index+1]
    query_label = labels[query_index]
    query_image_path = image_paths[query_index] if image_paths and query_index < len(image_paths) else f"Image {query_index}"
    
    print(f"Query image: {query_image_path}")
    print(f"Query label: {query_label}")
    
    # Find top-k similar images
    print(f"Finding top {top_k} similar images...")
    top_k_scores, top_k_indices = find_top_k_similar(query_embedding, embeddings, k=top_k+1)
    
    # Remove the query image itself from results (it will be the most similar)
    top_k_scores = top_k_scores[0, 1:top_k+1]
    top_k_indices = top_k_indices[0, 1:top_k+1]
    
    # Get image paths for top-k results
    top_k_image_paths = []
    for idx in top_k_indices:
        if idx < len(image_paths):
            top_k_image_paths.append(image_paths[idx])
        else:
            top_k_image_paths.append(f"Image {idx}")
    
    # Print results
    print(f"\nTop {top_k} similar images:")
    for i, (idx, score) in enumerate(zip(top_k_indices, top_k_scores)):
        img_path = top_k_image_paths[i]
        label = labels[idx]
        print(f"  {i+1}. Image {idx}: {img_path}")
        print(f"     Label: {label}, Similarity Score: {score:.3f}")
    
    # Visualize results
    if image_paths and all(os.path.exists(path) for path in [query_image_path] + top_k_image_paths):
        print("\nGenerating visualization...")
        visualize_top_k(query_image_path, top_k_image_paths, top_k_scores, save_path)
    else:
        print("\nCannot visualize: Some image paths are not valid or images don't exist")



# Main evaluation
with torch.no_grad():
    print("**Evaluating...**")
    
    # Use the evaluate function from helpers.py for all cases
    print(f"Using evaluation with hyp_c={model_args.hyp_c}")
    Recalls = evaluate(get_emb_s, args.dataset, model_args.hyp_c)
    print(f"Final R@1: {Recalls:.4f}")
    
    # For visualization, we can still use the cosine similarity approach
    if args.visualize:
        print("\n**Starting Visualization...**")
        visualize_similar_images(model, dl_ev, args.query_index, args.top_k, args.save_viz)

    
