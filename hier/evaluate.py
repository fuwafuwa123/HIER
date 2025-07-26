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

parser.add_argument('--metric', default='recall', type=str, choices=['recall', 'ndcg'],
    help='Evaluation metric to use (recall or ndcg)')

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

# Dataset Loader and Sampler
if args.dataset != 'Inshop':
    # For other datasets, use evaluation set
    ev_dataset = ds_class(args.data_path, "eval", eval_tr)
    dl_ev = DataLoader(
        ev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Also load training set for gallery
    train_dataset = ds_class(args.data_path, "train", eval_tr)
    dl_train = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
else:
    # For Inshop, use query and gallery datasets
    query_dataset = ds_class(args.data_path, "query", eval_tr)
    dl_query = DataLoader(
        query_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    gallery_dataset = ds_class(args.data_path, "gallery", eval_tr)
    dl_gallery = DataLoader(
        gallery_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

def get_emb_s(ds_type="eval"):
    """Wrapper function for get_emb to match the expected signature"""
    return get_emb(model, ds_class, args.data_path, mean_std, ds_type=ds_type)

def get_embeddings_from_dataloader(model, dataloader):
    """Extract embeddings from a dataloader"""
    model.eval()
    embeddings = []
    labels = []
    image_paths = []
    
    with torch.no_grad():
        for batch_idx, (x, y, indices) in enumerate(dataloader):
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

def visualize_top_k(query_image_path, top_k_image_paths, top_k_scores, query_label, top_k_labels, ndcg_scores, save_path=None):
    """Visualize query image and top-k similar images with colored borders based on NDCG thresholds"""
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
    
    # Load and display top-k similar images with colored borders based on NDCG
    for i, (img_path, score, label, ndcg_score) in enumerate(zip(top_k_image_paths, top_k_scores, top_k_labels, ndcg_scores)):
        img = Image.open(img_path)
        axes[i+1].imshow(img)
        
        # Determine border color based on NDCG threshold
        if ndcg_score >= 0.7:  # High NDCG threshold for green
            border_color = 'green'
            
        elif ndcg_score >= 0.3:  # Medium NDCG threshold for orange
            border_color = 'orange'
            
        else:  # Low NDCG threshold for red
            border_color = 'red'
    
        
        # Add colored border
        axes[i+1].set_title(f'Top {i+1}', fontsize=12, color=border_color)
        axes[i+1].axis('off')
        
        # Add colored border around the image
        for spine in axes[i+1].spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Visualization saved to {save_path}')
    
    plt.show()


def visualize_query_gallery_comparison(model, query_index=0, top_k=5, save_path=None):
    """Visualize query vs gallery comparison using proper dataloaders with colored borders and NDCG-based ranking"""
    print(f"Extracting embeddings using proper dataloaders...")
    
    # Use the appropriate dataloaders based on dataset
    if args.dataset == 'Inshop':
        # For Inshop, use query and gallery dataloaders
        query_emb, query_labels, query_paths = get_embeddings_from_dataloader(model, dl_query)
        gallery_emb, gallery_labels, gallery_paths = get_embeddings_from_dataloader(model, dl_gallery)
    else:
        # For other datasets, use eval as query and train as gallery
        query_emb, query_labels, query_paths = get_embeddings_from_dataloader(model, dl_ev)
        gallery_emb, gallery_labels, gallery_paths = get_embeddings_from_dataloader(model, dl_train)
    
    print(f"Query embeddings shape: {query_emb.shape}")
    print(f"Gallery embeddings shape: {gallery_emb.shape}")
    
    # Select query image
    if query_index >= len(query_emb):
        print(f"Query index {query_index} is out of range. Using index 0.")
        query_index = 0
    
    query_embedding = query_emb[query_index:query_index+1]
    query_label = query_labels[query_index]
    query_image_path = query_paths[query_index] if query_paths and query_index < len(query_paths) else f"Query_{query_index}"
    
    print(f"Query image: {query_image_path}")
    print(f"Query label: {query_label}")
    
    # Calculate similarity matrix using the appropriate metric
    print(f"Calculating similarity matrix...")
    
    if model_args.hyp_c > 0:
        # Use hyperbolic distance for hyperbolic models
        from hyptorch.pmath import dist_matrix
        sim = torch.empty(len(query_embedding), len(gallery_emb), device="cuda")
        for i in range(len(query_embedding)):
            sim[i : i + 1] = -dist_matrix(query_embedding[i : i + 1], gallery_emb, model_args.hyp_c)
    else:
        # Use cosine similarity for Euclidean models
        query_embedding_norm = F.normalize(query_embedding, p=2, dim=1)
        gallery_emb_norm = F.normalize(gallery_emb, p=2, dim=1)
        sim = F.linear(query_embedding_norm, gallery_emb_norm)
    
    # Calculate NDCG-based ranking
    print(f"Calculating NDCG-based ranking...")
    
    # Get all similarities for the query
    similarities = sim[0]  # Shape: [gallery_size]
    
    # Create relevance scores (1 if same class, 0 otherwise)
    relevances = (gallery_labels == query_label).float().to(similarities.device)
    
    # Sort by similarity (descending) and get indices
    sorted_similarities, sorted_indices = torch.sort(similarities, descending=True)
    sorted_relevances = relevances[sorted_indices]
    
    # Calculate DCG for the sorted list
    def calc_dcg(relevances, k):
        relevances = relevances[:k]
        if len(relevances) == 0:
            return 0.0
        return torch.sum((2**relevances - 1) / torch.log2(torch.arange(2, len(relevances) + 2, device=relevances.device).float()))
    
    # Calculate ideal DCG (perfect ranking)
    ideal_relevances = torch.sort(relevances, descending=True)[0]
    idcg = calc_dcg(ideal_relevances, top_k)
    
    # Calculate NDCG for each position
    ndcg_scores = []
    for k in range(1, top_k + 1):
        dcg = calc_dcg(sorted_relevances, k)
        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcg_scores.append(ndcg)
    
    # Get top-k indices based on similarity ranking
    top_k_indices = sorted_indices[:top_k]
    top_k_similarities = sorted_similarities[:top_k]
    
    # Get image paths and labels for top-k results
    top_k_image_paths = []
    top_k_labels = []
    for idx in top_k_indices:
        if idx < len(gallery_paths):
            top_k_image_paths.append(gallery_paths[idx])
        else:
            top_k_image_paths.append(f"Gallery_{idx}")
        top_k_labels.append(gallery_labels[idx])
    
    # Print results with NDCG scores
    print(f"\nTop {top_k} similar images from gallery (ranked by similarity, showing NDCG@k):")
    for i, (idx, similarity, label, ndcg_at_k) in enumerate(zip(top_k_indices, top_k_similarities, top_k_labels, ndcg_scores)):
        img_path = top_k_image_paths[i]
        print(f"  {i+1}. Gallery Image {idx}: {img_path}")
        print(f"     Label: {label}, Similarity: {similarity:.3f}, NDCG@{i+1}: {ndcg_at_k:.3f}")
    
    # Visualize results
    if query_paths and gallery_paths and all(os.path.exists(path) for path in [query_image_path] + top_k_image_paths):
        print("\nGenerating visualization...")
        visualize_top_k(query_image_path, top_k_image_paths, top_k_similarities, query_label, top_k_labels, ndcg_scores, save_path)
    else:
        print("\nCannot visualize: Some image paths are not valid or images don't exist")


# Main evaluation
with torch.no_grad():
    print("**Evaluating...**")
    
    # Use the evaluate function from helpers.py for all cases
    print(f"Using {args.metric} evaluation with hyp_c={model_args.hyp_c}")
    if args.metric == 'ndcg':
        score = evaluate(get_emb_s, args.dataset, model_args.hyp_c, metric="ndcg")
        print(f"Final NDCG@1: {score:.4f}")
    else:
        Recalls = evaluate(get_emb_s, args.dataset, model_args.hyp_c, metric="recall")
        print(f"Final R@1: {Recalls:.4f}")
    
    # For visualization, we can still use the cosine similarity approach
    if args.visualize:
        print("\n**Starting Visualization...**")
        visualize_query_gallery_comparison(model, args.query_index, args.top_k, args.save_viz)

    
