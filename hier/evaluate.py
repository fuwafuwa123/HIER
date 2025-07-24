import torch, math, time, argparse, json, os, sys, random
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
from helpers import get_emb
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

parser.add_argument('--resume', default='',
    help='Path of resuming model checkpoint'
)

parser.add_argument('--model', default='resnet50', type=str,
        choices=['resnet50', 
                 'deit_small_distilled_patch16_224', 'vit_small_patch16_224', 'dino_vits16'],
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")

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

parser.add_argument('--bn_freeze', type=bool, default=True)
parser.add_argument('--use_lastnorm', type=bool, default=True)
parser.add_argument('--emb', default=128, type=int, help='Embedding dimension')
parser.add_argument('--hyp_c', default=0.0, type=float, help='Hyperbolic curvature')
parser.add_argument('--clip_r', default=2.3, type=float, help='Clip radius for hyperbolic space')
parser.add_argument('--gpu_id', default=0, type=int, help='GPU ID to use')

args = parser.parse_args()

# Set device
device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() and args.gpu_id >= 0 else 'cpu')
if args.gpu_id >= 0:
    torch.cuda.set_device(args.gpu_id)

# Initialize model
model = init_model(args)
model = model.to(device)

# Load checkpoint if provided
if args.resume:
    if os.path.isfile(args.resume):
        print(f"Loading checkpoint from {args.resume}")
        try:
            # First try with weights_only=True (PyTorch 2.6+ default)
            checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        except Exception as e:
            print(f"Failed to load with weights_only=True: {e}")
            print("Retrying with weights_only=False (use only if you trust the checkpoint source)...")
            # Fallback to weights_only=False for older checkpoints
            checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        
        if 'student' in checkpoint:
            model.load_state_dict(checkpoint['student'])
        elif 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
        print("Checkpoint loaded successfully")
    else:
        print(f"No checkpoint found at {args.resume}")

model.eval()

# Set up data transforms
if args.model.startswith("vit"):
    mean_std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
elif args.model == "bn_inception":
    mean_std = (104, 117, 128), (1, 1, 1)
else:
    mean_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

# Dataset mapping
ds_list = {"CUB": CUBirds, "SOP": SOP, "Cars": Cars, "Inshop": Inshop_Dataset, "Food101": Food101, "NABirds": NABirds}
ds_class = ds_list[args.dataset]

# Create evaluation transform
eval_tr = T.Compose([
    T.Resize(256, interpolation=Image.BICUBIC),
    T.CenterCrop(224),
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

def get_embeddings_and_paths(model, dataloader):
    """Extract embeddings and image paths from the dataset"""
    model.eval()
    embeddings = []
    labels = []
    image_paths = []
    
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(dataloader):
            # Get embeddings
            x = x.to(device)
            m = model(x)
            embeddings.append(m.cpu())
            labels.append(y)
            
            # Get image paths for visualization
            if hasattr(dataloader.dataset, 'im_paths'):
                start_idx = batch_idx * dataloader.batch_size
                end_idx = min(start_idx + dataloader.batch_size, len(dataloader.dataset))
                image_paths.extend(dataloader.dataset.im_paths[start_idx:end_idx])
    
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
    fig, axes = plt.subplots(1, 6, figsize=(20, 4))
    
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
    if query_index >= len(dataloader.dataset):
        print(f"Query index {query_index} is out of range. Using index 0.")
        query_index = 0
    
    query_embedding = embeddings[query_index:query_index+1]
    query_label = labels[query_index]
    query_image_path = image_paths[query_index] if image_paths else f"Image {query_index}"
    
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

def evaluate_cos(model, dataloader):
    """Evaluate using cosine similarity"""
    print("Evaluating with cosine similarity...")
    embeddings, labels, _ = get_embeddings_and_paths(model, dataloader)
    
    # Normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    
    # Compute similarity matrix
    similarity = embeddings @ embeddings.T
    
    # Get top-k predictions for each query
    k_list = [1, 2, 4, 8] if args.dataset not in ['SOP', 'Inshop'] else ([1, 10, 100, 1000] if args.dataset == 'SOP' else [1, 10, 20, 30])
    
    recalls = []
    for k in k_list:
        top_k_values, top_k_indices = torch.topk(similarity, k=k+1, dim=1)
        
        # Remove self-similarity
        top_k_indices = top_k_indices[:, 1:]
        
        # Calculate recall@k
        correct = 0
        total = 0
        for i, (query_label, top_k_labels) in enumerate(zip(labels, top_k_indices)):
            if query_label in labels[top_k_labels]:
                correct += 1
            total += 1
        
        recall = correct / total if total > 0 else 0
        recalls.append(recall)
        print(f"R@{k}: {recall:.4f}")
    
    return recalls[0]  # Return R@1

def evaluate_cos_SOP(model, dataloader):
    """Evaluate SOP dataset specifically"""
    return evaluate_cos(model, dataloader)

def evaluate_cos_Inshop(model, dl_query, dl_gallery):
    """Evaluate Inshop dataset specifically"""
    print("Evaluating Inshop dataset...")
    
    # Get embeddings for query and gallery
    query_embeddings, query_labels, _ = get_embeddings_and_paths(model, dl_query)
    gallery_embeddings, gallery_labels, _ = get_embeddings_and_paths(model, dl_gallery)
    
    # Normalize embeddings
    query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
    gallery_embeddings = F.normalize(gallery_embeddings, p=2, dim=1)
    
    # Compute similarity matrix
    similarity = query_embeddings @ gallery_embeddings.T
    
    # Get top-k predictions
    k_list = [1, 10, 20, 30]
    recalls = []
    
    for k in k_list:
        top_k_values, top_k_indices = torch.topk(similarity, k=k, dim=1)
        
        # Calculate recall@k
        correct = 0
        total = 0
        for i, (query_label, top_k_labels) in enumerate(zip(query_labels, top_k_indices)):
            if query_label in gallery_labels[top_k_labels]:
                correct += 1
            total += 1
        
        recall = correct / total if total > 0 else 0
        recalls.append(recall)
        print(f"R@{k}: {recall:.4f}")
    
    return recalls[0]  # Return R@1

# Main evaluation
with torch.no_grad():
    print("**Evaluating...**")
    if args.dataset == 'Inshop':
        Recalls = evaluate_cos_Inshop(model, dl_query, dl_gallery)
        # For Inshop dataset, visualization would need to be modified to handle query-gallery pairs
        if args.visualize:
            print("Visualization for Inshop dataset is not implemented in this version.")
    elif args.dataset != 'SOP':
        Recalls = evaluate_cos(model, dl_ev)
        # Add visualization for non-SOP datasets
        if args.visualize:
            print("\n**Starting Visualization...**")
            visualize_similar_images(model, dl_ev, args.query_index, args.top_k, args.save_viz)
    else:
        Recalls = evaluate_cos_SOP(model, dl_ev)
        # Add visualization for SOP dataset
        if args.visualize:
            print("\n**Starting Visualization...**")
            visualize_similar_images(model, dl_ev, args.query_index, args.top_k, args.save_viz)

print(f"Final R@1: {Recalls:.4f}")

    
