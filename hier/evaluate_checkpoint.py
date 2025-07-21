#!/usr/bin/env python3
"""
Clean evaluation script for HIER model with visualization
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from PIL import Image

from models.model import init_model
from dataset import CUBirds, SOP, Cars
from dataset.Inshop import Inshop_Dataset
from helpers import get_emb, evaluate
from utils import compute_map, fix_random_seeds

def get_args_parser():
    parser = argparse.ArgumentParser('HIER Evaluation', add_help=False)
    
    # Model parameters
    parser.add_argument('--model', default='resnet50', type=str,
        choices=['resnet50', 'deit_small_distilled_patch16_224', 'vit_small_patch16_224', 'dino_vits16'],
        help="Model architecture")
    parser.add_argument('--emb', default=128, type=int, help="Embedding dimension")
    parser.add_argument('--hyp_c', type=float, default=0.1, help="Hyperbolic curvature")
    parser.add_argument('--clip_r', type=float, default=2.3, help="Clip radius for hyperbolic space")
    parser.add_argument('--use_lastnorm', type=bool, default=True, help="Use last normalization")
    parser.add_argument('--bn_freeze', type=bool, default=True, help="Freeze batch norm")
    parser.add_argument('--pool', default='token', type=str, choices=['token', 'avg'], help='ViT Pooling')
    parser.add_argument('--resize_size', type=int, default=256, help='Resize size for images')
    parser.add_argument('--crop_size', type=int, default=224, help='Crop size for images')
    
    # Dataset parameters
    parser.add_argument('--dataset', default='SOP', type=str, 
                        choices=["SOP", "CUB", "Cars", "Inshop"], help='Dataset to evaluate')
    parser.add_argument('--data_path', default='/kaggle/working/HIER/data', type=str,
        help='Path to the dataset')
    
    # Evaluation parameters
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint file')
    parser.add_argument('--batch_size', default=100, type=int, help='Evaluation batch size')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of data loading workers')
    parser.add_argument('--k_values', nargs='+', type=int, default=[1, 2, 4, 8], 
                        help='K values for recall@K')
    
    # Output parameters
    parser.add_argument('--output_dir', default='./evaluation_results', type=str, 
                        help='Directory to save results')
    parser.add_argument('--visualize', action='store_true', help='Visualize top-k results')
    parser.add_argument('--query_index', default=0, type=int, help='Index of query image to visualize')
    parser.add_argument('--top_k_viz', default=5, type=int, help='Number of top results to visualize')
    parser.add_argument('--save_viz_dir', default=None, type=str, help='Directory to save visualization images')
    
    return parser

def create_get_emb_function(model_name, model_obj, dataset_class, data_path, hyp_c):
    """Create a get_emb function for evaluation"""
    # Set up data transforms
    if model_name.startswith("vit"):
        mean_std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    elif model_name == "bn_inception":
        mean_std = (104, 117, 128), (1,1,1)
    else:
        mean_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    
    def get_emb_func(ds_type="eval"):
        return get_emb(
            model=model_obj, ds=dataset_class, path=data_path, 
            mean_std=mean_std, world_size=1, resize=256, crop=224, ds_type=ds_type
        )
    
    return get_emb_func

def compute_metrics(query_embeddings, query_labels, gallery_embeddings=None, gallery_labels=None, k_values=[1, 2, 4, 8]):
    """Compute recall@K and mAP metrics"""
    if gallery_embeddings is not None:
        return compute_inshop_metrics(query_embeddings, query_labels, gallery_embeddings, gallery_labels, k_values)
    else:
        return compute_single_set_metrics(query_embeddings, query_labels, k_values)

def compute_inshop_metrics(query_embeddings, query_labels, gallery_embeddings, gallery_labels, k_values):
    """Compute metrics for Inshop dataset (query-gallery setup)"""
    print("Computing metrics for Inshop dataset...")
    
    # Normalize embeddings
    query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
    gallery_embeddings = F.normalize(gallery_embeddings, p=2, dim=1)
    
    # Compute similarity matrix
    similarity_matrix = torch.mm(query_embeddings, gallery_embeddings.t())
    
    # Get rankings
    _, rankings = torch.sort(similarity_matrix, dim=1, descending=True)
    
    # Convert to numpy
    rankings = rankings.cpu().numpy()
    query_labels = query_labels.cpu().numpy()
    gallery_labels = gallery_labels.cpu().numpy()
    
    # Compute recall@K
    recalls = {}
    for k in k_values:
        correct = 0
        total = 0
        for i, query_label in enumerate(query_labels):
            matches = (gallery_labels[rankings[i, :k]] == query_label).sum()
            if matches > 0:
                correct += 1
            total += 1
        recalls[f'R@{k}'] = (correct / total) * 100 if total > 0 else 0
    
    # Compute mAP
    gnd = []
    for i, query_label in enumerate(query_labels):
        positive_indices = np.where(gallery_labels == query_label)[0]
        gnd.append({'ok': positive_indices, 'junk': []})
    
    map_score, aps, pr, prs = compute_map(rankings.T, gnd, kappas=k_values)
    
    return recalls, map_score, aps

def compute_single_set_metrics(embeddings, labels, k_values):
    """Compute metrics for datasets with single evaluation set"""
    print("Computing metrics for single evaluation set...")
    
    # Normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    
    # Compute similarity matrix
    similarity_matrix = torch.mm(embeddings, embeddings.t())
    
    # Remove self-similarity
    similarity_matrix.fill_diagonal_(-1)
    
    # Get rankings
    _, rankings = torch.sort(similarity_matrix, dim=1, descending=True)
    
    # Convert to numpy
    rankings = rankings.cpu().numpy()
    labels = labels.cpu().numpy()
    
    # Compute recall@K
    recalls = {}
    for k in k_values:
        correct = 0
        total = 0
        for i, label in enumerate(labels):
            matches = (labels[rankings[i, :k]] == label).sum()
            if matches > 0:
                correct += 1
            total += 1
        recalls[f'R@{k}'] = (correct / total) * 100 if total > 0 else 0
    
    # Compute mAP
    gnd = []
    for i, label in enumerate(labels):
        positive_indices = np.where(labels == label)[0]
        positive_indices = positive_indices[positive_indices != i]
        gnd.append({'ok': positive_indices, 'junk': [i]})
    
    map_score, aps, pr, prs = compute_map(rankings.T, gnd, kappas=k_values)
    
    return recalls, map_score, aps

def get_embeddings_and_paths(model, dataset_class, data_path, ds_type="eval"):
    """Extract embeddings and image paths from the dataset"""
    print(f"Extracting embeddings for {ds_type} set...")
    
    # Set up data transforms
    if model.module.__class__.__name__.startswith("ViT"):
        mean_std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    else:
        mean_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    
    # Get embeddings using existing function
    embeddings, labels, indices = get_emb(
        model=model, ds=dataset_class, path=data_path, 
        mean_std=mean_std, world_size=1, resize=256, crop=224, ds_type=ds_type
    )
    
    # Get image paths
    dataset_instance = dataset_class(data_path, ds_type, None)
    image_paths = dataset_instance.im_paths if hasattr(dataset_instance, 'im_paths') else []
    
    print(f"Extracted embeddings shape: {embeddings.shape}")
    return embeddings, labels, image_paths

def find_top_k_similar(query_embedding, gallery_embeddings, query_idx, k=5):
    """Find top-k most similar embeddings using cosine similarity"""
    # Normalize embeddings
    query_embedding = F.normalize(query_embedding, p=2, dim=1)
    gallery_embeddings = F.normalize(gallery_embeddings, p=2, dim=1)
    
    # Compute cosine similarity
    similarity = torch.mm(query_embedding, gallery_embeddings.t()).squeeze()
    
    # Remove self-similarity
    similarity[query_idx] = float('-inf')
    
    # Get top-k indices
    top_k_scores, top_k_indices = torch.topk(similarity, k=k)
    
    return top_k_scores, top_k_indices

def visualize_top_k_results(query_image_path, top_k_image_paths, top_k_scores, top_k_labels, 
                           query_label, save_path=None):
    """Visualize query image and top-k similar images"""
    fig, axes = plt.subplots(1, len(top_k_image_paths) + 1, figsize=(3 * (len(top_k_image_paths) + 1), 4))
    
    # Load and display query image
    query_img = Image.open(query_image_path).convert('RGB')
    axes[0].imshow(query_img)
    axes[0].set_title(f'Query (Class {query_label})', fontsize=10)
    axes[0].axis('off')
    
    # Add blue border for query image
    rect = patches.Rectangle((0, 0), query_img.width, query_img.height, 
                           linewidth=3, edgecolor='blue', facecolor='none')
    axes[0].add_patch(rect)
    
    # Load and display top-k similar images
    for i, (img_path, score, label) in enumerate(zip(top_k_image_paths, top_k_scores, top_k_labels)):
        img = Image.open(img_path).convert('RGB')
        axes[i+1].imshow(img)
        
        is_correct = (label == query_label)
        title = f'#{i+1} (Class {label})'
        
        axes[i+1].set_title(title, fontsize=10)
        axes[i+1].axis('off')
        
        # Add colored border
        color = 'green' if is_correct else 'red'
        rect = patches.Rectangle((0, 0), img.width, img.height, 
                               linewidth=3, edgecolor=color, facecolor='none')
        axes[i+1].add_patch(rect)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    plt.close(fig)

def visualize_similar_images(model, dataset_class, data_path, query_index=0, top_k=5, save_dir=None):
    """Main function to visualize top-k similar images"""
    print(f"Visualizing top-{top_k} results for query index {query_index}")
    
    # Get embeddings and paths
    embeddings, labels, image_paths = get_embeddings_and_paths(model, dataset_class, data_path, "eval")
    
    # Check query index
    if query_index >= len(embeddings):
        print(f"Query index {query_index} is out of range. Using index 0.")
        query_index = 0
    
    # Get query information
    query_embedding = embeddings[query_index:query_index+1]
    query_label = labels[query_index]
    query_image_path = image_paths[query_index] if query_index < len(image_paths) else f"Image {query_index}"
    
    print(f"Query image: {query_image_path}")
    print(f"Query label: {query_label}")
    
    # Find top-k similar images
    top_k_scores, top_k_indices = find_top_k_similar(query_embedding, embeddings, query_index, k=top_k)
    
    # Convert to numpy
    top_k_scores = top_k_scores.cpu().numpy()
    top_k_indices = top_k_indices.cpu().numpy()
    
    # Get image paths and labels for top-k results
    top_k_image_paths = []
    top_k_labels = []
    for idx in top_k_indices:
        if idx < len(image_paths):
            top_k_image_paths.append(image_paths[idx])
        else:
            top_k_image_paths.append(f"Image {idx}")
        top_k_labels.append(labels[idx])
    
    # Print results
    print(f"\nTop {top_k} similar images:")
    for i, (idx, score, label) in enumerate(zip(top_k_indices, top_k_scores, top_k_labels)):
        img_path = top_k_image_paths[i]
        is_correct = (label == query_label)
        status = "✓" if is_correct else "✗"
        print(f"  {i+1}. Image {idx}: {img_path}")
        print(f"     Label: {label} {status}, Similarity Score: {score:.3f}")
    
    # Visualize results
    if image_paths and all(os.path.exists(path) for path in [query_image_path] + top_k_image_paths):
        print("\nGenerating visualization...")
        
        # Determine save path
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            save_path = os.path.join(save_dir, f'viz_query_{query_index}_top{top_k}.png')
        else:
            save_path = f'viz_query_{query_index}_top{top_k}.png'
        
        visualize_top_k_results(query_image_path, top_k_image_paths, top_k_scores, 
                               top_k_labels, query_label, save_path)
    else:
        print("\nCannot visualize: Some image paths are not valid or images don't exist")

def main():
    parser = argparse.ArgumentParser('HIER Evaluation', parents=[get_args_parser()])
    args = parser.parse_args()
    
    # Fix random seeds
    fix_random_seeds(42)
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model
    print(f"Initializing model: {args.model}")
    model = init_model(args)
    model.eval()
    
    # Load checkpoint if provided
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        if os.path.isfile(args.checkpoint):
            checkpoint = torch.load(args.checkpoint, map_location=device)
            
            # Check if checkpoint is a tuple (embeddings, labels)
            if isinstance(checkpoint, tuple):
                print("Checkpoint contains embeddings and labels, not model weights.")
                print("Using pretrained model for evaluation...")
                checkpoint = None
            else:
                # Handle different checkpoint formats
                if 'stduent' in checkpoint:
                    model.load_state_dict(checkpoint['stduent'])
                    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
                elif 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                
                print("Checkpoint loaded successfully!")
        else:
            print(f"Checkpoint not found: {args.checkpoint}")
            print("Using pretrained model instead...")
    else:
        print("No checkpoint provided, using pretrained model...")
    
    # Set up dataset
    dataset_map = {"CUB": CUBirds, "SOP": SOP, "Cars": Cars, "Inshop": Inshop_Dataset}
    dataset_class = dataset_map[args.dataset]
    
    print(f"Evaluating on {args.dataset} dataset...")
    
    # Create get_emb function for evaluation
    get_emb_func = create_get_emb_function(args.model, model, dataset_class, args.data_path, args.hyp_c)
    
    # Use the existing evaluate function from helpers
    recall_at_1 = evaluate(get_emb_func, args.dataset, args.hyp_c)
    
    # Get embeddings for detailed metrics
    if args.dataset == "Inshop":
        query_embeddings, query_labels, query_indices = get_emb_func("query")
        gallery_embeddings, gallery_labels, gallery_indices = get_emb_func("gallery")
    else:
        eval_embeddings, eval_labels, eval_indices = get_emb_func("eval")
        query_embeddings, gallery_embeddings = eval_embeddings, eval_embeddings
        query_labels, gallery_labels = eval_labels, eval_labels
    
    # Compute detailed metrics
    recalls, map_score, aps = compute_metrics(
        query_embeddings, query_labels, gallery_embeddings, gallery_labels, args.k_values
    )
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Embedding dimension: {args.emb}")
    print(f"Hyperbolic curvature: {args.hyp_c}")
    print(f"Checkpoint: {args.checkpoint if args.checkpoint else 'pretrained'}")
    print("-"*50)
    
    print(f"R@1: {recall_at_1:.2f}%")
    print("-"*50)
    for k, recall in recalls.items():
        print(f"{k}: {recall:.2f}%")
    
    print(f"mAP: {map_score:.4f}")
    print("="*50)
    
    # Save results
    results = {
        'dataset': args.dataset,
        'model': args.model,
        'embedding_dim': args.emb,
        'hyp_c': args.hyp_c,
        'recall_at_1_helpers': recall_at_1,
        'recalls': recalls,
        'mAP': map_score,
        'checkpoint': args.checkpoint if args.checkpoint else 'pretrained'
    }
    
    checkpoint_name = os.path.basename(args.checkpoint).replace('.pth', '').replace('.pt', '') if args.checkpoint else 'pretrained'
    output_file = os.path.join(args.output_dir, f"{args.dataset}_{args.model}_{checkpoint_name}_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    # Visualization
    if args.visualize:
        print("\n" + "="*50)
        print("VISUALIZATION")
        print("="*50)
        visualize_similar_images(model, dataset_class, args.data_path, 
                               args.query_index, args.top_k_viz, args.save_viz_dir)

if __name__ == "__main__":
    main() 