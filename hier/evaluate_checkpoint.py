#!/usr/bin/env python3
"""
Evaluation script to load a trained checkpoint and compute recall and mAP metrics
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
import utils

def get_args_parser():
    parser = argparse.ArgumentParser('Evaluation', add_help=False)
    
    # Model parameters
    parser.add_argument('--model', default='resnet50', type=str,
        choices=['resnet50', 'deit_small_distilled_patch16_224', 'vit_small_patch16_224', 'dino_vits16'],
        help="Model architecture (resnet50, deit_small_distilled_patch16_224, vit_small_patch16_224, dino_vits16)")
    parser.add_argument('--emb', default=128, type=int, help="Embedding dimension")
    parser.add_argument('--hyp_c', type=float, default=0.1, help="Hyperbolic curvature (0 for Euclidean)")
    parser.add_argument('--clip_r', type=float, default=2.3, help="Clip radius for hyperbolic space")
    parser.add_argument('--use_lastnorm', type=bool, default=True, help="Use last normalization")
    parser.add_argument('--bn_freeze', type=bool, default=True, help="Freeze batch norm")
    
    # Dataset parameters
    parser.add_argument('--dataset', default='SOP', type=str, 
                        choices=["SOP", "CUB", "Cars", "Inshop"], help='Dataset to evaluate')
    parser.add_argument('--data_path', default='/kaggle/working/HIER/data', type=str,
        help='Path to the dataset')
    
    # Evaluation parameters
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint file (optional, will use pretrained model if not provided)')
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
    
    return parser

def create_get_emb_function(model, dataset_class, data_path, hyp_c):
    """
    Create a get_emb function for the evaluate function from helpers
    """
    # Set up data transforms
    if model.startswith("vit"):
        mean_std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    elif model == "bn_inception":
        mean_std = (104, 117, 128), (1,1,1)
    else:
        mean_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    
    def get_emb_func(ds_type="eval"):
        return get_emb(
            model=model, ds=dataset_class, path=data_path, 
            mean_std=mean_std, world_size=1, resize=256, crop=224, ds_type=ds_type
        )
    
    return get_emb_func

def compute_metrics(query_embeddings, query_labels, gallery_embeddings=None, gallery_labels=None, k_values=[1, 2, 4, 8]):
    """
    Compute recall@K and mAP metrics
    """
    if gallery_embeddings is not None:
        # For Inshop dataset (query-gallery setup)
        return compute_inshop_metrics(query_embeddings, query_labels, gallery_embeddings, gallery_labels, k_values)
    else:
        # For other datasets (single evaluation set)
        return compute_single_set_metrics(query_embeddings, query_labels, k_values)

def compute_inshop_metrics(query_embeddings, query_labels, gallery_embeddings, gallery_labels, k_values):
    """
    Compute metrics for Inshop dataset (query-gallery setup)
    """
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
            # Find matching gallery images
            matches = (gallery_labels[rankings[i, :k]] == query_label).sum()
            if matches > 0:
                correct += 1
            total += 1
        recalls[f'R@{k}'] = (correct / total) * 100 if total > 0 else 0
    
    # Compute mAP
    # For Inshop, we need to create ground truth structure
    gnd = []
    for i, query_label in enumerate(query_labels):
        # Find all gallery images with same label
        positive_indices = np.where(gallery_labels == query_label)[0]
        gnd.append({'ok': positive_indices, 'junk': []})
    
    map_score, aps, pr, prs = compute_map(rankings.T, gnd, kappas=k_values)
    
    return recalls, map_score, aps

def compute_single_set_metrics(embeddings, labels, k_values):
    """
    Compute metrics for datasets with single evaluation set
    """
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
            # Find matching images (excluding self)
            matches = (labels[rankings[i, :k]] == label).sum()
            if matches > 0:
                correct += 1
            total += 1
        recalls[f'R@{k}'] = (correct / total) * 100 if total > 0 else 0
    
    # Compute mAP
    gnd = []
    for i, label in enumerate(labels):
        # Find all images with same label (excluding self)
        positive_indices = np.where(labels == label)[0]
        positive_indices = positive_indices[positive_indices != i]  # Remove self
        gnd.append({'ok': positive_indices, 'junk': [i]})  # Self is junk
    
    map_score, aps, pr, prs = compute_map(rankings.T, gnd, kappas=k_values)
    
    return recalls, map_score, aps

def visualize_top_k_results(query_embeddings, query_labels, gallery_embeddings, gallery_labels, 
                           query_indices, gallery_indices, dataset_class, data_path, 
                           query_idx=0, top_k=5):
    """
    Visualize top-k retrieval results for a given query
    """
    print(f"Visualizing top-{top_k} results for query index {query_idx}")
    
    # Get query embedding and label
    query_emb = query_embeddings[query_idx:query_idx+1]
    query_label = query_labels[query_idx]
    
    # Normalize embeddings
    query_emb = F.normalize(query_emb, p=2, dim=1)
    gallery_emb = F.normalize(gallery_embeddings, p=2, dim=1)
    
    # Compute similarities
    similarities = torch.mm(query_emb, gallery_emb.t()).squeeze()
    
    # Get top-k indices
    top_k_scores, top_k_indices = torch.topk(similarities, top_k)
    
    # Convert to numpy
    top_k_scores = top_k_scores.cpu().numpy()
    top_k_indices = top_k_indices.cpu().numpy()
    
    # Load images
    images = []
    labels = []
    scores = []
    
    # Load query image
    if dataset_class == Inshop_Dataset:
        query_dataset = dataset_class(data_path, "query", None)
        query_img_path = query_dataset.im_paths[query_idx]
    else:
        eval_dataset = dataset_class(data_path, "eval", None)
        query_img_path = eval_dataset.im_paths[query_idx]
    
    query_img = Image.open(query_img_path).convert('RGB')
    images.append(query_img)
    labels.append(f"Query (Class {query_label})")
    scores.append(1.0)
    
    # Load top-k gallery images
    for i, (idx, score) in enumerate(zip(top_k_indices, top_k_scores)):
        if dataset_class == Inshop_Dataset:
            gallery_dataset = dataset_class(data_path, "gallery", None)
            img_path = gallery_dataset.im_paths[idx]
        else:
            eval_dataset = dataset_class(data_path, "eval", None)
            img_path = eval_dataset.im_paths[idx]
        
        img = Image.open(img_path).convert('RGB')
        images.append(img)
        
        gallery_label = gallery_labels[idx]
        is_correct = (gallery_label == query_label)
        label_text = f"#{i+1} (Class {gallery_label})"
        if is_correct:
            label_text += " ✓"
        else:
            label_text += " ✗"
        
        labels.append(label_text)
        scores.append(score)
    
    # Create visualization
    fig, axes = plt.subplots(1, top_k + 1, figsize=(3 * (top_k + 1), 4))
    if top_k == 0:
        axes = [axes]
    
    for i, (img, label, score) in enumerate(zip(images, labels, scores)):
        ax = axes[i]
        ax.imshow(img)
        ax.set_title(f"{label}", fontsize=10)
        ax.axis('off')
        
        # Add colored border based on correctness
        if i == 0:  # Query image
            color = 'blue'
            linewidth = 3
        elif gallery_labels[top_k_indices[i-1]] == query_label:  # Correct match
            color = 'green'
            linewidth = 2
        else:  # Incorrect match
            color = 'red'
            linewidth = 2
        
        rect = patches.Rectangle((0, 0), img.width, img.height, 
                               linewidth=linewidth, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
    
    plt.tight_layout()
    plt.show()
    
    # Print results
    print(f"\nQuery Class: {query_label}")
    print("Top-k Results:")
    for i, (idx, score) in enumerate(zip(top_k_indices, top_k_scores)):
        gallery_label = gallery_labels[idx]
        is_correct = (gallery_label == query_label)
        status = "✓" if is_correct else "✗"
        print(f"  #{i+1}: Class {gallery_label} {status}")

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
    
    # Load checkpoint if provided, otherwise use pretrained model
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        if os.path.isfile(args.checkpoint):
            checkpoint = torch.load(args.checkpoint, map_location=device)
            
            # Handle different checkpoint formats
            if 'stduent' in checkpoint:
                # Training checkpoint format
                model.load_state_dict(checkpoint['stduent'])
                print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
            elif 'model_state_dict' in checkpoint:
                # Standard format
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                # Assume it's just the model state dict
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
    get_emb_func = create_get_emb_function(args.model, dataset_class, args.data_path, args.hyp_c)
    
    # Use the existing evaluate function from helpers
    recall_at_1 = evaluate(get_emb_func, args.dataset, args.hyp_c)
    
    # For detailed metrics, we still need to get embeddings
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
        
        # Get dataset instances for image paths
        if args.dataset == "Inshop":
            query_dataset = dataset_class(args.data_path, "query", None)
            gallery_dataset = dataset_class(args.data_path, "gallery", None)
            query_indices = list(range(len(query_dataset)))
            gallery_indices = list(range(len(gallery_dataset)))
        else:
            eval_dataset = dataset_class(args.data_path, "eval", None)
            query_indices = list(range(len(eval_dataset)))
            gallery_indices = list(range(len(eval_dataset)))
        
        # Visualize top-k results
        visualize_top_k_results(
            query_embeddings, query_labels, gallery_embeddings, gallery_labels,
            query_indices, gallery_indices, dataset_class, args.data_path,
            query_idx=args.query_index, top_k=args.top_k_viz
        )

if __name__ == "__main__":
    main() 