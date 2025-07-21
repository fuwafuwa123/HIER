#!/bin/bash

# Example 1: Train with Proxy Anchor Loss (PA) without hyperbolic geometry
python hier/train.py \
    --model resnet50 \
    --dataset CUB \
    --data_path /path/to/your/data \
    --hyp_c 0 \
    --loss PA \
    --emb 128 \
    --batch_size 90 \
    --epochs 100 \
    --lr 1e-5 \
    --lambda1 1.0 \
    --lambda2 0.0 \
    --run_name "euclidean_pa"

# Example 2: Train with Multi-Similarity Loss (MS) without hyperbolic geometry
python hier/train.py \
    --model resnet50 \
    --dataset CUB \
    --data_path /path/to/your/data \
    --hyp_c 0 \
    --loss MS \
    --emb 128 \
    --batch_size 90 \
    --epochs 100 \
    --lr 1e-5 \
    --lambda1 1.0 \
    --lambda2 0.0 \
    --run_name "euclidean_ms"

# Example 3: Train with SupCon Loss without hyperbolic geometry
python hier/train.py \
    --model resnet50 \
    --dataset CUB \
    --data_path /path/to/your/data \
    --hyp_c 0 \
    --loss SupCon \
    --emb 128 \
    --batch_size 90 \
    --epochs 100 \
    --lr 1e-5 \
    --lambda1 1.0 \
    --lambda2 0.0 \
    --IPC 2 \
    --run_name "euclidean_supcon"

# Example 4: Train with Triplet Loss without hyperbolic geometry
python hier/train.py \
    --model resnet50 \
    --dataset CUB \
    --data_path /path/to/your/data \
    --hyp_c 0 \
    --loss Triplet \
    --emb 128 \
    --batch_size 90 \
    --epochs 100 \
    --lr 1e-5 \
    --lambda1 1.0 \
    --lambda2 0.0 \
    --run_name "euclidean_triplet"

# Example 5: Train with SoftTriple Loss without hyperbolic geometry
python hier/train.py \
    --model resnet50 \
    --dataset CUB \
    --data_path /path/to/your/data \
    --hyp_c 0 \
    --loss SoftTriple \
    --emb 128 \
    --batch_size 90 \
    --epochs 100 \
    --lr 1e-5 \
    --lambda1 1.0 \
    --lambda2 0.0 \
    --run_name "euclidean_softtriple" 