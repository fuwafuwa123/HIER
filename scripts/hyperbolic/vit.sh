CUDA_VISIBLE_DEVICES=0,1 python hier/train.py \
    --lr 1e-5 --epochs 50 --batch_size 90 \
    --hyp_c 0.1 --warmup_epochs 1 \
    --weight_decay 1e-2 --use_fp16 True \
    --clip_grad 1.0 --eval_freq 1 \
    --lambda1 1 --topk 20 --mrg 0.1 \
    --dataset Cars --model vit_small_patch16_224 \
    --IPC 2 --loss PA --emb 128 --use_lastnorm True \
    --optimizer adamw

CUDA_VISIBLE_DEVICES=0,1 python hier/train.py \
    --lr 1e-5 --epochs 50 --batch_size 90 \
    --hyp_c 0.1 --warmup_epochs 1 \
    --weight_decay 1e-2 --use_fp16 True \
    --clip_grad 1.0 --eval_freq 1 \
    --lambda1 1 --topk 20 --mrg 0.1 \
    --dataset CUB --model vit_small_patch16_224 \
    --IPC 2 --loss PA --emb 128 --use_lastnorm True \
    --optimizer adamw

CUDA_VISIBLE_DEVICES=0,1 python hier/train.py \
    --lr 1e-5 --epochs 150 --batch_size 90 \
    --hyp_c 0.1 --warmup_epochs 5 \
    --weight_decay 1e-4 --use_fp16 True \
    --clip_grad 1.0 --eval_freq 1 \
    --lambda1 1 --topk 20 --mrg 0.1 \
    --dataset Inshop --model vit_small_patch16_224 \
    --IPC 2 --loss PA --emb 512 --use_lastnorm True \
    --optimizer adamw --fc_lr_scale 1e2

CUDA_VISIBLE_DEVICES=0,1 python hier/train.py \
    --lr 1e-5 --epochs 150 --batch_size 90 \
    --hyp_c 0.1 --warmup_epochs 5 \
    --weight_decay 1e-4 --use_fp16 True \
    --clip_grad 1.0 --eval_freq 5 \
    --lambda1 1 --topk 20 --mrg 0.1 \
    --dataset SOP --model vit_small_patch16_224 \
    --IPC 2 --loss PA --emb 128  --use_lastnorm True \
    --optimizer adamw --fc_lr_scale 1e2