CUDA_VISIBLE_DEVICES=0,1 python hier/train.py \
    --lr 1e-4 --epochs 50 --batch_size 90 \
    --hyp_c 0.0 --warmup_epochs 1 \
    --weight_decay 1e-2 --use_fp16 True \
    --clip_grad 1.0 --eval_freq 1 \
    --lambda1 1 --topk 20 --mrg 0.1 \
    --dataset Cars --model resnet50 \
    --IPC 2 --loss PA --emb 512 --bn_freeze True --use_lastnorm True \
    --optimizer adamw --lr_decay 10 --lr_decay_gamma 0.5

CUDA_VISIBLE_DEVICES=0,1 python hier/train.py \
    --lr 1e-4 --epochs 50 --batch_size 90 \
    --hyp_c 0.0 --warmup_epochs 1 \
    --weight_decay 1e-2 --use_fp16 True \
    --clip_grad 1.0 --eval_freq 1 \
    --lambda1 1 --topk 20 --mrg 0.1 \
    --dataset CUB --model resnet50 \
    --IPC 2 --loss PA --emb 512 --bn_freeze True --use_lastnorm True \
    --optimizer adamw --lr_decay 5 --lr_decay_gamma 0.5

CUDA_VISIBLE_DEVICES=0,1 python hier/train.py \
    --lr 6e-4 --epochs 150 --batch_size 90 \
    --hyp_c 0.0 --warmup_epochs 5 \
    --weight_decay 1e-4 --use_fp16 True \
    --clip_grad 1.0 --eval_freq 1 \
    --lambda1 1 --topk 20 --mrg 0.1 \
    --dataset Inshop --model resnet50 \
    --IPC 2 --loss PA --emb 512 --bn_freeze False --use_lastnorm True \
    --optimizer adamw --fc_lr_scale 1 --lr_decay 20 --lr_decay_gamma 0.3 \


CUDA_VISIBLE_DEVICES=0,1 python hier/train.py \
    --lr 6e-4 --epochs 150 --batch_size 90 \
    --hyp_c 0.0 --warmup_epochs 5 \
    --weight_decay 1e-4 --use_fp16 True \
    --clip_grad 1.0 --eval_freq 5 \
    --lambda1 1 --topk 20 --mrg 0.1 \
    --dataset SOP --model resnet50 \
    --IPC 2 --loss PA --emb 512 --bn_freeze False --use_lastnorm True \
    --optimizer adamw --fc_lr_scale 1 --lr_decay 25 --lr_decay_gamma 0.3