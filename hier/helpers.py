import torch
import torchvision.transforms as T
import utils
from torch.utils.data import DataLoader
from hyptorch.pmath import dist_matrix
from hyptorch.pmath import mobius_matvec
from utils import Identity, RGBToBGR, ScaleIntensities
import PIL
import multiprocessing
import numpy as np

class MultiSample:
    def __init__(self, transform, n=2):
        self.transform = transform
        self.num = n

    def __call__(self, x):
        return tuple(self.transform(x) for _ in range(self.num))

def evaluate(get_emb_f, ds_name, hyp_c):
    if ds_name != "Inshop":
        emb_head = get_emb_f(ds_type="eval")
        recall_head = get_recall(*emb_head, *emb_head, ds_name, hyp_c)
        
    else:
        emb_head_query = get_emb_f(ds_type="query")
        emb_head_gal = get_emb_f(ds_type="gallery")
        recall_head = get_recall(*emb_head_query, *emb_head_gal, ds_name, hyp_c)
    return recall_head\


def calc_recall_at_k(T, Y, k):
    """
    T : [nb_samples] (target labels)
    Y : [nb_samples x k] (k predicted labels/neighbours)
    """

    s = 0
    for t,y in zip(T,Y):
        if t in torch.Tensor(y).long()[:k]:
            s += 1
    return s / (1. * len(T))


def get_recall(xq, yq, index_q, xg, yg, index_g, ds_name, hyp_c):
    if ds_name == "SOP":
        k_list = [1, 10, 100, 1000]
    elif ds_name == "Inshop":
        k_list = [1, 10, 20, 30]
    else:
        k_list = [1, 2, 4, 8]
    
    def part_dist_and_match(xq, yq, index_q, xg, yg, index_g, k):
        if hyp_c > 0:
            sim = torch.empty(len(xq), len(xg), device="cuda")
            for i in range(len(xq)):
                sim[i : i + 1] = -dist_matrix(xq[i : i + 1], xg, hyp_c)
        else:
            sim = xq @ xg.T
        
        sim_diff_idx = torch.where(index_g != index_q.unsqueeze(-1), sim, -torch.ones_like(sim) * 1e4)
        match_counter = ((yq.unsqueeze(-1) == yg[sim_diff_idx.topk(k)[1]]).sum(1) > 0).sum().item()
        return match_counter
    
    def recall_k(xq, yq, index_q, xg, yg, index_g, k, split_size = 5000):
        match_counter = 0
        splits = range(0, len(xq), split_size)
        if split_size < len(xq):
            for i in range(0, len(splits)-1):
                match_counter += part_dist_and_match(xq[splits[i]:splits[i+1]], yq[splits[i]:splits[i+1]], index_q[splits[i]:splits[i+1]], xg, yg, index_g, k)
        match_counter += part_dist_and_match(xq[splits[-1]:], yq[splits[-1]:], index_q[splits[-1]:], xg, yg, index_g, k)
        return match_counter / len(xq)

    recall = [recall_k(xq, yq, index_q, xg, yg, index_g, k) for k in k_list]
    print(recall)
    return recall[0]


def get_emb(model, ds, path, mean_std, resize=256, crop=224, ds_type="eval", world_size=1, skip_head=False):
    is_inception = mean_std[0][0] > 1
    eval_tr = T.Compose(
        [
            RGBToBGR() if is_inception else Identity(),
            T.Resize(resize, interpolation=PIL.Image.BICUBIC),
            T.CenterCrop(crop),
            T.ToTensor(),
            ScaleIntensities([0, 1], [0, 255]) if is_inception else Identity(),
            T.Normalize(*mean_std),
        ]
    )
    ds_eval = ds(path, ds_type, eval_tr)
    if world_size == 1:
        sampler = None
    else:
        sampler = torch.utils.data.distributed.DistributedSampler(ds_eval)
    dl_eval = DataLoader(
        dataset=ds_eval,
        batch_size=500,
        shuffle=False,
        num_workers=multiprocessing.cpu_count() // world_size,
        pin_memory=True,
        drop_last=False,
        sampler=sampler,
    )
    model.eval()
    x, y, index = eval_dataset(model, dl_eval, skip_head)
    y = y.cuda()
    index = index.cuda()
    if world_size > 1:
        x = utils.all_gather(x)
        y = utils.all_gather(y)
        index = utils.all_gather(index)
    model.train()
    return x, y, index


def eval_dataset(model, dl, skip_head):
    all_x, all_y, all_index = [], [], []
    for x, y, index in dl:
        with torch.no_grad():
            x = x.cuda(non_blocking=True)
            all_x.append(model(x, skip_head=skip_head))
        all_y.append(y)
        all_index.append(index)
    return torch.cat(all_x), torch.cat(all_y), torch.cat(all_index)



def calc_dcg(relevances):
    """Compute Discounted Cumulative Gain (DCG)"""
    relevances = np.asfarray(relevances)
    if relevances.size:
        return np.sum((2**relevances - 1) / np.log2(np.arange(2, relevances.size + 2)))
    return 0.

def calc_ndcg(y_true, y_pred, k=10):
    """
    y_true: [nb_samples] (ground truth labels)
    y_pred: [nb_samples, k] (top-k predicted labels for each sample)
    """
    ndcg_scores = []
    for true, pred in zip(y_true, y_pred):
        relevances = [1 if true == p else 0 for p in pred[:k]]
        dcg = calc_dcg(relevances)
        idcg = calc_dcg(sorted(relevances, reverse=True))
        ndcg = dcg / idcg if idcg > 0 else 0.
        ndcg_scores.append(ndcg)
    return np.mean(ndcg_scores)

def calc_ap(y_true, y_pred, k=10):
    """
    Compute Average Precision for a single sample.
    y_true: int (ground truth label)
    y_pred: [k] (top-k predicted labels)
    """
    hits = 0
    sum_precisions = 0
    for i, p in enumerate(y_pred[:k]):
        if p == y_true:
            hits += 1
            sum_precisions += hits / (i + 1)
    return sum_precisions / hits if hits > 0 else 0.

def calc_map(y_true, y_pred, k=10):
    """
    y_true: [nb_samples] (ground truth labels)
    y_pred: [nb_samples, k] (top-k predicted labels for each sample)
    """
    ap_scores = [calc_ap(t, p, k) for t, p in zip(y_true, y_pred)]
    return np.mean(ap_scores)