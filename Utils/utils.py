import numpy as np
import os
import torch
import scipy.linalg as linalg
import subprocess
import pandas as pd

def cos_sim_vec(a,b):
     return torch.dot(a,b)/(torch.norm(a)*torch.norm(b))

def matrix_sim(A):
    n = len(A)
    matrix_sim = torch.zeros((n,n))
    avg_corr = 0
    avg_norm = 0
    for i in range(n):
        avg_norm += torch.norm(torch.flatten(A[i])).item()
        for j in range(n):
            matrix_sim[i][j] = cos_sim_vec(torch.flatten(A[i]), torch.flatten(A[j]))
            avg_corr += torch.abs(matrix_sim[i][j])
    
    avg_norm = avg_norm/n
    avg_corr = ((avg_corr-n)/(n*(n-1))).item()

    return avg_norm, avg_corr, matrix_sim

def get_gpu_usage():
    try:
        output = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'])
        gpu_usage = [int(x) for x in output.decode('utf-8').strip().split('\n')]
        return gpu_usage
    except subprocess.CalledProcessError as e:
        print(f"Error running nvidia-smi: {e}")
        return None

def rand_mask(N, ratio):
    k = round(N*ratio)
    b = list(range(k))
    return b 

def get_ratio(selected_users):
    total_train_samples = 0
    for user in selected_users:
        total_train_samples += user.train_samples
    return [user.train_samples/total_train_samples for user in selected_users]

def get_gratio(selected_users, groupid, num_cluster):
    total_train_samples = [0 for _ in range(num_cluster)]
    gids = []
    num_cs = [0 for _ in range(num_cluster)]
    for user in selected_users:
        gid = groupid[user.id]
        gids.append(gid)
        num_cs[gid] += 1
        total_train_samples[gid] += user.train_samples
    return [user.train_samples/total_train_samples[gids[i]] for i, user in enumerate(selected_users)], num_cs

def add_data_to_metric(nested_dict, keys, data_pairs):
    for key, value in nested_dict.items():
        if key in keys:
            nested_dict[key].append(data_pairs[keys.index(key)])
        elif isinstance(value, dict):
            add_data_to_metric(value, keys, data_pairs)

def save_results_to_cluster(data, ids, decimal):
    id_loss_dict = {}

    for i, id in enumerate(ids):
        if id not in id_loss_dict:
            id_loss_dict[id] = [data[i]]
        else:
            id_loss_dict[id].append(data[i])
    id_avg_loss_dict = {}

    for id, values in id_loss_dict.items():
        id_avg_loss_dict[id] = round(sum(values) / len(values), decimal)
    output_list = []
    for key, value in id_avg_loss_dict.items():
        # Perform some operation and create a new dictionary
        new_dict = {key: value}

        # Append the new dictionary to the output list
        output_list.append(new_dict)
    return output_list

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
            
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1 : Numpy array containing the activations of the pool_3 layer of the
             inception net ( like returned by the function 'get_predictions')
             for generated samples.
    -- mu2   : The sample mean over activations of the pool_3 layer, precalcualted
               on an representive data set.
    -- sigma1: The covariance matrix over activations of the pool_3 layer for
               generated samples.
    -- sigma2: The covariance matrix over activations of the pool_3 layer,
               precalcualted on an representive data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    covmean = linalg.sqrtm(linalg.sqrtm(sigma1)@sigma2@linalg.sqrtm(sigma1))

    tr_covmean = np.trace(covmean.real)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean  

def torch_save(model, save_path, save_state_dict=True):
    # TODO: hacky way to save state dict
    if save_state_dict and isinstance(model, torch.nn.Module):
        model = model.state_dict()
    if os.path.dirname(save_path) != "":
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model, save_path)

def torch_load(save_path, device=None):
    model = torch.load(save_path, map_location="cpu")
    if device is not None:
        model = model.to(device)
    return model

class LabelSmoothing(torch.nn.Module):
    def __init__(self, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

def assign_learning_rate(param_group, new_lr):
    param_group["lr"] = new_lr

def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length

def cosine_lr(optimizer, base_lrs, warmup_length, steps):
    if not isinstance(base_lrs, list):
        base_lrs = [base_lrs for _ in optimizer.param_groups]
    assert len(base_lrs) == len(optimizer.param_groups)

    def _lr_adjuster(step):
        for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
            if step < warmup_length:
                lr = _warmup_lr(base_lr, warmup_length, step)
            else:
                e = step - warmup_length
                es = steps - warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
            assign_learning_rate(param_group, lr)

    return _lr_adjuster

def read_ft_accuracy(file_path):
    ## csv file
    df = pd.read_csv(file_path)
    cols = df.columns
    # read the row where Method is 'Individual'
    row = df.loc[df['Method'] == 'Individual']
    # get the accuracy for every column
    client_ft_acc_dict = {}
    for col in cols:
        if col != 'Method':
            client_ft_acc_dict[col.lower()] = row[col].values[0]
    return client_ft_acc_dict

def load_snapshot(t, args):
    client_state_dicts = [None for _ in range(args.NUM_CLIENTS)]
    client_state_masks = [None for _ in range(args.NUM_CLIENTS)]
    print(f"==> Loading snapshot {t}")
    for i in range(args.NUM_CLIENTS):
        snapshot_folder = os.path.join(args.SAVEDIR, args.SAVENAME, f"t{t}")
        client_state_dicts[i] = torch.load(os.path.join(snapshot_folder, f"client_{i}_state_dict.pt"), map_location="cpu")
        client_state_masks[i] = torch.load(os.path.join(snapshot_folder, f"client_{i}_state_masks.pt"), map_location="cpu")
    
    return client_state_dicts, client_state_masks
