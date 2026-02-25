import copy
from collections import OrderedDict
from typing import Dict, List
import torch
from torch import Tensor
from math import sqrt

def find_cos_sim(A, B, server):
    """
    A, B: Dict[str, Tensor]

    Computes the cosine similarity between the entire task vectors of two models.

    Returns: float
    """
    A = server.state_dict_to_vector(A)
    B = server.state_dict_to_vector(B)
    return torch.dot(A, B)/(torch.norm(A)*torch.norm(B))

def find_euclidean_sim(A, B, server):
    """
    A, B: Dict[str, Tensor]

    Computes the euclidean distance between the entire task vectors of two models.
    Multiplying by -1 to convert distance to similarity.

    Returns: float
    """
    A = server.state_dict_to_vector(A)
    B = server.state_dict_to_vector(B)
    return -torch.norm(A-B)/ sqrt(A.numel())

def find_sign_sim(A, B, server):
    """
    A, B: Dict[str, Tensor]

    Computes the sign similarity between the entire task vectors of two models.

    Returns: float
    """
    A = server.state_dict_to_vector(A)
    B = server.state_dict_to_vector(B)
    return torch.sum(torch.sign(A) == torch.sign(B))/A.numel()

def find_sim(A, B, server, metric):
    if metric == 'cosine': return find_cos_sim(A, B, server)
    elif metric == 'euclidean': return find_euclidean_sim(A, B, server)
    elif metric == 'sign': return find_sign_sim(A, B, server)
    else: raise ValueError("Invalid metric")

def find_constituent_sim(As, Bs, server, linkage): ### only supporting cosine similarity
    eps = 1e-8

    As_flattened = [server.state_dict_to_vector(A) for A in As]
    Bs_flattened = [server.state_dict_to_vector(B) for B in Bs]
    As_norms = [torch.norm(A) for A in As_flattened]
    Bs_norms = [torch.norm(B) for B in Bs_flattened]

    eps = 1e-8
    all_pairs = [
        torch.dot(A, B) / (norm_A * norm_B + eps)
        for A, norm_A in zip(As_flattened, As_norms)
        for B, norm_B in zip(Bs_flattened, Bs_norms)
    ]
    
    if linkage == 'average':
        return sum(all_pairs)/len(all_pairs)
    elif linkage == 'single':
        return min(all_pairs)
    elif linkage == 'complete':
        return max(all_pairs)
    else:
        raise ValueError(f"Unsupported linkage type: {linkage}")

def init_sim(metric):
    if metric == 'cosine': return -1.1
    elif metric == 'euclidean': return float('-inf')
    elif metric == 'sign': return -0.1
    else: raise ValueError("Invalid metric")

def layerwise_cos_sim(A, B):
    """
    A, B: Dict[str, Tensor]

    Computes the average layerwise task vector cosine similarity between two models.

    Returns: float
    """
    cos_sims = []
    for layer in A:
        layer_A = A[layer].flatten()
        layer_B = B[layer].flatten()
        cos_sim = torch.dot(layer_A, layer_B)/(torch.norm(layer_A)*torch.norm(layer_B))
        cos_sims.append(cos_sim)
    
    return sum(cos_sims)/len(cos_sims)

### For measuring size change for masking based methods
def get_size_change(size_A, size_B, has_pretrained):
    """
    Returns:
        to_remove: int
        to_add: int
        to_add_masks: int
    """
    if size_A == 1 and size_B == 1:
        return 0, 0, 0
    elif (size_A == 2 and size_B == 1) or (size_A == 1 and size_B == 2):
        to_add = 1 if has_pretrained else 2
        return 3, to_add, 3
    elif (size_A > 2 and size_B == 1) or (size_A == 1 and size_B > 2):
        return 1, 0, 1
    elif size_A == 2 and size_B == 2:
        to_add = 1 if has_pretrained else 2
        return 4, to_add, 4
    elif (size_A > 2 and size_B == 2) or (size_A == 2 and size_B > 2):
        return 2, 0, 2
    elif size_A > 2 and size_B > 2:
        return 1, 0, 0

### Defining functions for Channel-Merging
def cosine_distance(u, v):
    """Cosine distance (1 - cosine similarity)"""
    u_norm = u / torch.norm(u)
    v_norm = v / torch.norm(v)
    return 1 - torch.dot(u_norm, v_norm)

def pairwise_cosine_distances(X_stacked, Y):
    """Compute pairwise cosine distances between rows of X_stacked and Y."""
    X_norm = X_stacked / torch.norm(X_stacked, dim=1, keepdim=True)
    Y_norm = Y / torch.norm(Y, dim=1, keepdim=True)
    return 1 - torch.matmul(X_norm, Y_norm.T)

def kmeans_plus_plus_init(X_stacked, n_clusters, random_state=None):
    if random_state is not None:
        torch.manual_seed(random_state)
    n_samples = X_stacked.size(0)
    
    if n_clusters >= n_samples:
        return X_stacked.clone()

    centers = []

    # Pick the first center randomly
    first_idx = torch.randint(n_samples, (1,)).item()
    centers.append(X_stacked[first_idx])

    for _ in range(1, n_clusters):
        # Compute distance to nearest center
        distances = torch.min(pairwise_cosine_distances(X_stacked, torch.stack(centers)), dim=1).values
        probs = distances ** 2
        probs /= probs.sum()

        next_idx = torch.multinomial(probs, 1).item()
        centers.append(X_stacked[next_idx])

    return torch.stack(centers)

def kmeans_cosine(X, n_clusters, max_iters=100, tol=1e-4, random_state=None):
    X_stacked = torch.stack(X)

    if n_clusters >= X_stacked.size(0):
        centers = X_stacked.clone()
        labels = torch.arange(X_stacked.size(0))
        return centers, labels

    # Initialize centers using KMeans++
    centers = kmeans_plus_plus_init(X_stacked, n_clusters, random_state=random_state)

    for iteration in range(max_iters):
        # Assign each point to nearest center
        distances = pairwise_cosine_distances(X_stacked, centers)
        labels = torch.argmin(distances, dim=1)

        # Update centers
        new_centers = []
        for k in range(n_clusters):
            cluster_points = X_stacked[labels == k]
            if cluster_points.size(0) == 0:
                # Empty cluster, reinitialize randomly
                new_center = X_stacked[torch.randint(X_stacked.size(0), (1,)).item()]
            else:
                # Compute mean and renormalize to unit norm (because of cosine)
                new_center = cluster_points.mean(dim=0)
                new_center = new_center / (torch.norm(new_center) + 1e-8)  # avoid div by zero
            new_centers.append(new_center)

        new_centers = torch.stack(new_centers)

        # Check for convergence
        center_shift = torch.norm(centers - new_centers)
        if center_shift < tol:
            break

        centers = new_centers

    return centers, labels

def pairwise_distance_matrix(task_vectors):
    """
    task_vectors: List[Dict[str, Tensor]]
    
    Returns: List[List[float]]
    """
    n = len(task_vectors)
    distance_matrix = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            distance_matrix[i][j] = 1 - layerwise_cos_sim(task_vectors[i], task_vectors[j])
            distance_matrix[j][i] = distance_matrix[i][j]
    
    return distance_matrix

def dare_preprocess(task_vectors, p=0.1, seed2=27):
    """
    task_vectors: List[Dict[str, Tensor]]
    
    Returns: List[Dict[str, Tensor]]
    """
    rng = torch.Generator()
    rng.manual_seed(seed2)
    tvs_flattened = [state_dict_to_vector(tv) for tv in task_vectors]
    tvs_dare = []
    for tv in tvs_flattened:
        # randomly drop p% of the parameters
        mask = torch.rand(tv.size(), generator=rng) > p
        # rescale the remaining parameters by 1/(1-p)
        tv_dare = tv * mask * (1/(1 - p))
        tvs_dare.append(vector_to_state_dict(tv_dare, task_vectors[0]))
    return tvs_dare

def get_blocks_splitted(n_mlp_groups=48) -> List[str]:
    # blocks are in reverse order
    new_blocks = ['ln_post']

    for i in range(12):
        i = 11 - i

        # add mlp with n_mlp_groups groups
        for j in range(n_mlp_groups):
            new_blocks.append(f'resblocks.{i}.mlp.c_proj.g{j}.')
            new_blocks.append(f'resblocks.{i}.mlp.c_fc.g{j}.')
        
        # add ln_2
        new_blocks.append(f'resblocks.{i}.ln_2')

        new_blocks.append(f'resblocks.{i}.attn.out_proj')

        # add attn with 12 heads and 3 groups
        for j in range(12):
            new_blocks.append(f'resblocks.{i}.attn.k.h{j}.')
            new_blocks.append(f'resblocks.{i}.attn.q.h{j}.')
            new_blocks.append(f'resblocks.{i}.attn.v.h{j}.')
                
        # add ln_1
        new_blocks.append(f'resblocks.{i}.ln_1')
    
    beginning_blocks = ['visual.ln_pre', 'visual.conv1', 'visual.proj', 'visual.positional_embedding',
    'visual.class_embedding']

    new_blocks += beginning_blocks

    return new_blocks

def get_transformer_blocks(n_blocks=12):
    blocks = []
    for i in range(n_blocks-1, -1, -1):
        blocks.append(f'resblocks.{i}.mlp')
        blocks.append(f'resblocks.{i}.ln_2')
        blocks.append(f'resblocks.{i}.ln_1')
        blocks.append(f'resblocks.{i}.attn')
    return blocks

def get_t0_3b_blocks():
    blocks = []
    for i in range(23, -1, -1):
        for j in range(2, -1, -1):
            blocks.append(f'transformer.decoder.block.{i}.layer.{j}')

    for i in range(23, -1, -1):
        for j in range(1, -1, -1):
            blocks.append(f'transformer.encoder.block.{i}.layer.{j}')
    return blocks

def get_t5_base_blocks():
    blocks = []
    for i in range(11, -1, -1):
        for j in range(2, -1, -1):
            blocks.append(f'transformer.decoder.block.{i}.layer.{j}')

    for i in range(11, -1, -1):
        for j in range(1, -1, -1):
            blocks.append(f'transformer.encoder.block.{i}.layer.{j}')
            
    blocks += [
        "transformer.decoder.final_layer_norm.weight",
        "transformer.lm_head.weight", 
        "transformer.encoder.final_layer_norm.weight", 
        "transformer.decoder.embed_tokens.weight",
        "transformer.shared.weight",
        "transformer.encoder.embed_tokens.weight"
    ]
    return blocks


def get_t5_large_blocks():
    blocks = []
    for i in range(23, -1, -1):
        for j in range(2, -1, -1):
            blocks.append(f'transformer.decoder.block.{i}.layer.{j}')

    for i in range(23, -1, -1):
        for j in range(1, -1, -1):
            blocks.append(f'transformer.encoder.block.{i}.layer.{j}')
    
    blocks += [
        "transformer.shared.weight",
        "transformer.encoder.embed_tokens.weight",
        "transformer.encoder.final_layer_norm.weight",
        "transformer.decoder.embed_tokens.weight",
        "transformer.decoder.final_layer_norm.weight",
        "transformer.lm_head.weight"
    ]
    return blocks


block_defs = {
    'transformer' : ['layers.11', 'layers.10', 'layers.9', 'layers.8', \
            'layers.7',  'layers.6',  'layers.5', 'layers.4', \
            'layers.3',  'layers.2',  'layers.1', 'layers.0'],

    ### without pre and post layers as blocks for ViT-B-32 in FL
    'transformer+': ['layers.11.self_attn', 'layers.11.layer_norm1', 'layers.11.mlp', 'layers.11.layer_norm2', 
            'layers.10.self_attn', 'layers.10.layer_norm1', 'layers.10.mlp', 'layers.10.layer_norm2',
            'layers.9.self_attn', 'layers.9.layer_norm1', 'layers.9.mlp', 'layers.9.layer_norm2',
            'layers.8.self_attn', 'layers.8.layer_norm1', 'layers.8.mlp', 'layers.8.layer_norm2',
            'layers.7.self_attn', 'layers.7.layer_norm1', 'layers.7.mlp', 'layers.7.layer_norm2',
            'layers.6.self_attn', 'layers.6.layer_norm1', 'layers.6.mlp', 'layers.6.layer_norm2',
            'layers.5.self_attn', 'layers.5.layer_norm1', 'layers.5.mlp', 'layers.5.layer_norm2',
            'layers.4.self_attn', 'layers.4.layer_norm1', 'layers.4.mlp', 'layers.4.layer_norm2',
            'layers.3.self_attn', 'layers.3.layer_norm1', 'layers.3.mlp', 'layers.3.layer_norm2',
            'layers.2.self_attn', 'layers.2.layer_norm1', 'layers.2.mlp', 'layers.2.layer_norm2',
            'layers.1.self_attn', 'layers.1.layer_norm1', 'layers.1.mlp', 'layers.1.layer_norm2',
            'layers.0.self_attn', 'layers.0.layer_norm1', 'layers.0.mlp', 'layers.0.layer_norm2'],
    
    ### transformer+ plus pre and post layers as blocks for ViT-B-32 in FL
    'transformer+PP': ['post_layernorm',
            'layers.11.mlp', 'layers.11.layer_norm2', 'layers.11.layer_norm1', 'layers.11.self_attn',
            'layers.10.mlp', 'layers.10.layer_norm2', 'layers.10.layer_norm1', 'layers.10.self_attn',
            'layers.9.mlp', 'layers.9.layer_norm2', 'layers.9.layer_norm1', 'layers.9.self_attn',
            'layers.8.mlp', 'layers.8.layer_norm2', 'layers.8.layer_norm1', 'layers.8.self_attn',
            'layers.7.mlp', 'layers.7.layer_norm2', 'layers.7.layer_norm1', 'layers.7.self_attn',
            'layers.6.mlp', 'layers.6.layer_norm2', 'layers.6.layer_norm1', 'layers.6.self_attn',
            'layers.5.mlp', 'layers.5.layer_norm2', 'layers.5.layer_norm1', 'layers.5.self_attn',
            'layers.4.mlp', 'layers.4.layer_norm2', 'layers.4.layer_norm1', 'layers.4.self_attn',
            'layers.3.mlp', 'layers.3.layer_norm2', 'layers.3.layer_norm1', 'layers.3.self_attn',
            'layers.2.mlp', 'layers.2.layer_norm2', 'layers.2.layer_norm1', 'layers.2.self_attn',
            'layers.1.mlp', 'layers.1.layer_norm2', 'layers.1.layer_norm1', 'layers.1.self_attn',
            'layers.0.mlp', 'layers.0.layer_norm2', 'layers.0.layer_norm1', 'layers.0.self_attn',
            'pre_layrnorm', 'embeddings'],
    
    ### transformer plus pre and post layers as blocks for ViT-B-32
    'MTL-transformerPP': ['ln_post',
            'resblocks.11', 'resblocks.10', 'resblocks.9', 'resblocks.8', 'resblocks.7',
            'resblocks.6', 'resblocks.5', 'resblocks.4', 'resblocks.3', 'resblocks.2',
            'resblocks.1', 'resblocks.0', 'visual.ln_pre', 'visual.conv1', 'visual.proj', 
            'visual.positional_embedding', 'visual.class_embedding'],            
    
    ### transformer+ plus pre and post layers as blocks for ViT-B-32
    'MTL-transformer+PP': ['ln_post'] + get_transformer_blocks(12) +
            ['visual.ln_pre', 'visual.conv1', 'visual.proj', 'visual.positional_embedding',
            'visual.class_embedding'],

    ### equivalent definition for ViT-L-14
    'MTL-transformer-L+PP': ['ln_post'] + get_transformer_blocks(24) +
            ['visual.ln_pre', 'visual.conv1', 'visual.proj', 'visual.positional_embedding',
            'visual.class_embedding'],

    'MTL-IA3': get_t0_3b_blocks(),
    'MTL-t5-base': get_t5_base_blocks(),
    'MTL-t5-large': get_t5_large_blocks(),
}

def get_blocks(block_gran, n_mlp_groups=48):
    if "splitted" in block_gran: return get_blocks_splitted(n_mlp_groups)
    else: return block_defs[block_gran]

### Code for TIES-MERGING borrowed from: https://github.com/tanganke/peta/blob/main/peta/tasks/ties_merging.py
def normalize(tensor: Tensor, dim: int = 0, eps: float = 1e-8) -> Tensor:
    """
    Normalizes a tensor along a given dimension.

    Args:
        tensor (Tensor): The tensor to normalize.
        dim (int, optional): The dimension along which to normalize the tensor. Defaults to 0.
        eps (float, optional): A small value to add to the denominator to avoid division by zero. Defaults to 1e-8.

    Returns:
        Tensor: The normalized tensor.
    """
    return tensor / torch.clamp(torch.norm(tensor, dim=dim, keepdim=True), min=eps)

def state_dict_to_vector(state_dict: Dict[str, Tensor], remove_keys: List[str] = []):
    R"""
    Converts a PyTorch state dictionary to a 1D tensor.

    Args:
        state_dict (Dict[str, Tensor]): A dictionary containing the state of a PyTorch model.
        remove_keys (List[str], optional): A list of keys to remove from the state dictionary before converting it to a tensor. Defaults to [].

    Returns:
        Tensor: A 1D tensor containing the values of the state dictionary, sorted by key.
    """
    shared_state_dict = copy.deepcopy(state_dict)
    for key in remove_keys:
        if key in shared_state_dict:
            del shared_state_dict[key]
    sorted_shared_state_dict = OrderedDict(sorted(shared_state_dict.items()))
    return torch.nn.utils.parameters_to_vector(
        [value.reshape(-1) for _, value in sorted_shared_state_dict.items()]
    )

def vector_to_state_dict(
    vector: Tensor, state_dict: Dict[str, Tensor], remove_keys: List[str] = []
) -> Dict[str, Tensor]:
    """
    Converts a 1D tensor to a PyTorch state dictionary.

    Args:
        vector (Tensor): A 1D tensor containing the values of the state dictionary, sorted by key.
        state_dict (Dict[str, Tensor]): A dictionary containing the state of a PyTorch model.
        remove_keys (List[str], optional): A list of keys to remove from the state dictionary before converting it to a tensor. Defaults to [].

    Returns:
        Dict[str, Tensor]: A dictionary containing the state of a PyTorch model, with the values of the state dictionary replaced by the values in the input tensor.
    """
    # create a reference dict to define the order of the vector
    reference_dict = copy.deepcopy(state_dict)
    for key in remove_keys:
        if key in reference_dict:
            del reference_dict[key]
    sorted_reference_dict = OrderedDict(sorted(reference_dict.items()))

    # create a shared state dict using the refence dict
    torch.nn.utils.vector_to_parameters(vector, sorted_reference_dict.values())

    return sorted_reference_dict

def topk_values_mask(M: torch.Tensor, K: float = 0.7, return_mask: bool = False):
    """
    Returns a tensor with the top k values of each row of the input tensor M, where k is a fraction of the number of columns.

    Args:
        M (torch.Tensor): A 2D tensor of shape (n, d) where n is the number of rows and d is the number of columns.
        K (float, optional): The fraction of the number of columns to keep. Defaults to 0.7.
        return_mask (bool, optional): Whether to return the mask tensor used to select the top k values. Defaults to False.

    Returns:
        torch.Tensor: A tensor of the same shape as M with the top k values of each row.
        torch.Tensor: A 1D tensor of shape (n,) containing the mean of the mask tensor for each row.
        torch.Tensor: A tensor of the same shape as M with True for the top k values of each row and False otherwise. Only returned if return_mask is True.
    """
    assert M.dim() <= 2, "M must be a 1D or 2D tensor"
    if K > 1:
        K /= 100

    original_shape = M.shape
    if M.dim() == 1:
        M = M.unsqueeze(0)

    _, d = M.shape
    k = int(d * K)
    k = d - k  # Keep top k elements instead of bottom k elements
    if k <= 0:
        k = 1

    # Find the k-th smallest element by magnitude for each row
    # kthvalue: https://pytorch.org/docs/stable/generated/torch.kthvalue.html
    kth_values, _ = M.abs().kthvalue(k, dim=1, keepdim=True)
    # Create a mask tensor with True for the top k elements in each row
    mask: torch.Tensor = M.abs() >= kth_values
    final_mask = mask.squeeze() if original_shape == M.squeeze().shape else mask

    if return_mask:
        return M * final_mask, final_mask.float().mean(dim=1), final_mask
    return M * final_mask, final_mask.float().mean(dim=1)

def resolve_zero_signs(sign_to_mult: Tensor, method="majority"):
    """
    Resolves zero signs in a tensor of signs that will be multiplied together.

    Args:
        sign_to_mult (torch.Tensor): A 1D tensor of signs to be multiplied together.
        method (str, optional): The method to use for resolving zero signs. Can be "majority" or "minority". Defaults to "majority".

    Returns:
        torch.Tensor: A 1D tensor of signs with zero signs resolved according to the specified method.
    """
    majority_sign = torch.sign(sign_to_mult.sum())
    if majority_sign == 0:
        majority_sign = 1

    if method == "majority":
        sign_to_mult[sign_to_mult == 0] = majority_sign
    elif method == "minority":
        sign_to_mult[sign_to_mult == 0] = -1 * majority_sign
    else:
        raise ValueError(f"Method {method} is not defined.")
    return sign_to_mult

def resolve_sign(M: Tensor) -> Tensor:
    """
    Computes the majority sign of the input tensor and resolves zero signs using the "majority" method.

    Args:
        Tensor (torch.Tensor): A 2D tensor of shape (n, d) where n is the number of rows and d is the number of columns.

    Returns:
        torch.Tensor: A 1D tensor of shape (d,) containing the majority sign of each column, with zero signs resolved using the "majority" method.
    """
    sign_to_mult = torch.sign(M.sum(dim=0))  # \gamma_m^p
    sign_to_mult = resolve_zero_signs(sign_to_mult, "majority")
    return sign_to_mult

def disjoint_merge(M: Tensor, merge_func: str, sign_to_mult: Tensor):
    """
    Merges the entries of a tensor M that correspond to disjoint sets.

    Args:
        M (torch.Tensor): A 2D tensor of shape (n, d) where n is the number of sets and d is the number of entries per set.
        merge_func (str): The merge function to use. Can be "mean", "sum", or "max".
        sign_to_mult (torch.Tensor, optional): A 1D tensor of signs to be multiplied with the selected entries. Defaults to None.

    Returns:
        torch.Tensor: A 1D tensor of shape (d,) containing the merged entries.
    """
    # Extract the merge function from the input string
    merge_func = merge_func.split("-")[-1].lower()
    assert merge_func in [
        "mean",
        "sum",
        "max",
    ], f"Merge method {merge_func} is not defined."

    # If sign is provided then we select the corresponding entries and aggregate.
    # If `sign_to_mult` is not None, the function creates a boolean tensor `rows_to_keep` that has the same shape as `M`.
    # The values in `rows_to_keep` are `True` for entries in `M` that have the same sign as the corresponding entry in `sign_to_mult`, and `False` otherwise.
    if sign_to_mult is not None:
        rows_to_keep = torch.where(sign_to_mult.unsqueeze(0) > 0, M > 0, M < 0)
        selected_entries = M * rows_to_keep
    # Else we select all non-zero entries and aggregate.
    else:
        rows_to_keep = M != 0
        selected_entries = M * rows_to_keep

    if merge_func == "mean":
        non_zero_counts = (selected_entries != 0).sum(dim=0).float()
        disjoint_aggs = torch.sum(selected_entries, dim=0) / torch.clamp(
            non_zero_counts, min=1
        )
    elif merge_func == "sum":
        disjoint_aggs = torch.sum(selected_entries, dim=0)
    elif merge_func == "max":
        disjoint_aggs = selected_entries.abs().max(dim=0)[0]
        disjoint_aggs *= sign_to_mult
    else:
        raise ValueError(f"Merge method {merge_func} is not defined.")

    return disjoint_aggs

def ties_merging(
    flat_task_checks: Tensor,
    reset_thresh: float = None,
    merge_func: str = "mean",
    trim: bool = True,
):
    """
    Merges the task checks of a flat tensor using the TIES algorithm.

    Args:
        flat_task_checks (torch.Tensor): A 2D tensor of shape (n, d) where n is the number of tasks and d is the number of checks per task.
        reset_thresh (float, optional): The threshold for resetting the task checks (the top-K% parameters to be keeped, if this is 1, keep all the parameters).
            Should be a float between 0 and 1.
            If None, no resetting is performed. Defaults to None.
        merge_func (str, optional): The merge function to use for aggregating the task checks.
            Can be "mean", "sum", or "max". Defaults to "mean".
        trim (bool, optional): Whether to trim the task checks before merging. Defaults to True.

    Returns:
        torch.Tensor: A 1D tensor of shape (d,) containing the merged task checks.
    """
    all_checks = flat_task_checks.clone()

    # 1. Trim
    if trim:
        updated_checks, *_ = topk_values_mask(all_checks, K=reset_thresh, return_mask=False)
    else:
        updated_checks = all_checks

    # 2. Elect
    final_signs = resolve_sign(updated_checks)
    assert final_signs is not None

    # 3. Disjoint Merge
    merged_tv = disjoint_merge(updated_checks, merge_func, final_signs)

    return merged_tv

### Borrowed from: https://github.com/duguodong7/pcb-merging/blob/main/pcb-merging.py
def normalize(x, dim=0):
    min_values, _ = torch.min(x, dim=dim, keepdim=True)
    max_values, _ = torch.max(x, dim=dim, keepdim=True)
    y = (x - min_values) / (max_values - min_values)
    return y

### Borrowed from: https://github.com/duguodong7/pcb-merging/blob/main/pcb-merging.py
def clamp(x, min_ratio=0, max_ratio=0):
    if len(x.size())==1:
        d = x.size(0)
        sorted_x, _ = torch.sort(x)
        min=sorted_x[int(d * min_ratio)]
        max=sorted_x[int(d * (1-max_ratio)-1)]
    else:
        d = x.size(1)
        sorted_x, _ = torch.sort(x, dim=1)
        min=sorted_x[:, int(d * min_ratio)].unsqueeze(1)
        max=sorted_x[:, int(d * (1-max_ratio)-1)].unsqueeze(1)
    clamped_x= torch.clamp(x, min, max)
    return clamped_x

### Borrowed from: https://github.com/duguodong7/pcb-merging/blob/main/pcb-merging.py
def act(x):
    y = torch.tanh(x)
    return y

### Borrowed from: https://github.com/duguodong7/pcb-merging/blob/main/pcb-merging.py
def pcb_merging(flat_task_checks, K=0.1, trim=True, involved_client_ids=None, client_min_max_dict=None):
    if not trim:
        assert involved_client_ids is not None and client_min_max_dict is not None
    
    all_checks = flat_task_checks.clone()
    n, _ = all_checks.shape   

    all_checks_abs = clamp(torch.abs(all_checks), min_ratio=0.0001, max_ratio=0.0001)
    clamped_all_checks = torch.sign(all_checks)*all_checks_abs
    self_pcb = normalize(all_checks_abs, 1)**2
    self_pcb_act = torch.exp(n*self_pcb)
    cross_pcb = all_checks * torch.sum(all_checks, dim=0)
    cross_pcb_act = act(cross_pcb)
    task_pcb = self_pcb_act * cross_pcb_act
    if trim:
        scale = normalize(clamp(task_pcb, 1-K, 0), dim=1)
    else:
        scale = torch.zeros_like(task_pcb)
        for i, client_id in enumerate(involved_client_ids):
            min_val, max_val = client_min_max_dict[client_id]
            scale[i] = normalize(torch.clamp(task_pcb[i], min_val, max_val))
    tvs = clamped_all_checks
    merged_tv = torch.sum(tvs * scale, dim=0) / torch.clamp(torch.sum(scale, dim=0), min=1e-12)
    return merged_tv

def sparsify(state_dicts: List[Dict], K: float = 0.2) -> List[Dict]:
    """
    Sparsifies a list of state dictionaries by zeroing out the bottom sparsity fraction of the weights.

    Args:
        state_dicts (List[Dict]): A list of state dictionaries.
        sparsity (float, optional): The fraction of weights to zero out. Defaults to 0.2.

    Returns:
        List[Dict]: A list of sparsified state dictionaries.
    """
    task_vectors = [state_dict_to_vector(state_dict) for state_dict in state_dicts]
    task_vectors = torch.stack(task_vectors, dim=0)
    updated_checks, *_ = topk_values_mask(task_vectors, K=K, return_mask=False)
    sparsified_task_vectors = [vector_to_state_dict(tv, state_dicts[0]) for tv in updated_checks]
    return sparsified_task_vectors

def sparsify_pcb(state_dicts: List[Dict], K: float = 0.2) -> List[Dict]:
    all_checks = [state_dict_to_vector(state_dict) for state_dict in state_dicts]
    all_checks = torch.stack(all_checks, dim=0)
    n, d = all_checks.size()
    all_checks_abs = clamp(torch.abs(all_checks), min_ratio=0.0001, max_ratio=0.0001)
    self_pcb = normalize(all_checks_abs, dim=1)**2
    self_pcb_act = torch.exp(n*self_pcb)
    cross_pcb = all_checks * torch.sum(all_checks, dim=0)
    cross_pcb_act = act(cross_pcb)
    task_pcb = self_pcb_act * cross_pcb_act
    sorted_x, _ = torch.sort(task_pcb, dim=1)
    min=sorted_x[:, int(d * K)]
    max=sorted_x[:, int(d - 1)]
    client_min_max_dict = {}
    for i in range(n):
        client_min_max_dict[i] = (min[i].item(), max[i].item())
        print(f"Client {i}: Min: {min[i].item()}, Max: {max[i].item()}")
    return client_min_max_dict

def emr_merge(task_vectors):
    sum_param = {}
    for m in range(len(task_vectors)):
        n2p_temp = task_vectors[m]
        for n in n2p_temp:
            if n not in sum_param:
                sum_param[n] = []
            sum_param[n].append(n2p_temp[n])
    sum_param = {k: torch.stack(v, 0).mean(0) for k, v in sum_param.items()}
    vector_unified = {}
    scales = torch.zeros(len(task_vectors))
    masks = {}
    for n in sum_param:
        masks[n] = []
        flag = (sum_param[n]>0) * 2 - 1
        param_max = torch.zeros_like(task_vectors[0][n])
        for m in range(len(task_vectors)):
            param = task_vectors[m][n]
            mask = (param * flag) > 0
            masks[n].append(mask)
            param_abs = torch.abs(mask*param)
            param_max = torch.where(param_abs>param_max, param_abs, param_max)
            scales[m] += torch.mean(torch.abs(param))
        vector_unified[n] =  param_max * flag
    new_scales = torch.zeros(len(task_vectors))
    for m in range(len(task_vectors)):
        for n in vector_unified:
            p = vector_unified[n] * masks[n][m]
            new_scales[m] += torch.mean(torch.abs(p))
    rescalers = scales / new_scales

    print(f"Rescalers shape: {rescalers.shape} | Rescalers: {rescalers}")

    final_masks = [{} for _ in range(len(task_vectors))]
    for m in range(len(task_vectors)):
        density_rate = 0.0; count = 0
        for n in vector_unified:
            final_masks[m][n] = masks[n][m].float() * rescalers[m]
            density_rate += masks[n][m].sum().item()
            count += masks[n][m].numel()
        print("Density rate: {:.3f}".format(density_rate / count))

    return vector_unified, final_masks

### update in-place
def split_state_dict(sd, n_mlp_groups=48):
    # attn.k.h1.in_proj_weight, attn.k.h2.in_proj_weight, ...
    # attn.q.h1.in_proj_weight, attn.q.h2.in_proj_weight, ...
    # attn.v.h1.in_proj_weight, attn.v.h2.in_proj_weight, ...

    # attn.k.h1.in_proj_bias, attn.k.h2.in_proj_bias, ...
    # attn.q.h1.in_proj_bias, attn.q.h2.in_proj_bias, ...
    # attn.v.h1.in_proj_bias, attn.v.h2.in_proj_bias, ...

    # split the keys into 3 groups and heads
    keys = list(sd.keys())
    n_heads = 12
    to_add = {}
    to_remove = []

    for key in keys:
        if "attn.in_proj_weight" in key:
            k, q, v = sd[key].chunk(3, dim=0)
            k_hs = k.chunk(n_heads, dim=0)
            q_hs = q.chunk(n_heads, dim=0)
            v_hs = v.chunk(n_heads, dim=0)
            for i in range(n_heads):
                to_add[key.replace("attn.in_proj_weight", f"attn.k.h{i}.in_proj_weight")] = k_hs[i]
                to_add[key.replace("attn.in_proj_weight", f"attn.q.h{i}.in_proj_weight")] = q_hs[i]
                to_add[key.replace("attn.in_proj_weight", f"attn.v.h{i}.in_proj_weight")] = v_hs[i]
            
            to_remove.append(key)
        
        elif "attn.in_proj_bias" in key:
            k, q, v = sd[key].chunk(3, dim=0)
            k_hs = k.chunk(n_heads, dim=0)
            q_hs = q.chunk(n_heads, dim=0)
            v_hs = v.chunk(n_heads, dim=0)
            for i in range(n_heads):
                to_add[key.replace("attn.in_proj_bias", f"attn.k.h{i}.in_proj_bias")] = k_hs[i]
                to_add[key.replace("attn.in_proj_bias", f"attn.q.h{i}.in_proj_bias")] = q_hs[i]
                to_add[key.replace("attn.in_proj_bias", f"attn.v.h{i}.in_proj_bias")] = v_hs[i]
            
            to_remove.append(key)

        elif "mlp.c_fc.weight" in key:
            w = sd[key]
            w_groups = w.chunk(n_mlp_groups, dim=0)
            for i in range(n_mlp_groups):
                to_add[key.replace("mlp.c_fc.weight", f"mlp.c_fc.g{i}.weight")] = w_groups[i]
        
            to_remove.append(key)
        
        elif "mlp.c_fc.bias" in key:
            b = sd[key]
            b_groups = b.chunk(n_mlp_groups, dim=0)
            for i in range(n_mlp_groups):
                to_add[key.replace("mlp.c_fc.bias", f"mlp.c_fc.g{i}.bias")] = b_groups[i]
        
            to_remove.append(key)
        
        elif "mlp.c_proj.weight" in key:
            w = sd[key]
            w_groups = w.chunk(n_mlp_groups, dim=0)
            for i in range(n_mlp_groups):
                to_add[key.replace("mlp.c_proj.weight", f"mlp.c_proj.g{i}.weight")] = w_groups[i]
        
            to_remove.append(key)
        
        elif "mlp.c_proj.bias" in key:
            b = sd[key]
            b_groups = b.chunk(n_mlp_groups, dim=0)
            for i in range(n_mlp_groups):
                to_add[key.replace("mlp.c_proj.bias", f"mlp.c_proj.g{i}.bias")] = b_groups[i]
        
            to_remove.append(key)

    for key in to_remove:
        sd.pop(key)

    sd.update(to_add)

### update in-place
def merge_state_dict(sd, n_mlp_groups=48):
    for transformer_block_idx in range(12):
        # attn weights
        k_vals = [sd[f"model.visual.transformer.resblocks.{transformer_block_idx}.attn.k.h{i}.in_proj_weight"] for i in range(12)]
        q_vals = [sd[f"model.visual.transformer.resblocks.{transformer_block_idx}.attn.q.h{i}.in_proj_weight"] for i in range(12)]
        v_vals = [sd[f"model.visual.transformer.resblocks.{transformer_block_idx}.attn.v.h{i}.in_proj_weight"] for i in range(12)]
        weight = torch.cat(k_vals + q_vals + v_vals, dim=0)
        sd[f"model.visual.transformer.resblocks.{transformer_block_idx}.attn.in_proj_weight"] = weight

        # attn bias
        k_vals = [sd[f"model.visual.transformer.resblocks.{transformer_block_idx}.attn.k.h{i}.in_proj_bias"] for i in range(12)]
        q_vals = [sd[f"model.visual.transformer.resblocks.{transformer_block_idx}.attn.q.h{i}.in_proj_bias"] for i in range(12)]
        v_vals = [sd[f"model.visual.transformer.resblocks.{transformer_block_idx}.attn.v.h{i}.in_proj_bias"] for i in range(12)]
        bias = torch.cat(k_vals + q_vals + v_vals, dim=0)
        sd[f"model.visual.transformer.resblocks.{transformer_block_idx}.attn.in_proj_bias"] = bias

        # mlp c_fc weights
        g_vals = [sd[f"model.visual.transformer.resblocks.{transformer_block_idx}.mlp.c_fc.g{i}.weight"] for i in range(n_mlp_groups)]
        weight = torch.cat(g_vals, dim=0)
        sd[f"model.visual.transformer.resblocks.{transformer_block_idx}.mlp.c_fc.weight"] = weight

        # mlp c_fc bias
        g_vals = [sd[f"model.visual.transformer.resblocks.{transformer_block_idx}.mlp.c_fc.g{i}.bias"] for i in range(n_mlp_groups)]
        bias = torch.cat(g_vals, dim=0)
        sd[f"model.visual.transformer.resblocks.{transformer_block_idx}.mlp.c_fc.bias"] = bias

        # mlp c_proj weights
        g_vals = [sd[f"model.visual.transformer.resblocks.{transformer_block_idx}.mlp.c_proj.g{i}.weight"] for i in range(n_mlp_groups)]
        weight = torch.cat(g_vals, dim=0)
        sd[f"model.visual.transformer.resblocks.{transformer_block_idx}.mlp.c_proj.weight"] = weight

        # mlp c_proj bias
        g_vals = [sd[f"model.visual.transformer.resblocks.{transformer_block_idx}.mlp.c_proj.g{i}.bias"] for i in range(n_mlp_groups)]
        bias = torch.cat(g_vals, dim=0)
        sd[f"model.visual.transformer.resblocks.{transformer_block_idx}.mlp.c_proj.bias"] = bias
    
    for k in list(sd.keys()):
        # remove all the h and g keys
        if ".h" in k or ".g" in k:
            sd.pop(k)