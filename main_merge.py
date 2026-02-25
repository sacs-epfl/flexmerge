import torch
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances
import numpy as np
import os
import wandb
from copy import deepcopy
import time

from Utils.server_utils import create_server
from Utils.args_utils import get_args
from Utils.merging_utils import find_sim, get_blocks, sparsify, sparsify_pcb, \
    state_dict_to_vector, split_state_dict, pairwise_distance_matrix, init_sim, \
        find_constituent_sim, kmeans_cosine, get_size_change, dare_preprocess
from Utils.variables_and_paths import get_finetuned_path
from Utils.utils import read_ft_accuracy
from Utils.merging_datastructure import DisjointSetUnionWithTaskVectors, tune_lambda

def wandb_init_metrics(datasets):
    metrics = ["merge_iter", "avg_loss", "avg_acc", "avg_norm_acc", \
               "compression", "n_models", "cos_sim"]
    for dataset in datasets:
        metrics += [f"{dataset}/acc", f"{dataset}/norm_acc"]
    for metric in metrics:
        wandb.define_metric(metric)

def run(args):

    # check if args.save_dir exists
    full_save_dir = os.path.join(args.SAVEDIR, args.SAVENAME)
    if not os.path.exists(full_save_dir):
        os.makedirs(full_save_dir)
    
    # write args to a file
    with open(os.path.join(full_save_dir, 'params.txt'), 'w') as f:
        for key, value in vars(args).items():
            f.write(f'{key}\t:\t{value}\n')
    
    if args.WANDB:
        # initialize wandb
        wandb.init(
            config=args,
            project=args.WANDB_PROJECT, 
            entity=args.WANDB_ENTITY,
            name=args.SAVENAME,
        )

    server = create_server(args)

    ### zero-shot accuracy
    print("="*5, "Evaluating Zero Shot", "="*5)
    server.selected_clients = server.clients
    # server.test('clientside_with_servermodel', clients=server.clients)

    if args.RUN_TYPE == 'train': ### only for federated setups
        print("="*5, "Training Client Models", "="*5)
        assert args.ROUNDS == 1, "ROUNDS should be 1 for training client models"
        for i in range(args.ROUNDS):
            print ("\nTraining Client Models\n")
            server.train(rounds=i, args_print=True)
            ### evaluate the performance of the FedAvg model
            server.test('clientside_with_servermodel', clients=server.clients)
        
        return

    print("="*5, "Evaluating Model Merging", "="*5)

    n = args.NUM_CLIENTS

    for i in range(n):
        print("="*5, f"Loading client {i} model")
        model = deepcopy(server.model)
        if args.METHOD == 'FedAvg':
            model_path = os.path.join(args.MODELDIR, f'client_{i}.pt')
        else:
            model_path = get_finetuned_path(args.MODELDIR, server.clients[i].dataset, args.MODEL)
        model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True), strict=False)
        server.client_state_dicts[i] = model.state_dict()

    server.selected_clients = server.clients

    ### initial ensemble performance
    # server.evaluate_decentralized(args_print=True)

    ### dictionary to store the layers which are not trainable
    not_train_dict = dict((k, p.detach().cpu())
                    for k, p in server.model.named_parameters()
                    if not p.requires_grad) 

    if args.METHOD == 'MTL':
        ### add to not_train_dict the layers which don't start with 'model.visual.'
        to_skip = dict((k, p.detach().cpu())
                    for k, p in server.model.named_parameters()
                    if not k.startswith('model.visual.')) ### these are not used by image_encoder
        print("Layers to skip", to_skip.keys())
        not_train_dict.update(to_skip)
    
    ### remove the layers which are not trainable from the client state dict
    for layer in not_train_dict:
        for i in range(n):
            server.client_state_dicts[i].pop(layer)

    task_vectors_all_layers =[server.state_dict_sub(server.client_state_dicts[client.id], server.model.state_dict(), strict=False) \
                                for client in server.selected_clients]  ### create task vectors for each client
    sparsified_task_vectors_all_layers = None ### sparsified copy to be used for merging

    if args.DARE:
        start = time.time()
        sparsified_task_vectors_all_layers = dare_preprocess(task_vectors_all_layers, args.DARE_P, args.SEED2)
        end = time.time()
        print("Time taken for DARE", end-start, "seconds")

    ### global sparsification for ties and pcb
    if args.ALG != 'cluster-merge' and args.ALG != 'channel-merge': ### for cluster-merge and channel-merge we sparsify within the algorithm
        if args.MERGE_METHOD == 'ties' and args.K < 1.0:
            print("="*5, "Sparsifying Task Vectors using TIES", "="*5)
            sparsified_task_vectors_all_layers = sparsify(task_vectors_all_layers, args.K)
        
        elif args.MERGE_METHOD == 'pcb' and args.K < 1.0:
            print("="*5, "Sparsifying Task Vectors using PCB", "="*5)
            server.client_min_max_dict = sparsify_pcb(task_vectors_all_layers, args.K) ### store in server for use by union method

    if sparsified_task_vectors_all_layers is None: 
        sparsified_task_vectors_all_layers = deepcopy(task_vectors_all_layers)

    ### read ft accuracy to evaluate normalized accuracy
    n1 = 8 if n < 8 else n
    file_path = os.path.join(args.MODELDIR, f'ft-accuracy-n{n1}-{args.MODEL}.csv')
    if not os.path.exists(file_path):
        ### currently raising error, can be relaxed to print a warning
        raise FileNotFoundError(f"Missing ft accuracy file: {file_path}")
    else:
        client_ft_acc_dict = read_ft_accuracy(file_path)
        for k,v in client_ft_acc_dict.items():
            print(k,v)
    
    start = time.time()

    if args.ALG == 'greedy':
        greedy_block_merge(args, server, task_vectors_all_layers, sparsified_task_vectors_all_layers, 
                           client_ft_acc_dict=client_ft_acc_dict)
    
    elif args.ALG == 'left-right':
        structured_block_merge(args, server, task_vectors_all_layers, sparsified_task_vectors_all_layers, 
                               direction='forward', client_ft_acc_dict=client_ft_acc_dict)
    
    elif args.ALG == 'right-left':
        structured_block_merge(args, server, task_vectors_all_layers, sparsified_task_vectors_all_layers, 
                               direction='backward', client_ft_acc_dict=client_ft_acc_dict)
    
    elif args.ALG == 'random':
        random_block_merge(args, server, task_vectors_all_layers, sparsified_task_vectors_all_layers, 
                           client_ft_acc_dict=client_ft_acc_dict)
    
    elif args.ALG == 'cluster-merge':
        cluster_merge(args, server, task_vectors_all_layers, client_ft_acc_dict=client_ft_acc_dict)
    
    elif args.ALG == 'channel-merge':
        channel_merge(args, server, task_vectors_all_layers, client_ft_acc_dict=client_ft_acc_dict)
    
    else:
        raise ValueError("Invalid Algorithm")
    
    end = time.time()
    print("Time taken", end-start, "seconds")

def channel_merge(args, server, task_vectors_all_layers, client_ft_acc_dict=None):
    n = args.NUM_CLIENTS
    print("="*10, f"Number of clients: {n}")
    datasets = [server.clients[i].dataset.lower() for i in range(n)]
    if args.WANDB:
        wandb_init_metrics(datasets)

    if args.MERGE_METHOD == 'ties' and args.K < 1.0:
        ### global sparsify but used only for merging whereas clustering is done on original task vectors
        trimmed_task_vectors_all_layers = sparsify(task_vectors_all_layers, args.K) ### operates on a new copy
    else:
        ### no sparsification for other methods
        trimmed_task_vectors_all_layers = task_vectors_all_layers
    
    base_state_dict = server.model.state_dict()
    
    params_in_model = 0
    for layer in task_vectors_all_layers[0]:
        params_in_model += task_vectors_all_layers[0][layer].numel()

    total_params = n*params_in_model  ### total number of parameters in all the clients
    store_params = total_params  ### total number of parameters that we are storing in the server

    for k in range(1, n+1):
        print("="*5, "K =", k, "="*5)

        for layer in task_vectors_all_layers[0]:
            print(f"==> Layer {layer}")
            layer_task_vectors = [task_vectors_all_layers[i][layer].flatten() for i in range(n)]
            _, labels = kmeans_cosine(layer_task_vectors, n_clusters=k, random_state=args.SEED)
            print("Labels", labels)

            for cluster_idx in range(k):
                clients_in_cluster = [i for i in range(n) if labels[i] == cluster_idx]

                if len(clients_in_cluster) == 1: ### we retain the unsparsified version
                    i = clients_in_cluster[0]
                    server.client_state_dicts[i][layer] = base_state_dict[layer] + task_vectors_all_layers[i][layer]
                    print("Not updating client", clients_in_cluster[0], ", client alone in cluster")
                    continue

                cluster_state_dicts = [{layer: trimmed_task_vectors_all_layers[i][layer]} for i in clients_in_cluster]

                ### Actual merging
                if args.MERGE_METHOD == 'avg' or len(clients_in_cluster) == 1: ### no need to merge when only one client
                    merged_group_task_vector = server.state_dict_avg(cluster_state_dicts)
                elif args.MERGE_METHOD == 'ta':
                    merged_group_task_vector = server.state_dict_avg(cluster_state_dicts)
                    if args.USE_VAL:
                        lambdas = [0.8, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]
                        
                        best_l = tune_lambda(server, merged_group_task_vector, involved_clients=clients_in_cluster, \
                                                n_batches=50, lambdas=lambdas)
                    else:
                        best_l = args.LAMBDA

                    merged_group_task_vector = server.state_dict_mul(merged_group_task_vector, best_l)
                elif args.MERGE_METHOD == 'ties':
                    ### we use global trimming hence trim is False here
                    merged_group_task_vector = server.state_dict_avg_ties(cluster_state_dicts, K=args.K, trim=False)
                    if args.USE_VAL:
                        lambdas = [0.8, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]

                        best_l = tune_lambda(server, merged_group_task_vector, involved_clients=clients_in_cluster, \
                                                n_batches=50, lambdas=lambdas)
                    else:
                        best_l = args.LAMBDA
                    
                    merged_group_task_vector = server.state_dict_mul(merged_group_task_vector, best_l)
                else:
                    raise ValueError("Invalid merge method")
                
                for i in clients_in_cluster:
                    print("Updating client", i)
                    for layer in merged_group_task_vector:
                        server.client_state_dicts[i][layer] = base_state_dict[layer] + merged_group_task_vector[layer]

        store_params = k*params_in_model
    
        avg_loss, avg_acc, client_acc_dict = server.evaluate_decentralized(args_print=True)
        client_norm_acc_dict = {}
        for dset in client_acc_dict:
            if client_ft_acc_dict is None:
                client_norm_acc_dict[dset] = client_acc_dict[dset]
            else:
                client_norm_acc_dict[dset] = client_acc_dict[dset] / client_ft_acc_dict[dset]
        
        metric_dict = {
            "avg_loss": avg_loss,
            "avg_acc": avg_acc,
            "avg_norm_acc": sum(client_norm_acc_dict.values()) / len(client_norm_acc_dict),
            "compression": total_params/store_params,
            "merge_iter": k,
            "n_models": k
        }
        
        print("Avg Norm. Acc", metric_dict["avg_norm_acc"])

        for dataset in datasets:
            metric_dict[f"{dataset}/acc"] = client_acc_dict[dataset]
            metric_dict[f"{dataset}/norm_acc"] = client_norm_acc_dict[dataset]

        if args.WANDB: 
            wandb.log(metric_dict)

    print("="*5, "Run Complete", "="*5)

def cluster_merge(args, server, task_vectors_all_layers, client_ft_acc_dict=None):
    n = args.NUM_CLIENTS
    print("="*10, f"Number of clients: {n}")
    datasets = [server.clients[i].dataset.lower() for i in range(n)]
    if args.WANDB:
        wandb_init_metrics(datasets)

    base_state_dict = server.model.state_dict()
    
    params_in_model = 0
    for layer in task_vectors_all_layers[0]:
        params_in_model += task_vectors_all_layers[0][layer].numel()

    total_params = n*params_in_model  ### total number of parameters in all the clients
    store_params = total_params  ### total number of parameters that we are storing in the server

    if args.LAYERWISE_DISTANCE:
        cosine_dist_matrix = pairwise_distance_matrix(task_vectors_all_layers) ### layerwise distance, own implementation
    else:
        tvs = [state_dict_to_vector(task_vectors_all_layers[i]) for i in range(n)]
        tvs = torch.stack(tvs, dim=0).numpy()
        cosine_dist_matrix = cosine_distances(tvs) ### non-layerwise distance, sklearn implementation

    for k in range(1, n+1):
        print("="*5, "K =", k, "="*5)

        if k == n:
            labels = [i for i in range(n)]
        else:
            # cluster using heirarchical clustering
            model = AgglomerativeClustering(metric='precomputed', linkage=args.LINKAGE, n_clusters=k)
            labels = model.fit_predict(cosine_dist_matrix)
        
        print("Labels", labels)
            
        ### get cluster centers
        for cluster_idx in range(k):
            ### get the indices of the clients in the cluster
            clients_in_cluster = [i for i in range(n) if labels[i] == cluster_idx]
            client_objects = [server.clients[i] for i in clients_in_cluster]

            # print datasets that got grouped together
            print("Cluster", [client.dataset for client in client_objects])

            ### get the state dicts of the clients in the cluster
            cluster_state_dicts = [task_vectors_all_layers[i] for i in clients_in_cluster]

            if args.MERGE_METHOD == 'avg' or len(clients_in_cluster) == 1: ### no need to merge when only one client
                merged_group_task_vector = server.state_dict_avg(cluster_state_dicts)
            elif args.MERGE_METHOD == 'ta':
                merged_group_task_vector = server.state_dict_avg(cluster_state_dicts)
                if args.USE_VAL:
                    lambdas = [0.8, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]
                    
                    best_l = tune_lambda(server, merged_group_task_vector, involved_clients=clients_in_cluster, \
                                            n_batches=50, lambdas=lambdas)
                else:
                    best_l = args.LAMBDA

                merged_group_task_vector = server.state_dict_mul(merged_group_task_vector, best_l)
            elif args.MERGE_METHOD == 'ties':
                ### we use global trimming hence trim is False here
                merged_group_task_vector = server.state_dict_avg_ties(cluster_state_dicts, K=args.K)
                if args.USE_VAL:
                    lambdas = [0.8, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]

                    best_l = tune_lambda(server, merged_group_task_vector, involved_clients=clients_in_cluster, \
                                            n_batches=50, lambdas=lambdas)
                else:
                    best_l = args.LAMBDA
                
                merged_group_task_vector = server.state_dict_mul(merged_group_task_vector, best_l)
            elif args.MERGE_METHOD == 'pcb':
                ### we use global trimming hence trim is False here
                merged_group_task_vector = server.state_dict_pcb(cluster_state_dicts, K=args.K)
                if args.USE_VAL:
                    lambdas = [0.8, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]

                    best_l = tune_lambda(server, merged_group_task_vector, involved_clients=clients_in_cluster, \
                                            n_batches=50, lambdas=lambdas)
                else:
                    best_l = args.LAMBDA
                
                merged_group_task_vector = server.state_dict_mul(merged_group_task_vector, best_l)
            else:
                raise ValueError("Invalid merge method")

            for i in clients_in_cluster:
                print ("Updating client", i)
                for layer in merged_group_task_vector:
                    server.client_state_dicts[i][layer] = base_state_dict[layer] + merged_group_task_vector[layer]

        store_params = k*params_in_model
            
        avg_loss, avg_acc, client_acc_dict = server.evaluate_decentralized(args_print=True)
        client_norm_acc_dict = {}
        for dset in client_acc_dict:
            if client_ft_acc_dict is None:
                client_norm_acc_dict[dset] = client_acc_dict[dset]
            else:
                client_norm_acc_dict[dset] = client_acc_dict[dset] / client_ft_acc_dict[dset]
        
        metric_dict = {
            "avg_loss": avg_loss,
            "avg_acc": avg_acc,
            "avg_norm_acc": sum(client_norm_acc_dict.values()) / len(client_norm_acc_dict),
            "compression": total_params/store_params,
            "merge_iter": k,
            "n_models": k
        }
        
        for dataset in datasets:
            metric_dict[f"{dataset}/acc"] = client_acc_dict[dataset]
            metric_dict[f"{dataset}/norm_acc"] = client_norm_acc_dict[dataset]

        if args.WANDB: 
            wandb.log(metric_dict)

    print("="*5, "Run Complete", "="*5)

def structured_block_merge(args, server, task_vectors_all_layers, sparsified_task_vectors_all_layers, 
                           direction='forward', client_ft_acc_dict=None):
    n = args.NUM_CLIENTS
    datasets = [server.clients[i].dataset.lower() for i in range(n)]
    if args.WANDB:
        wandb_init_metrics(datasets)

    base_state_dict = server.model.state_dict()

    total_params = 0
    for layer in task_vectors_all_layers[0]:
        total_params += task_vectors_all_layers[0][layer].numel()

    total_params = n*total_params  ### total number of parameters in all the clients
    store_params = total_params  ### total number of parameters that we are storing in the server

    blocks = get_blocks(args.BLOCK_GRANULARITY, args.N_MLP_GROUPS) ### blocks to be fused
    if direction == 'forward': ### blocks defs are backward
        blocks = blocks[::-1]

    if "splitted" in args.BLOCK_GRANULARITY:  ### split layers further
        for i in range(n):
            split_state_dict(task_vectors_all_layers[i], args.N_MLP_GROUPS)
            split_state_dict(server.client_state_dicts[i], args.N_MLP_GROUPS)
        split_state_dict(base_state_dict, args.N_MLP_GROUPS)

    task_vectors = [{} for _ in range(n)]
    sparsified_task_vectors = [{} for _ in range(n)]
    for block in blocks:
        for i in range(n):
            task_vectors[i][block] = {}  ### create task vectors for each block and each client
            sparsified_task_vectors[i][block] = {}

    for layer in task_vectors_all_layers[0]:
        find_block = None
        for block in blocks:
            if block in layer:
                find_block = block
                break
        print(f"{layer} exists in block {find_block}")
        if find_block is None:
            continue
        for i in range(n):
            ### assign the task vectors to the respective blocks
            task_vectors[i][find_block][layer] = task_vectors_all_layers[i][layer]
            sparsified_task_vectors[i][find_block][layer] = sparsified_task_vectors_all_layers[i][layer]
    
    T = args.MERGE_ITER
    n = args.NUM_CLIENTS
    T_max = (n - 1) * len(blocks) ### maximum number of iterations
    T = min(T, T_max)
    T_TUNE_START = int(args.TUNE_START * T)
    EVAL_AFTER = int(args.EVAL_AFTER * T)
    print("Total iterations: ", T)
    eval_freq = max(int(0.05 * T), 1) ### evaluate the performance every 5% of the iterations
    dsu_blocks = {}
    for block in blocks:
        ### create disjoint set union data structure to store the clusters in each block
        dsu_blocks[block] = DisjointSetUnionWithTaskVectors(
            n, 
            [task_vectors[i][block] for i in range(n)], 
            [sparsified_task_vectors[i][block] for i in range(n)],
            server,
            args.MERGE_METHOD, 
            args.K, 
            args.N_VAL_BATCHES, 
            args.LAMBDA, 
            args.CONSENSUS_LAMBDA
        )

    cur_block_index = 0
    has_pretrained_block = {block: False for block in blocks}
    for t in range(T):
        to_tune = True if t >= T_TUNE_START and args.USE_VAL else False
        ### block where fusion is occuring
        block_max = blocks[cur_block_index]
        print("Running block:", block_max)
        
        i_max, j_max, max_cos_sim = 0, 0, -1.1
        unique_groups = list(dsu_blocks[block_max].find_unique_groups())  ### find clusters in each block
        print(unique_groups)
        n1 = len(unique_groups)
        for i in range(n1):
            for j in range(i+1,n1): ### iterate over all pairs of clusters
                A = dsu_blocks[block_max].get_group_task_vector(unique_groups[i])  
                B = dsu_blocks[block_max].get_group_task_vector(unique_groups[j])
                cos_sim = find_sim(A, B, server, args.DISTANCE)
                if cos_sim > max_cos_sim: ### find the clusters with maximum similarity
                    i_max, j_max, max_cos_sim = unique_groups[i], unique_groups[j], cos_sim
                
        print("Max cosine sim", max_cos_sim)
        print("Iteration ", t, "Fused Block Layer ", block_max, "Fused block similarity" , max_cos_sim.item(), "Fused Group indices" , (i_max, j_max))
        
        size_i_max = dsu_blocks[block_max].get_group_size(i_max)
        size_j_max = dsu_blocks[block_max].get_group_size(j_max)
        to_remove, to_add, to_add_masks = get_size_change(size_i_max, size_j_max, has_pretrained_block[block_max])
        if size_i_max + size_j_max > 2: has_pretrained_block[block_max] = True
        
        dsu_blocks[block_max].union(i_max, j_max, to_tune)  ### merge the clusters with maximum similarity
        merged_group = dsu_blocks[block_max].find(i_max) ### identity of the merged cluster
        merged_group_task_vector = dsu_blocks[block_max].get_group_task_vector(i_max) ### get the task vector of the merged cluster
        
        if args.MERGE_METHOD in ['consensus_ties', 'consensus_ta', 'emr']:
            for layer in merged_group_task_vector:
                store_params -= merged_group_task_vector[layer].numel() * to_remove  
                store_params += merged_group_task_vector[layer].numel() * to_add
                store_params += merged_group_task_vector[layer].numel() * to_add_masks * 1/32 ### Assuming 32-bit model
        else:
            for layer in merged_group_task_vector:
                store_params -= merged_group_task_vector[layer].numel()  ### update the number of parameters stored in the server

        if not ((args.MERGE_METHOD in ['consensus_ties', 'consensus_ta', 'emr']) and (size_i_max + size_j_max <= 2)):
            for i in range(n):
                if dsu_blocks[block_max].find(i) == merged_group:
                        print ("Updating client ", i)
                        for layer in merged_group_task_vector:
                            server.client_state_dicts[i][layer] = base_state_dict[layer] + merged_group_task_vector[layer]  ### update the client models in the merged cluster with the merged task vector
        else:
            print(f'==> Skipping update for clients in block {block_max} as the size is less than 2')

        if (t % eval_freq == 1 and t > EVAL_AFTER) or t == T - 1:
            if not args.NO_EVAL: ### evaluation can be explicitly turned off
                avg_loss, avg_acc, client_acc_dict = server.evaluate_decentralized(args_print=True)
                client_norm_acc_dict = {}
                for k in client_acc_dict:
                    if client_ft_acc_dict is None:
                        client_norm_acc_dict[k] = client_acc_dict[k]
                    else:
                        client_norm_acc_dict[k] = client_acc_dict[k] / client_ft_acc_dict[k]
                
                metric_dict = {
                    "avg_loss": avg_loss,
                    "avg_acc": avg_acc,
                    "avg_norm_acc": sum(client_norm_acc_dict.values()) / len(client_norm_acc_dict),
                    "compression": total_params/store_params,
                    "merge_iter": t,
                    "cos_sim": max_cos_sim.item(),
                    "n_models": n * store_params / total_params
                }
                
                for dataset in datasets:
                    metric_dict[f"{dataset}/acc"] = client_acc_dict[dataset]
                    metric_dict[f"{dataset}/norm_acc"] = client_norm_acc_dict[dataset]
                
                if args.WANDB:
                    wandb.log(metric_dict)
            
            if args.SNAPSHOT:
                server.save_snapshot()

        ### all pairs are merged, move to the next block
        if len(list(dsu_blocks[block_max].find_unique_groups())) == 1:
            cur_block_index += 1
        
    print("="*5, "Run Complete", "="*5)

def greedy_block_merge(args, server, task_vectors_all_layers, sparsified_task_vectors_all_layers, 
                       client_ft_acc_dict=None):
    n = args.NUM_CLIENTS
    datasets = [server.clients[i].dataset.lower() for i in range(n)]
    if args.WANDB:
        wandb_init_metrics(datasets)

    base_state_dict = server.model.state_dict()
        
    total_params = 0
    for layer in task_vectors_all_layers[0]:
        total_params += task_vectors_all_layers[0][layer].numel()

    total_params = n*total_params  ### total number of parameters in all the clients
    store_params = total_params  ### total number of parameters that we are storing in the server

    blocks = get_blocks(args.BLOCK_GRANULARITY, args.N_MLP_GROUPS) ### blocks to be fused

    if "splitted" in args.BLOCK_GRANULARITY:  ### split layers further
        for i in range(n):
            split_state_dict(task_vectors_all_layers[i], args.N_MLP_GROUPS)
            split_state_dict(server.client_state_dicts[i], args.N_MLP_GROUPS)
        split_state_dict(base_state_dict, args.N_MLP_GROUPS)

    task_vectors = [{} for _ in range(n)]
    sparsified_task_vectors = [{} for _ in range(n)]
    for block in blocks:
        for i in range(n):
            task_vectors[i][block] = {}  ### create task vectors for each block and each client
            sparsified_task_vectors[i][block] = {}

    for layer in task_vectors_all_layers[0]:
        find_block = None
        for block in blocks:
            if block in layer:
                find_block = block
                break
        print(f"{layer} exists in block {find_block}")
        if find_block is None:
            continue
        for i in range(n):
            ### assign the task vectors to the respective blocks
            task_vectors[i][find_block][layer] = task_vectors_all_layers[i][layer]
            sparsified_task_vectors[i][find_block][layer] = sparsified_task_vectors_all_layers[i][layer]

    T = args.MERGE_ITER
    n = args.NUM_CLIENTS
    T_max = (n - 1) * len(blocks) ### maximum number of iterations
    T = min(T, T_max)
    T_TUNE_START = int(args.TUNE_START * T)
    EVAL_AFTER = int(args.EVAL_AFTER * T)
    print("Total iterations: ", T)
    eval_freq = max(int(0.05 * T), 1) ### evaluate the performance every 5% of the iterations
    dsu_blocks = {}
    print("Using sparsified task vectors for both selection and merging")
    for block in blocks:
        ### create disjoint set union data structure to store the clusters in each block
        dsu_blocks[block] = DisjointSetUnionWithTaskVectors(
            n, 
            [sparsified_task_vectors[i][block] for i in range(n)],
            [sparsified_task_vectors[i][block] for i in range(n)],
            server,
            args.MERGE_METHOD, 
            args.K, 
            args.N_VAL_BATCHES, 
            args.LAMBDA, 
            args.CONSENSUS_LAMBDA
        )

    block_max, blockwise_best = None, {}
    has_pretrained_block = {block: False for block in blocks}
    for t in range(T):
        to_tune = True if t >= T_TUNE_START and args.USE_VAL else False
        if block_max is not None:
            blocks_to_update = [block_max] ### update only the block with maximum similarity in the previous iteration
        else:
            blocks_to_update = blocks ### update all the blocks in the first iteration

        for block in blocks_to_update:
            print("Running block:", block)
            i_max, j_max, max_cos_sim = 0, 0, init_sim(args.DISTANCE)
            unique_groups = list(dsu_blocks[block].find_unique_groups())  ### find clusters in each block
            print(unique_groups)
            n1 = len(unique_groups)
            for i in range(n1):
                for j in range(i+1,n1): ### iterate over all pairs of clusters
                    
                    if args.USE_CONSTITUENTS_DISTANCE: ### compute distance on the constituent task vectors
                        As = dsu_blocks[block].get_constituent_task_vectors(unique_groups[i])
                        Bs = dsu_blocks[block].get_constituent_task_vectors(unique_groups[j])
                        cos_sim = find_constituent_sim(As, Bs, server, args.LINKAGE) ### only supports cosine
                    else: ### compute distance on the merged task vectors
                        A = dsu_blocks[block].get_group_task_vector(unique_groups[i])  
                        B = dsu_blocks[block].get_group_task_vector(unique_groups[j])
                        cos_sim = find_sim(A, B, server, args.DISTANCE)

                    if cos_sim > max_cos_sim: ### find the clusters with maximum similarity
                        i_max, j_max, max_cos_sim = unique_groups[i], unique_groups[j], cos_sim
            if n1 > 1: ### merge possible only if there is more than one cluster in the block
                blockwise_best[block] = (i_max, j_max, max_cos_sim)
            elif block in blockwise_best: ### no merge possible, delete the block from the dictionary
                del blockwise_best[block]     
                
        ### find the block with maximum similarity
        block_max = max(blockwise_best, key=lambda x: blockwise_best[x][2])
        ### get the indices of the clusters with maximum similarity
        i_max, j_max, max_cos_sim = blockwise_best[block_max]
        print("Iteration ", t, "Fused Block Layer ", block_max, "Fused block similarity" , max_cos_sim.item(), "Fused Group indices" , (i_max, j_max))

        size_i_max = dsu_blocks[block_max].get_group_size(i_max)
        size_j_max = dsu_blocks[block_max].get_group_size(j_max)
        to_remove, to_add, to_add_masks = get_size_change(size_i_max, size_j_max, has_pretrained_block[block_max])
        if size_i_max + size_j_max > 2: has_pretrained_block[block_max] = True

        dsu_blocks[block_max].union(i_max, j_max, to_tune)  ### merge the clusters with maximum similarity
        merged_group = dsu_blocks[block_max].find(i_max) ### identity of the merged cluster
        merged_group_task_vector = dsu_blocks[block_max].get_group_task_vector(i_max) ### get the task vector of the merged cluster

        if args.MERGE_METHOD in ['consensus_ties', 'consensus_ta', 'emr']:
            for layer in merged_group_task_vector:
                store_params -= merged_group_task_vector[layer].numel() * to_remove  
                store_params += merged_group_task_vector[layer].numel() * to_add
                store_params += merged_group_task_vector[layer].numel() * to_add_masks * 1/32 ### Assuming 32-bit model
        else:
            for layer in merged_group_task_vector:
                store_params -= merged_group_task_vector[layer].numel()  ### update the number of parameters stored in the server

        if not ((args.MERGE_METHOD in ['consensus_ties', 'consensus_ta', 'emr']) and (size_i_max + size_j_max <= 2)):
            for i in range(n):
                if dsu_blocks[block_max].find(i) == merged_group:
                        print ("Updating client ", i)
                        for layer in merged_group_task_vector:
                            server.client_state_dicts[i][layer] = base_state_dict[layer] + merged_group_task_vector[layer]  ### update the client models in the merged cluster with the merged task vector
        else:
            print(f'==> Skipping update for clients in block {block_max} as the size is less than 2')

        if (t % eval_freq == 1 and t > EVAL_AFTER) or t == T - 1:
            if not args.NO_EVAL: ### evaluation can be explicitly turned off
                avg_loss, avg_acc, client_acc_dict = server.evaluate_decentralized(args_print=True)
                client_norm_acc_dict = {}
                for k in client_acc_dict:
                    if client_ft_acc_dict is None:
                        client_norm_acc_dict[k] = client_acc_dict[k]
                    else:
                        client_norm_acc_dict[k] = client_acc_dict[k] / client_ft_acc_dict[k]
                
                metric_dict = {
                    "avg_loss": avg_loss,
                    "avg_acc": avg_acc,
                    "avg_norm_acc": sum(client_norm_acc_dict.values()) / len(client_norm_acc_dict),
                    "compression": total_params/store_params,
                    "merge_iter": t,
                    "cos_sim": max_cos_sim.item(),
                    "n_models": n * store_params / total_params
                }
                
                for dataset in datasets:
                    metric_dict[f"{dataset}/acc"] = client_acc_dict[dataset]
                    metric_dict[f"{dataset}/norm_acc"] = client_norm_acc_dict[dataset]
                
                if args.WANDB:
                    wandb.log(metric_dict)

            if args.SNAPSHOT:
                server.save_snapshot()
            
            if args.MEMORY_EFFICIENT_SNAPSHOT:
                print("==> Saving snapshot")
                snapshot = {}
                for block in blocks:
                    unique_groups = list(dsu_blocks[block].find_unique_groups())
                    n_groups = len(unique_groups)    
                    group_to_index_mapping = {g:i for i, g in enumerate(unique_groups)}
                    node_to_index_mapping = {i: group_to_index_mapping[dsu_blocks[block].find(i)] for i in range(n)}
                    merged_task_vectors = [None for _ in range(n_groups)]
                    for g in unique_groups:
                        merged_task_vector = dsu_blocks[block].get_group_task_vector(g)
                        merged_task_vectors[group_to_index_mapping[g]] = merged_task_vector

                    snapshot[block] = (node_to_index_mapping, merged_task_vectors)
                
                save_folder = os.path.join(args.SAVEDIR, args.SAVENAME, "s_" + str(n * store_params / total_params))
                os.makedirs(save_folder, exist_ok=True)
                torch.save(snapshot, os.path.join(save_folder, f"snapshot.pt"))
        
    print("="*5, "Run Complete", "="*5)

def random_block_merge(args, server, task_vectors_all_layers, sparsified_task_vectors_all_layers, 
                       client_ft_acc_dict=None):
    rng = np.random.default_rng(seed=args.SEED2)
    
    n = args.NUM_CLIENTS
    datasets = [server.clients[i].dataset.lower() for i in range(n)]
    if args.WANDB:
        wandb_init_metrics(datasets)

    base_state_dict = server.model.state_dict()
        
    total_params = 0
    for layer in task_vectors_all_layers[0]:
        total_params += task_vectors_all_layers[0][layer].numel()

    total_params = n*total_params  ### total number of parameters in all the clients
    store_params = total_params  ### total number of parameters that we are storing in the server

    blocks = get_blocks(args.BLOCK_GRANULARITY, args.N_MLP_GROUPS) ### blocks to be fused

    if "splitted" in args.BLOCK_GRANULARITY:  ### split layers further
        for i in range(n):
            split_state_dict(task_vectors_all_layers[i], args.N_MLP_GROUPS)
            split_state_dict(server.client_state_dicts[i], args.N_MLP_GROUPS)
        split_state_dict(base_state_dict, args.N_MLP_GROUPS)

    task_vectors = [{} for _ in range(n)]
    sparsified_task_vectors = [{} for _ in range(n)]
    for block in blocks:
        for i in range(n):
            task_vectors[i][block] = {}  ### create task vectors for each block and each client
            sparsified_task_vectors[i][block] = {}

    for layer in task_vectors_all_layers[0]:
        find_block = None
        for block in blocks:
            if block in layer:
                find_block = block
                break
        print(f"{layer} exists in block {find_block}")
        if find_block is None:
            continue
        for i in range(n):
            ### assign the task vectors to the respective blocks
            task_vectors[i][find_block][layer] = task_vectors_all_layers[i][layer]
            sparsified_task_vectors[i][find_block][layer] = sparsified_task_vectors_all_layers[i][layer]

    T = args.MERGE_ITER
    n = args.NUM_CLIENTS
    T_max = (n - 1) * len(blocks) ### maximum number of iterations
    T = min(T, T_max)
    T_TUNE_START = int(args.TUNE_START * T)
    EVAL_AFTER = int(args.EVAL_AFTER * T)
    print("Total iterations: ", T)
    eval_freq = max(int(0.05 * T), 1) ### evaluate the performance every 5% of the iterations
    dsu_blocks = {}
    for block in blocks:
        ### create disjoint set union data structure to store the clusters in each block
        dsu_blocks[block] = DisjointSetUnionWithTaskVectors(
            n, 
            [task_vectors[i][block] for i in range(n)], 
            [sparsified_task_vectors[i][block] for i in range(n)],
            server,
            args.MERGE_METHOD, 
            args.K, 
            args.N_VAL_BATCHES, 
            args.LAMBDA, 
            args.CONSENSUS_LAMBDA
        )

    block_max = None
    has_pretrained_block = {block: False for block in blocks}
    blocks_remaining = blocks.copy()
    for t in range(T):
        to_tune = True if t >= T_TUNE_START and args.USE_VAL else False
        
        # choose a block randomly
        block_max = rng.choice(blocks_remaining)
        unique_groups = list(dsu_blocks[block_max].find_unique_groups())  ### find clusters in each block
        print(unique_groups)
        n1 = len(unique_groups)
        if not n1 > 1: raise Exception("No merge possible")
        i_max, j_max = rng.choice(unique_groups, size=2, replace=False)        

        size_i_max = dsu_blocks[block_max].get_group_size(i_max)
        size_j_max = dsu_blocks[block_max].get_group_size(j_max)
        to_remove, to_add, to_add_masks = get_size_change(size_i_max, size_j_max, has_pretrained_block[block_max])
        if size_i_max + size_j_max > 2: has_pretrained_block[block_max] = True
        
        print("Iteration ", t, "Fused Block Layer ", block_max, "Fused block similarity" , 0.0, "Fused Group indices" , (i_max, j_max))
        dsu_blocks[block_max].union(i_max, j_max, to_tune)  ### merge the clusters with maximum similarity
        
        # check if the block has only one cluster remaining
        n_remaining = len(list(dsu_blocks[block_max].find_unique_groups()))
        if n_remaining == 1:
            blocks_remaining.remove(block_max)
        
        # apply the merged task vector
        merged_group = dsu_blocks[block_max].find(i_max) ### identity of the merged cluster
        merged_group_task_vector = dsu_blocks[block_max].get_group_task_vector(i_max) ### get the task vector of the merged cluster

        if args.MERGE_METHOD in ['consensus_ties', 'consensus_ta', 'emr']:
            for layer in merged_group_task_vector:
                store_params -= merged_group_task_vector[layer].numel() * to_remove  
                store_params += merged_group_task_vector[layer].numel() * to_add
                store_params += merged_group_task_vector[layer].numel() * to_add_masks * 1/32 ### Assuming 32-bit model
        else:
            for layer in merged_group_task_vector:
                store_params -= merged_group_task_vector[layer].numel()  ### update the number of parameters stored in the server

        if not ((args.MERGE_METHOD in ['consensus_ties', 'consensus_ta', 'emr']) and (size_i_max + size_j_max <= 2)):
            for i in range(n):
                if dsu_blocks[block_max].find(i) == merged_group:
                        print ("Updating client ", i)
                        for layer in merged_group_task_vector:
                            server.client_state_dicts[i][layer] = base_state_dict[layer] + merged_group_task_vector[layer]  ### update the client models in the merged cluster with the merged task vector
        else:
            print(f'==> Skipping update for clients in block {block_max} as the size is less than 2')

        if (t % eval_freq == 1 and t > EVAL_AFTER) or t == T - 1:
            if not args.NO_EVAL: ### evaluation can be explicitly turned off
                avg_loss, avg_acc, client_acc_dict = server.evaluate_decentralized(args_print=True)
                client_norm_acc_dict = {}
                for k in client_acc_dict:
                    if client_ft_acc_dict is None:
                        client_norm_acc_dict[k] = client_acc_dict[k]
                    else:
                        client_norm_acc_dict[k] = client_acc_dict[k] / client_ft_acc_dict[k]
                
                metric_dict = {
                    "avg_loss": avg_loss,
                    "avg_acc": avg_acc,
                    "avg_norm_acc": sum(client_norm_acc_dict.values()) / len(client_norm_acc_dict),
                    "compression": total_params/store_params,
                    "merge_iter": t,
                    "cos_sim": 0.0,
                    "n_models": n * store_params / total_params
                }
                
                for dataset in datasets:
                    metric_dict[f"{dataset}/acc"] = client_acc_dict[dataset]
                    metric_dict[f"{dataset}/norm_acc"] = client_norm_acc_dict[dataset]
                
                if args.WANDB:
                    wandb.log(metric_dict)

            if args.SNAPSHOT:
                server.save_snapshot()
        
    print("="*5, "Run Complete", "="*5)

if __name__ == '__main__':
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # used to disable warnings
    
    args = get_args() 
    
    torch.manual_seed(args.SEED)
    np.random.seed(args.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    run(args)

    if args.WANDB:
        wandb.finish()