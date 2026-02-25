import torch
import numpy as np
import os
import wandb

from Utils.server_utils import create_server
from Utils.args_utils import get_args
from Utils.merging_utils import find_sim, get_blocks, sparsify, sparsify_pcb, init_sim, \
    find_constituent_sim, get_size_change
from Utils.utils import read_ft_accuracy
from Utils.merging_datastructure import DisjointSetUnionWithTaskVectors

from beit3.merge_utils import load_beit3, vector_to_state_dict

task_metrics = {
    "coco_captioning": ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "METEOR", "ROUGE_L", "CIDEr"],
    "imagenet": ["acc1", "acc5"],
    "nlvr2": ["acc"],
    "coco_retrieval": ["tr_r10", "tr_r5", "tr_r1", "ir_r10", "ir_r5", "ir_r1", "average_score"],
    "vqav2": []
}

def wandb_init_metrics(datasets):
    metrics = ["compression", "merge_iter", "cos_sim", "n_models", "avg_acc", "avg_norm_acc"]
    # metrics = ["merge_iter", "avg_loss", "avg_acc", "avg_norm_acc", \
    #             "compression", "n_models", "cos_sim"]
    for dataset in datasets:
        if dataset not in task_metrics:
            raise ValueError(f"Dataset {dataset} not found in task metrics")
        metrics += [
            f"{dataset}/{metric}"
            for metric in task_metrics[dataset]
        ] + [
            f"{dataset}/{metric}_norm"
            for metric in task_metrics[dataset]
        ]
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
    print(server)
    

    ### zero-shot accuracy
    print("="*5, "Evaluating Zero Shot", "="*5)
    server.selected_clients = server.clients
    # server.test('clientside_with_servermodel', clients=server.clients)

    print("="*5, "Evaluating Model Merging", "="*5)

    n = args.NUM_CLIENTS
    
    (
        tv_flat_checks,
        flat_ptm,
        flat_ft,
        ft_checks,
        ptm_check,
        selected_keys,
    ) = load_beit3(server)
    
    for i in range(n):
        print("="*5, f"Loading client {i} model")
        server.client_state_dicts[i] = ft_checks[i]
    
    server.selected_clients = server.clients
    
    # ### initial ensemble performance
    # server.evaluate_decentralized(ptm_check, args_print=True)
    
    task_vectors_all_layers = [vector_to_state_dict(tv_flat_checks[i], ft_checks[0], selected_keys) \
                                for i in range(n)]  ### create task vectors for each client
    
    ### global sparsification for ties and pcb
    if args.ALG != 'cluster-merge': ### for cluster-merge we sparsify within the algorithm
        if args.MERGE_METHOD == 'ties' and args.K < 1.0:
            print("="*5, "Sparsifying Task Vectors using TIES", "="*5)
            task_vectors_all_layers = sparsify(task_vectors_all_layers, args.K)
        
        elif args.MERGE_METHOD == 'pcb' and args.K < 1.0:
            print("="*5, "Sparsifying Task Vectors using PCB", "="*5)
            server.client_min_max_dict = sparsify_pcb(task_vectors_all_layers, args.K) ### store in server for use by union method

    ### read ft accuracy to evaluate normalized accuracy
    file_path = os.path.join(os.path.abspath(args.BEIT3_CHECKPOINT_DIR.strip()), "ft_individual_metrics.csv")
    if not os.path.exists(file_path):
        ### currently raising error, can be relaxed to print a warning
        raise FileNotFoundError(f"Missing ft accuracy file: {file_path}")
    else:
        client_ft_acc_dict = read_ft_accuracy(file_path)
        for k,v in client_ft_acc_dict.items():
            print(k,v)
    
    if args.ALG == 'greedy':
        greedy_block_merge(args, server, task_vectors_all_layers, ptm_check, selected_keys=selected_keys, client_ft_acc_dict=client_ft_acc_dict)
    else:
        raise ValueError("Invalid Algorithm")
    

def greedy_block_merge(args, server, task_vectors_all_layers, ptm_check, selected_keys, client_ft_acc_dict=None):
    n = args.NUM_CLIENTS
    datasets = [server.clients[i].dataset.lower() for i in range(n)]
    if args.WANDB:
        wandb_init_metrics(datasets)

    base_state_dict = ptm_check
        
    total_params = 0
    for layer in task_vectors_all_layers[0]:
        total_params += task_vectors_all_layers[0][layer].numel()

    total_params = n*total_params  ### total number of parameters in all the clients
    store_params = total_params  ### total number of parameters that we are storing in the server

    blocks = selected_keys

    task_vectors = [{} for _ in range(n)]
    for block in blocks:
        for i in range(n):
            task_vectors[i][block] = {}  ### create task vectors for each block and each client

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
        dsu_blocks[block] = DisjointSetUnionWithTaskVectors(n, [task_vectors[i][block] for i in range(n)], server, \
                                    args.MERGE_METHOD, args.K, args.N_VAL_BATCHES, args.LAMBDA, args.CONSENSUS_LAMBDA)

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
            n1 = len(unique_groups)
            maybe_nan_flag = True
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
                        maybe_nan_flag = False ### set to False if this loop is executed even once
            if n1 > 1: ### merge possible only if there is more than one cluster in the block
                if maybe_nan_flag:
                    print("!"*10, "Warning: NaN similarity detected", "!"*10)
                    i_max, j_max, max_cos_sim = unique_groups[0], unique_groups[1], torch.tensor(-1.1) ### pick first two
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

        if (t % eval_freq == 1 and t > EVAL_AFTER) or t == T - 1 or t == 0:
            if not args.NO_EVAL: ### evaluation can be explicitly turned off
                eval_res = server.evaluate_decentralized(ptm_check, args_print=True)
                
                metric_dict = {
                    "compression": total_params/store_params,
                    "merge_iter": t,
                    "cos_sim": max_cos_sim.item(),
                    "n_models": n * store_params / total_params
                }
                
                for metric in eval_res:
                    metric_dict[metric] = eval_res[metric]
                
                if client_ft_acc_dict is not None:
                    new_entries = {}
                    for metric in metric_dict:
                        if metric in client_ft_acc_dict:
                            new_entries[f"{metric}_norm"] = metric_dict[metric] / client_ft_acc_dict[metric]
                    metric_dict.update(new_entries)
                
                metric_dict["avg_acc"] = 0
                if client_ft_acc_dict is not None:
                    metric_dict["avg_norm_acc"] = 0
                total_avg_acc_entries = 0
                for acc_metrics in [
                    "imagenet/acc1", "nlvr2/acc", "coco_retrieval/average_score"
                ]:
                    if acc_metrics in metric_dict:
                        metric_dict["avg_acc"] += metric_dict[acc_metrics]
                        total_avg_acc_entries += 1                       
                        if client_ft_acc_dict is not None:
                            metric_dict["avg_norm_acc"] += metric_dict[f"{acc_metrics}_norm"]
                
                if total_avg_acc_entries > 0:
                    metric_dict["avg_acc"] /= total_avg_acc_entries
                    if client_ft_acc_dict is not None:
                        metric_dict["avg_norm_acc"] /= total_avg_acc_entries
                else:
                    del metric_dict["avg_acc"]
                    if client_ft_acc_dict is not None:
                        del metric_dict["avg_norm_acc"]
                
                
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