import torch
import numpy as np
import os
import wandb
from copy import deepcopy
import time

from Utils.server_utils import create_server
from Utils.args_utils import get_args
from Utils.variables_and_paths import get_finetuned_path
    
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
            project=args.WANDB_PROJECT, 
            entity=args.WANDB_ENTITY,
            name=args.SAVENAME,
        )
    
    server = create_server(args)

    ### zero-shot accuracy
    # print("="*5, "Evaluating Zero Shot", "="*5)
    server.selected_clients = server.clients
    # if not args.NO_EVAL:
    #     server.test('clientside_with_servermodel', clients=server.clients)

    print("="*5, "Evaluating Model Merging", "="*5)

    n = args.NUM_CLIENTS

    for i in range(n):
        print("="*5, f"Loading client {i} model")
        model = deepcopy(server.model)
        if args.METHOD == 'FedAvg':
            model_path = os.path.join(args.MODELDIR, f'client_{i}.pt')
        else:
            model_path = get_finetuned_path(args.MODELDIR, server.clients[i].dataset, args.MODEL)
        model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=False)
        server.client_state_dicts[i] = model.state_dict()

    server.selected_clients = server.clients

    ### initial ensemble performance
    # print("="*5, "Fine-tuning Performance", "="*5)
    # # if not args.NO_EVAL:
    # #     # server.evaluate_decentralized(args_print=True)

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
    base_state_dict = deepcopy(server.model.state_dict())

    start_time = time.time()

    if args.MERGE_METHOD == 'avg':
        print("="*5, "Merging using Average", "="*5)
        merged_task_vector = server.state_dict_avg(task_vectors_all_layers)
        model_state_dict = server.state_dict_add(merged_task_vector, base_state_dict, strict=False)

        if not args.NO_EVAL:
            server.model.load_state_dict(model_state_dict, strict=False)
            server.test('clientside_with_servermodel', clients=server.clients)
    
    elif args.MERGE_METHOD == 'ties':
        print("="*5, "Merging using Ties", "="*5)
        if args.USE_VAL:
            Ks = [0.05, 0.1, 0.2]
            lambdas = np.linspace(0.8, 2.5, 18).tolist()
            if args.WANDB:
                for K in Ks:
                    wandb.define_metric(f"lambdas_for_K={K}")
                    wandb.define_metric(f"accuracy_for_K={K}", step_metric=f"lambdas_for_K={K}")
        
            best_acc = 0
            best_pair = None
            for K in Ks:
                print("="*5, f"K = {K}", "="*5)
                merged_task_vector = server.state_dict_avg_ties(task_vectors_all_layers, K = K)
                for l in lambdas:
                    print("="*5, f"Lambda = {l}", "="*5)
                    model_state_dict = server.state_dict_mul(merged_task_vector, l)
                    model_state_dict = server.state_dict_add(model_state_dict, base_state_dict, strict=False)
                    server.model.load_state_dict(model_state_dict, strict=False)
                    _, avg_acc, _ = server.test('clientside_with_servermodel', clients=server.clients, is_val=True, n_batches=50)
                    if args.WANDB: wandb.log({f"lambdas_for_K={K}": l, f"accuracy_for_K={K}": avg_acc})
                    if avg_acc > best_acc:
                        best_acc = avg_acc
                        best_pair = (K, l)
        else:
            best_pair = (args.K, args.LAMBDA)

        print("="*5, f"Best K: {best_pair[0]}, Best Lambda: {best_pair[1]}", "="*5)
        best_K, best_l = best_pair
        merged_task_vector = server.state_dict_avg_ties(task_vectors_all_layers, K = best_K)
        model_state_dict = server.state_dict_mul(merged_task_vector, best_l)
        model_state_dict = server.state_dict_add(model_state_dict, base_state_dict, strict=False)

        if not args.NO_EVAL:
            server.model.load_state_dict(model_state_dict, strict=False)
            _, avg_acc, _ = server.test('clientside_with_servermodel', clients=server.clients)
            if args.WANDB: wandb.log({"test_accuracy": avg_acc})
    
    elif args.MERGE_METHOD == 'ta':
        print("="*5, "Merging using TA", "="*5)
        merged_task_vector = server.state_dict_avg(task_vectors_all_layers)
        if args.USE_VAL:
            lambdas = [0.8, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]
        
            best_acc = 0
            best_l = None
            if args.WANDB:
                wandb.define_metric("lambda")
                wandb.define_metric("accuracy", step_metric="lambda")
            for l in lambdas:
                print("="*5, f"Lambda = {l}", "="*5)
                model_state_dict = server.state_dict_mul(merged_task_vector, l)
                model_state_dict = server.state_dict_add(model_state_dict, base_state_dict, strict=False)
                server.model.load_state_dict(model_state_dict, strict=False)
                _, avg_acc, _ = server.test('clientside_with_servermodel', clients=server.clients, is_val=True, n_batches=50)
                if args.WANDB: wandb.log({"lambda": l, "accuracy": avg_acc})
                if avg_acc > best_acc:
                    best_acc = avg_acc
                    best_l = l
        else:
            best_l = args.LAMBDA

        print("="*5, f"Best Lambda: {best_l}", "="*5)
        model_state_dict = server.state_dict_mul(merged_task_vector, best_l)
        model_state_dict = server.state_dict_add(model_state_dict, base_state_dict, strict=False)

        if not args.NO_EVAL:
            server.model.load_state_dict(model_state_dict, strict=False)
            _, avg_acc, _ = server.test('clientside_with_servermodel', clients=server.clients)
            if args.WANDB: wandb.log({"test_accuracy": avg_acc})
    
    elif args.MERGE_METHOD == 'pcb':
        print("="*5, "Merging using PCB", "="*5)
        if args.USE_VAL:
            Ks = [0.05, 0.1, 0.2]
            lambdas = np.linspace(0.8, 2.5, 18).tolist()
            if args.WANDB:
                for K in Ks: 
                    wandb.define_metric(f"lambdas_for_K={K}")
                    wandb.define_metric(f"accuracy_for_K={K}", step_metric=f"lambdas_for_K={K}")
        
            best_acc = 0
            best_pair = None
            for K in Ks:
                print("="*5, f"K = {K}", "="*5)
                merged_task_vector = server.state_dict_avg_pcb(task_vectors_all_layers, K = K)
                for l in lambdas:
                    print("="*5, f"Lambda = {l}", "="*5)
                    model_state_dict = server.state_dict_mul(merged_task_vector, l)
                    model_state_dict = server.state_dict_add(model_state_dict, base_state_dict, strict=False)
                    server.model.load_state_dict(model_state_dict, strict=False)
                    _, avg_acc, _ = server.test('clientside_with_servermodel', clients=server.clients, is_val=True, n_batches=50)
                    if args.WANDB: wandb.log({f"lambdas_for_K={K}": l, f"accuracy_for_K={K}": avg_acc})
                    if avg_acc > best_acc:
                        best_acc = avg_acc
                        best_pair = (K, l)
        else:
            best_pair = (args.K, args.LAMBDA)

        print("="*5, f"Best K: {best_pair[0]}, Best Lambda: {best_pair[1]}", "="*5)
        best_K, best_l = best_pair
        merged_task_vector = server.state_dict_avg_pcb(task_vectors_all_layers, K = best_K)
        model_state_dict = server.state_dict_mul(merged_task_vector, best_l)
        model_state_dict = server.state_dict_add(model_state_dict, base_state_dict, strict=False)

        if not args.NO_EVAL:
            server.model.load_state_dict(model_state_dict, strict=False)
            _, avg_acc, _ = server.test('clientside_with_servermodel', clients=server.clients)
            if args.WANDB: wandb.log({"test_accuracy": avg_acc})
    
    end_time = time.time()
    # time in seconds
    print(f"Time taken: {end_time - start_time} seconds")

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