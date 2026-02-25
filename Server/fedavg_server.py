import numpy as np
import torch
import copy
from typing import List, Dict

from Server.base_server import Server
from client import Client
from Utils.utils import matrix_sim
from Utils.merging_utils import ties_merging, state_dict_to_vector, vector_to_state_dict

class FedAvg_server(Server):
    
    def __init__(self, args, model):
        super().__init__(args, model)
        
        self.args = args
        self.client_state_dicts = []
        self.prev_state = None
       

        for id in range(args.NUM_CLIENTS):
            client = Client(args, id, args.trainloaders[id], args.valloaders[id], args.ctestloaders[id]) 
            self.clients.append(client)
            self.total_train_samples += client.train_samples
            self.client_state_dicts.append(None)

        self.__name__ = 'FedAvg_server'
          
    def add_parameters(self, client, ratio):
    
        for server_params, client_params in zip(self.model.parameters(), client.model.parameters()):
            server_params.data = server_params.data + client_params.data.clone() * ratio

    def aggregate_parameters(self, rounds = 0):

        assert (self.selected_clients is not None and len(self.selected_clients) > 0)
        finetuned_task_vectors = \
            [self.state_dict_sub(self.client_state_dicts[client.id], self.model.state_dict(), strict=False) \
             for client in self.selected_clients]
        p = [self.args.WEIGHTS[client.id] for client in self.selected_clients]

        if(rounds%1==0):
            M = [self.state_dict_to_vector(finetuned_task_vectors[i]) for i in range(len(finetuned_task_vectors))]
            avg_norm, avg_corr, matrix_corr = matrix_sim(M)
            print ("avg_weight_norm={}, avg_weight_corr={}".format(avg_norm, avg_corr))

        if self.args.MERGE_METHOD == 'avg':
            merged_task_vector_no_scale = self.state_dict_avg(finetuned_task_vectors, p = p)
        elif self.args.MERGE_METHOD == 'ties':
            merged_task_vector_no_scale = self.state_dict_avg_ties(finetuned_task_vectors, p = p)
        assert set(merged_task_vector_no_scale.keys()).issubset(self.model.state_dict().keys())


        merged_task_vector = self.state_dict_mul(merged_task_vector_no_scale, self.args.SCALE)
        model_state_dict = self.state_dict_add(merged_task_vector, self.model.state_dict(), strict=False)
        self.model.load_state_dict(model_state_dict, strict=False)
          
    def train(self, rounds=0, args_print=True, lr_decay = True):

        print("{}-th round algo-{}:".format(rounds+1, self.algo))

        train_losses, train_acc, num_samples = [], [], []
        
        self.selected_clients, self.selected_clients_idx = self.select_clients(self.num_selected)
        
        for client in self.selected_clients: 

            loss_i, acc_i, client_state_dict = client.train(self.model, self.text_embeds, iter)
            self.client_state_dicts[client.id] = client_state_dict

            train_losses.append(loss_i)
            train_acc.append(acc_i)
            num_samples.append(client.train_samples)


        if(self.args.SAVE_CLIENT_MODELS):
            print ("Saving client models")
            self.save_client_models(self.args, rounds)
        if(lr_decay):
            self.args.LEARNING_RATE = self.args.LEARNING_RATE*self.args.LR_DECAY
        self.aggregate_parameters(rounds = rounds)
        

        train_loss_aggr = round(np.sum([x * y for (x, y) in zip(num_samples, train_losses)]).item() / np.sum(num_samples), self.decimal)
        train_acc_aggr = round(np.sum([x * y for (x, y) in zip(num_samples, train_acc)]).item() / np.sum(num_samples), self.decimal)


        if args_print:
            print("loss_train={}, acc_train={}"\
                .format(train_loss_aggr, train_acc_aggr))  
    
    # average method which zero-out the weights which are not in the top-k before averaging
    def state_dict_avg_ties(self, state_dicts: List[Dict], p: List[float] = None, K=0.2, trim=True):
        """
        Returns the ties average of a list of state dicts.

        Args:
            state_dicts (List[Dict]): A list of state dicts.
            p (List[float], optional): Weights for averaging each state dict.
            K (float, optional): Fraction of values to retain based on magnitude.
            trim (bool, optional): Whether to trim using topK.

        Returns:
            Dict: The average of the state dicts.
        """
        task_vectors = [state_dict_to_vector(state_dict) for state_dict in state_dicts]
        task_vectors = torch.stack(task_vectors, dim=0)
        merged_task_vector = ties_merging(task_vectors, reset_thresh=K, trim=trim)
        merged_state_dict = vector_to_state_dict(merged_task_vector, state_dicts[0])
        return merged_state_dict