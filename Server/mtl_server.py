import torch
from typing import List, Dict
from copy import deepcopy
import os

from Server.base_server import Server
from client import MTL_Client
from Utils.utils import matrix_sim
from Utils.variables_and_paths import ALL_DATASETS
from Utils.merging_utils import merge_state_dict, split_state_dict

class MTL_server(Server):
    
    def __init__(self, args, model):
        super().__init__(args, model)
        
        self.args = args
        self.client_state_dicts = []
        self.client_state_masks = [] ### used by Consensus, EMR-Merging, etc.
        self.prev_state = None
        self.is_splitted = True if "splitted" in args.BLOCK_GRANULARITY else False
        self.client_min_max_dict = None ### used for PCB merging, otherwise None
        self.snapshot_count = 0 ### used for saving the state_dicts and state_masks
       
        for id in range(args.NUM_CLIENTS):
            dataset = ALL_DATASETS[id]
            client = MTL_Client(args, id, model.train_preprocess, model.val_preprocess, dataset) 
            self.clients.append(client)
            self.total_train_samples += client.train_samples
            self.client_state_dicts.append(None)
            self.client_state_masks.append({})

        self.__name__ = 'MTL_server'
          
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
        raise NotImplementedError("Train method not implemented in MTL_server")

    def test(self, testsite, clients, is_val=False, n_batches=-1):
        avg_acc, avg_loss = 0.0, 0.0

        if testsite == 'clientside_with_servermodel':
            client_acc_dict = {}
            num_samples, total_correct, losses = [], [], []
            avg_acc, avg_loss = 0.0, 0.0
            for client in clients:
                cid = client.id
                c_crt, c_ns, c_loss = client.test(self.model, is_val=is_val, n_batches=n_batches)
                client_acc = c_crt/c_ns
                avg_acc += client_acc
                avg_loss += c_loss
                print("Accuracy on client {} ({}) is {:.3f}".format(cid, client.dataset, client_acc))
                num_samples.append(c_ns)
                total_correct.append(c_crt*1.0)
                losses.append(c_loss)
                client_acc_dict[client.dataset.lower()] = client_acc
            
            avg_acc /= len(self.selected_clients)
            avg_loss /= len(self.selected_clients)
            print("Across clients: average accuracy {:.3f}, average loss {:.3f}".format(avg_acc, avg_loss))
        
        elif testsite == "clientside":
            client_acc_dict = {}
            avg_acc, avg_loss = 0.0, 0.0
            base_state_dict = self.model.state_dict()
            if self.is_splitted:
                split_state_dict(base_state_dict, self.args.N_MLP_GROUPS) ### in-place split
            
            for client in clients:
                cid = client.id
                client_model = deepcopy(self.model)
                
                client_mask_dict = self.client_state_masks[cid]
                client_state_dict = deepcopy(self.client_state_dicts[cid])
                for layer in client_mask_dict:
                    client_state_dict[layer] = (client_state_dict[layer] - base_state_dict[layer]) * client_mask_dict[layer] + base_state_dict[layer]

                if self.is_splitted:  ### merge the splitted blocks to get match the structure of the original state_dict
                    merge_state_dict(client_state_dict, self.args.N_MLP_GROUPS) ### in-place merge on deepcopied state_dict

                client_model.load_state_dict(client_state_dict, strict=False)
                c_crt, c_ns, c_loss = client.test(client_model, is_val=is_val, n_batches=n_batches)
                client_acc = c_crt/c_ns
                avg_acc += client_acc
                avg_loss += c_loss
                print("Accuracy on client {} ({}) is {:.3f}".format(cid, client.dataset, client_acc))
                client_acc_dict[client.dataset.lower()] = client_acc
                del client_model
            
            avg_acc /= len(clients)
            avg_loss /= len(clients)
            print("Across clients: average accuracy {:.3f}, average loss {:.3f}".format(avg_acc, avg_loss))
        
        return avg_loss, avg_acc, client_acc_dict
    
    def evaluate_decentralized(self, args_print=True):
        
        return self.test(testsite="clientside", clients=self.selected_clients)
    
    def save_snapshot(self):
        # save all client_state_dicts and client_state_masks
        for i in range(self.args.NUM_CLIENTS):
            save_folder = os.path.join(self.args.SAVEDIR, self.args.SAVENAME, f"t{self.snapshot_count}")
            os.makedirs(save_folder, exist_ok=True)
            torch.save(self.client_state_dicts[i], os.path.join(save_folder, f"client_{i}_state_dict.pt"))
            torch.save(self.client_state_masks[i], os.path.join(save_folder, f"client_{i}_state_masks.pt"))

        self.snapshot_count += 1

    def load_snapshot(self, t):
        print(f"==> Loading snapshot {t}")
        for i in range(self.args.NUM_CLIENTS):
            snapshot_folder = os.path.join(self.args.SAVEDIR, self.args.SAVENAME, f"t{t}")
            self.client_state_dicts[i] = torch.load(os.path.join(snapshot_folder, f"client_{i}_state_dict.pt"), map_location="cpu")
            self.client_state_masks[i] = torch.load(os.path.join(snapshot_folder, f"client_{i}_state_masks.pt"), map_location="cpu")