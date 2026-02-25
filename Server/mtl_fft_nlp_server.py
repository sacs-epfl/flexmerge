from copy import deepcopy
import os
import torch

from Server.base_server import Server
from Utils.variables_and_paths import ALL_NLPFFT_DATASETS
from client import MTL_FFTNLP_Client
from NLP.model.load_model import loadCheckpoint_intoModel
from NLP.inference import inference

class MTL_FFTNLP_server(Server):
    
    def __init__(self, args, model, tokenizer, config_toInit, model_config, kwargs):
        super().__init__(args, model)
        self.args = args
        self.client_state_dicts = []
        self.client_state_masks = [] ### used by Consensus, EMR-Merging, etc.
        self.prev_state = None
        self.is_splitted = True if "splitted" in args.BLOCK_GRANULARITY else False
        self.client_min_max_dict = None ### used for PCB merging, otherwise None
        self.snapshot_count = 0 ### used for saving the state_dicts and state_masks
        self.tokenizer = tokenizer
        self.config_toInit = config_toInit
        self.model_config = model_config
        self.kwargs = kwargs
        self.experiment_dir = os.path.join(args.SAVEDIR, args.SAVENAME)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
       
        for id in range(args.NUM_CLIENTS):
            dataset = ALL_NLPFFT_DATASETS[id]
            client = MTL_FFTNLP_Client(id, dataset) 
            self.clients.append(client)
            self.client_state_dicts.append(None)
            self.client_state_masks.append({})

        self.__name__ = 'MTL_FFTNLP_server'

    def test(self, testsite, clients, ptm_check, is_val=False, n_batches=-1):
        avg_acc, avg_loss, client_acc_dict = 0.0, 0.0, {}
        if testsite == 'clientside_with_servermodel':
            client_acc_dict = {}
            avg_acc, avg_loss = 0.0, 0.0
            for client in clients:
                cid = client.id
            
                client_mask_dict = self.client_state_masks[cid]
                client_state_dict = deepcopy(self.client_state_dicts[cid])
                for layer in client_mask_dict:
                    client_state_dict[layer] = (client_state_dict[layer] - ptm_check[layer]) * client_mask_dict[layer] + ptm_check[layer]

                self.model.to(self.device)
                
                kwargs_copy = deepcopy(self.kwargs)
                if not is_val:
                    kwargs_copy['split'] = 'test'
                else:
                    kwargs_copy['split'] = 'validation'
                
                _, scores_perDataset = inference(self.model, self.tokenizer, self.config_toInit, self.model_config, {}, False, \
                                                 self.experiment_dir, [client.dataset], kwargs_copy, self.device)
                keys = list(scores_perDataset.keys())
                key = [k for k in keys if k != 'average'][0]
                scores_perDataset[key] = scores_perDataset[key]["accuracy"]
                client_acc_dict[key.lower()] = scores_perDataset[key]
                avg_acc += scores_perDataset[key]
                self.model.to('cpu')
        
            avg_acc /= len(clients)
            print("Across clients: average accuracy {:.3f}, average loss {:.3f}".format(avg_acc, avg_loss))
        
        elif testsite == "clientside":
            client_acc_dict = {}
            avg_acc, avg_loss = 0.0, 0.0
            for client in clients:
                cid = client.id
                client_model = deepcopy(self.model)
            
                client_mask_dict = self.client_state_masks[cid]
                client_state_dict = deepcopy(self.client_state_dicts[cid])
                for layer in client_mask_dict:
                    client_state_dict[layer] = (client_state_dict[layer] - ptm_check[layer]) * client_mask_dict[layer] + ptm_check[layer]

                loadCheckpoint_intoModel(client_state_dict, client_model)
                client_model.to(self.device)
                
                kwargs_copy = deepcopy(self.kwargs)
                if not is_val:
                    kwargs_copy['split'] = 'test'
                else:
                    kwargs_copy['split'] = 'validation'
                    kwargs_copy['max_datapoints_per_dataset_without_templates'] = 500
                
                _, scores_perDataset = inference(client_model, self.tokenizer, self.config_toInit, self.model_config, {}, False, \
                                                 self.experiment_dir, [client.dataset], kwargs_copy, self.device)
                keys = list(scores_perDataset.keys())
                key = [k for k in keys if k != 'average'][0]
                scores_perDataset[key] = scores_perDataset[key]["accuracy"]
                client_acc_dict[key.lower()] = scores_perDataset[key]
                avg_acc += scores_perDataset[key]
                client_model.to('cpu')
                del client_model
        
            avg_acc /= len(clients)
            print("Across clients: average accuracy {:.3f}, average loss {:.3f}".format(avg_acc, avg_loss))
        
        return avg_loss, avg_acc, client_acc_dict
    
    def evaluate_decentralized(self, ptm_check, args_print=True):
        
        return self.test(testsite="clientside", clients=self.selected_clients, ptm_check=ptm_check)