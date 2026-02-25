from copy import deepcopy
import os
import json
import torch
import pandas as pd

from Server.base_server import Server
from client import MTL_BEIT3_Client

from beit3.default_args import is_supported_beit3_task, create_personalized_args

def load_checkpoint_info(checkpoint_info_path):
    def create_model_config(model_name, task):
        if task in ("flickr30k", "coco_retrieval"):
            model_config = "%s_retrieval" % model_name
        elif task in ("coco_captioning", "nocaps"):
            model_config = "%s_captioning" % model_name
        elif task in ("imagenet"):
            model_config = "%s_imageclassification" % model_name
        else:
            model_config = "%s_%s" % (model_name, task)
        return model_config
    checkpoint_info = pd.read_csv(checkpoint_info_path)
    checkpoint_info["model_config"] = checkpoint_info.apply(
        lambda row: create_model_config(row["model_name"], row["task_name"]), axis=1
    )
    checkpoint_info = checkpoint_info.set_index("task_name")
    return checkpoint_info

def get_data_path(data_root_path, task_name):
    postfix = None
    if task_name in ["coco_captioning", "coco_retrieval", "vqav2"]:
        postfix = "coco"
    elif task_name == "imagenet":
        postfix = "imagenet"
    elif task_name == "nlvr2":
        postfix = "nlvr2"
    if postfix is None:
        raise ValueError(f"Unsupported task name: {task_name}")
    data_path = os.path.join(data_root_path, postfix)
    return data_path

class MTL_BEIT3_server(Server):
    
    def __init__(self, args, eval_tasks, checkpoint_dir, base_data_path, eval_config_path):
        super().__init__(args, None)
        self.args = args
        self.client_state_dicts = []
        self.client_state_masks = [] ### used by Consensus, EMR-Merging, etc.
        self.prev_state = None
        self.is_splitted = True if "splitted" in args.BLOCK_GRANULARITY else False
        self.client_min_max_dict = None ### used for PCB merging, otherwise None
        self.snapshot_count = 0 ### used for saving the state_dicts and state_masks
        self.experiment_dir = os.path.join(args.SAVEDIR, args.SAVENAME)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_dir = checkpoint_dir

        with open(eval_config_path, 'r') as f:
            eval_config = json.load(f)
        sentencepiece_model_path = os.path.join(checkpoint_dir, "beit3.spm")
        checkpoints_info = load_checkpoint_info(os.path.join(checkpoint_dir, "checkpoints_info.csv"))
        base_output_dir = os.path.join(os.path.abspath(args.SAVEDIR), args.SAVENAME)

        for i, task in enumerate(eval_tasks):
            if not is_supported_beit3_task(task):
                raise ValueError(f"Task {task} is not supported by BEiT-3.")
            checkpoint_path = os.path.join(checkpoint_dir, checkpoints_info.loc[task]["filename"])
            data_path = get_data_path(base_data_path, task)
            output_dir = os.path.join(base_output_dir, task)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            task_args = create_personalized_args(
                task_name=task,
                sentencepiece_model_path=sentencepiece_model_path,
                finetuned_model_path=checkpoint_path,
                data_path=data_path,
                output_dir=output_dir,
                batch_size=eval_config["batch_size"][task],
                num_workers=eval_config["num_workers"][task],
                dist_eval=eval_config["dist_eval"],
            )
            model_config = checkpoints_info.loc[task]["model_config"]
            client = MTL_BEIT3_Client(i, args=task_args, model_config=model_config)
            
            self.clients.append(client)
            self.client_state_dicts.append(None)
            self.client_state_masks.append({})
        args.NUM_CLIENTS = len(self.clients)

        self.__name__ = 'MTL_BEIT3_server'

    def test(self, testsite, clients, ptm_check, is_val=False, n_batches=-1):
        assert testsite == "clientside", "Only clientside evaluation is supported for beit3"
        assert is_val == False, "only evaluation on test set is supported for beit3 clientside"
        avg_acc, avg_loss, client_acc_dict = 0.0, 0.0, {}
        client_res_dict = {}
        for client in clients:
            cid = client.id
            
            client_mask_dict = self.client_state_masks[cid]
            client_state_dict = deepcopy(self.client_state_dicts[cid])
            for layer in client_mask_dict:
                client_state_dict[layer] = (client_state_dict[layer] - ptm_check[layer]) * client_mask_dict[layer] + ptm_check[layer]

            client.load_state_dict(client_state_dict)

            client_eval_result = client.eval(self.device)
            for key in client_eval_result:
                client_res_dict[f'{client.args.task}/{key}'] = client_eval_result[key]
        return client_res_dict
            
        #     client_acc_dict = {}
        #     avg_acc, avg_loss = 0.0, 0.0
        #     for client in clients:
        #         cid = client.id
        #         client_model = deepcopy(self.model)
            
        #         client_mask_dict = self.client_state_masks[cid]
        #         client_state_dict = deepcopy(self.client_state_dicts[cid])
        #         for layer in client_mask_dict:
        #             client_state_dict[layer] = (client_state_dict[layer] - ptm_check[layer]) * client_mask_dict[layer] + ptm_check[layer]

        #         loadCheckpoint_intoModel(client_state_dict, client_model)
        #         client_model.to(self.device)
                
        #         kwargs_copy = deepcopy(self.kwargs)
        #         if not is_val:
        #             kwargs_copy['split'] = 'test'
        #         else:
        #             kwargs_copy['split'] = 'validation'
        #             kwargs_copy['max_datapoints_per_dataset_without_templates'] = 500
                
        #         _, scores_perDataset = inference(client_model, self.tokenizer, self.config_toInit, self.model_config, {}, False, \
        #                                          self.experiment_dir, [client.dataset], kwargs_copy, self.device)
        #         keys = list(scores_perDataset.keys())
        #         key = [k for k in keys if k != 'average'][0]
        #         scores_perDataset[key] = scores_perDataset[key]["accuracy"]
        #         client_acc_dict[key.lower()] = scores_perDataset[key]
        #         avg_acc += scores_perDataset[key]
        #         client_model.to('cpu')
        #         del client_model
        
        #     avg_acc /= len(clients)
        #     print("Across clients: average accuracy {:.3f}, average loss {:.3f}".format(avg_acc, avg_loss))
        
        # return avg_loss, avg_acc, client_acc_dict
    
    def evaluate_decentralized(self, ptm_check, args_print=True):
        
        return self.test(testsite="clientside", clients=self.selected_clients, ptm_check=ptm_check)