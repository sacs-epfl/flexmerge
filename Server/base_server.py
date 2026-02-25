import sys
import torch
import os
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from transformers import CLIPProcessor
from torch import Tensor, nn
from typing import Any, Dict, List, Tuple, Union
from copy import deepcopy
from tqdm import tqdm
from collections import OrderedDict

sys.path.append('../..')

from Utils.utils import add_data_to_metric
from Data.text_for_classes import get_classes
from Utils.merging_utils import ties_merging, state_dict_to_vector, vector_to_state_dict, \
    pcb_merging, emr_merge

class Server:
    
    def __init__(self, args, model):
        
        self.model = model
        self.client_models = []
        self.prev_model = None
        self.client_personal_lora_models=[]
        self.device = args.DEVICE
        self.seed = args.SEED

        self.dataset = args.DATATYPE
        self.decimal = args.SAVE_DECIMAL

        self.num_clients = args.NUM_CLIENTS
        self.clients = []
        self.min_clients = args.MIN_CLIENTS
        self.sampling_client_rate = args.SAMRATE_CLIENTS
        self.num_selected = max(round(self.num_clients*self.sampling_client_rate), self.min_clients)
        self.selected_clients = []

        self.local_epochs = args.LOCAL_EPOCH
        self.num_rounds = args.ROUNDS
        self.batch_size = args.BATCH_SIZE
        self.learning_rate = args.LEARNING_RATE


        self.total_train_samples = 0
        self.algo = args.METHOD
        self.init_loss_fn()

        if self.algo == 'FedAvg':
            self.testloaders = args.stestloaders
            self.valloaders = args.valloaders
                        
            clip_processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
            classes = get_classes(self.dataset)
            text = [f"a photo of a {c}" for c in classes] 
            text_input = clip_processor(text, return_tensors="pt", padding=True)
            text_embeds = self.model.get_text_features(**text_input)
            text_embeds = text_embeds.to(self.device)
            self.text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

    def state_dict_to_vector(self, state_dict: Dict[str, Tensor]):
        """
        Converts a PyTorch state dictionary to a 1D tensor.

        Args:
            state_dict (Dict[str, Tensor]): A dictionary containing the state of a PyTorch model.

        Returns:
            Tensor: A 1D tensor containing the values of the state dictionary, sorted by key.
        """
        sorted_shared_state_dict = OrderedDict(sorted(state_dict.items()))
        flat_tensor = torch.nn.utils.parameters_to_vector(
            [value.reshape(-1) for _, value in sorted_shared_state_dict.items()]
        )
        if self.args.GPU_MERGE:
            return flat_tensor.cuda()
        else:
            return flat_tensor
    
    def state_dict_add(self, a: Dict, b: Dict, strict: bool = True):
        """
        Returns the sum of two state dicts.

        Args:
            a (Dict): The first state dict.
            b (Dict): The second state dict.
            strict (bool): Whether to check if the keys of the two state dicts are the same.

        Returns:
            Dict: The sum of the two state dicts.
        """
        if strict:
            assert set(a.keys()) == set(b.keys())

        diff = {}
        for k in a:
            if k in b:
                diff[k] = a[k] + b[k]
        return diff

    def state_dict_sub(self, a: Dict, b: Dict, strict: bool = True):
        """
        Returns the difference of two state dicts.

        Args:
            a (Dict): The first state dict.
            b (Dict): The second state dict.
            strict (bool): Whether to check if the keys of the two state dicts are the same.

        Returns:
            Dict: The difference of the two state dicts.
        """
        if strict:
            assert set(a.keys()) == set(b.keys())

        diff = {}
        for k in a:
            if k in b:
                diff[k] = a[k] - b[k]
        return diff

    def state_dict_avg(self, state_dicts: List[Dict], p: List[float] = None):
        """
        Returns the average of a list of state dicts.

        Args:
            state_dicts (List[Dict]): A list of state dicts.

        Returns:
            Dict: The average of the state dicts.
        """
        diff = {}
        if(p is None):
            p = [1/len(state_dicts) for i in range(len(state_dicts))]
        for k in state_dicts[0]:
            diff[k] = sum([p[i]*state_dict[k] for (i,state_dict) in enumerate(state_dicts)])/sum(p)
        return diff

    def state_dict_mul(self, state_dict: Dict, scalar: float):
        """
        Returns the product of a state dict and a scalar.

        Args:
            state_dict (Dict): The state dict to be multiplied.
            scalar (float): The scalar to multiply the state dict with.

        Returns:
            Dict: The product of the state dict and the scalar.
        """
        diff = {}
        for k in state_dict:
            diff[k] = scalar * state_dict[k]
        return diff
    
    def select_clients(self, num_clients):

        if(num_clients == len(self.clients)):
            print("Running full participation")
            return self.clients, [i for i in range(len(self.clients))]
        else:
            assert (num_clients>0), f"No clients are selected"
            selected_idxs = np.random.choice(range(len(self.clients)), num_clients, replace=False)
            self.selected_clients =  [self.clients[i] for i in selected_idxs]

            print("Running partial participation, selected clients: {}".format([client.id for client in self.selected_clients]))

            return self.selected_clients, selected_idxs

    def test(self, testsite, clients, is_val=False, n_batches=-1):
 
        if testsite == "serverside":
            total_corrects, loss, num_samples = self._test(model = self.model, testloader = self.testloaders[-1])
            acc = total_corrects/num_samples
            return loss, acc

        if testsite == "val_serverside":
            total_corrects, loss, num_samples = self._test(model = self.model, testloader = self.valloaders[-1])
            acc = total_corrects/num_samples
            return loss, acc

        elif testsite == "clientside":
            num_samples, total_correct, losses = [], [], []
            avg_acc, avg_loss = 0.0, 0.0
            for client in self.selected_clients:
                cid = client.id
                client_model = deepcopy(self.model)
                client_model.load_state_dict(self.client_state_dicts[cid], strict=False)
                dataloader = client.valloader if is_val else client.testloader
                c_crt, c_loss, c_ns = self._test(model = client_model, testloader=dataloader, n_batches=n_batches)
                client_acc = c_crt/c_ns
                avg_acc += client_acc
                avg_loss += c_loss
                print("Accuracy on client {} is {}".format(cid, client_acc))
                del client_model
                num_samples.append(c_ns)
                total_correct.append(c_crt*1.0)
                losses.append(c_loss)
            
            avg_acc /= len(self.selected_clients)
            avg_loss /= len(self.selected_clients)
            print("Across clients: average accuracy {}, average loss {}".format(avg_acc, avg_loss))
            ids = [client.id for client in clients]

            return avg_loss, avg_acc

        elif testsite == 'clientside_with_servermodel':
            num_samples, total_correct, losses = [], [], []
            avg_acc, avg_loss = 0.0, 0.0
            for client in self.selected_clients:
                cid = client.id
                dataloader = client.valloader if is_val else client.testloader
                c_crt, c_loss, c_ns = self._test(model = self.model, testloader=dataloader, n_batches=n_batches)
                client_acc = c_crt/c_ns
                avg_acc += client_acc
                avg_loss += c_loss
                print("Accuracy on client {} is {}".format(cid, client_acc))
                num_samples.append(c_ns)
                total_correct.append(c_crt*1.0)
                losses.append(c_loss)
            
            avg_acc /= len(self.selected_clients)
            avg_loss /= len(self.selected_clients)
            print("Across clients: average accuracy {}, average loss {}".format(avg_acc, avg_loss))
            ids = [client.id for client in clients]

            return avg_loss, avg_acc

    def _test(self, model, testloader, n_batches=-1):

        model.eval()
        total_corrects, loss, num_eg = 0, 0.0, 0        
        model.to(self.device)

        with torch.no_grad():
            idx = 0
            for images, labels in tqdm(testloader):
                
                images, labels = images.to(self.device), labels.to(self.device)
                image_embeds = model.get_image_features(pixel_values=images)

                # normalized features
                image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
            

                # cosine similarity as logits
                logit_scale = model.logit_scale.exp().item()
                logits_per_text = torch.matmul(self.text_embeds, image_embeds.t()) * logit_scale
                logits_per_image = logits_per_text.t()

                loss += F.cross_entropy(logits_per_image, labels).detach()
                num_eg += labels.size(0)

                pred = logits_per_image.argmax(dim=-1)
                total_corrects += (pred == labels).sum().item()

                idx += 1
                if n_batches > 0 and idx >= n_batches - 1:
                    break
                
        # Moving models from GPU to CPU
        model.to("cpu")
        # Clearing GPU cache
        torch.cuda.empty_cache() 

        return total_corrects, loss.to('cpu').numpy() / idx, num_eg 

    def evaluate_decentralized(self, args_print=True): # -> evaluate_decentralized
        
        print('evaluating on client-side testloaders...')

        ctest_ids, ctest_samples, ctest_corrects, ctest_losses = self.test(testsite="clientside", clients=self.selected_clients)

        ctest_acc_list, ctest_loss_list = [], []
        print (len(ctest_ids), len(ctest_samples), len(ctest_corrects), len(ctest_losses))

        for ns, crts, closs in zip(ctest_samples, ctest_corrects, ctest_losses):
            ctest_acc_list.append(round(crts/ns, self.decimal))
            ctest_loss_list.append(round(closs, self.decimal))
 
        if args_print:
            avg_loss = sum(ctest_loss_list)/len(ctest_loss_list)
            avg_acc = sum(ctest_acc_list)/len(ctest_acc_list)
            print("Average loss_clienttest={}, Average acc_clienttest={}".format(avg_loss, avg_acc))
            return avg_loss, avg_acc
    
    def evaluate_centralized(self, save=True, args_print=True, val = False):  # -> evaluate_centralized

        if(val == False):
            print('evaluating on server-side testloader...')

            stest_ids, stest_samples, stest_corrects, stest_losses = self.test(testsite="serverside", clients=self.selected_clients)
        else:
            print('evaluating on server-side valloader...')

            stest_ids, stest_samples, stest_corrects, stest_losses = self.test(testsite="val_serverside", clients=self.selected_clients)                                                                                                                     
        if save:
            keys_to_update = ['servertest_acc', 'servertest_loss']
            # data_values = [test_acc_aggr, test_loss_aggr]
            data_values = [round(stest_corrects[0]/stest_samples[0], self.decimal), round(stest_losses[0], self.decimal)]
            add_data_to_metric(self.metrics, keys_to_update, data_values )  
        if args_print and self.algo !="Local":

            print("Loss_servertest={}, acc_servertest={}".format(data_values[1], data_values[0]))

        return data_values[1], data_values[0]
    
    def save_client_models(self, args, rounds):

        for i in range(args.NUM_CLIENTS):

            base_dir = os.path.join(args.SAVEDIR, args.SAVENAME)

            if not os.path.exists(base_dir):
                os.makedirs(base_dir, exist_ok=True)

            torch.save(self.client_state_dicts[i], os.path.join(base_dir, f"client_{i}" + ".pt"))

    def save_model(self):
        model_path = os.path.join("Saved_models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.model, os.path.join(model_path, "server" + ".pt"))

    def load_model(self):
        model_path = os.path.join("Saved_models", self.dataset, "server" + ".pt")
        assert (os.path.exists(model_path))
        self.model = torch.load(model_path) 

    def model_exists(self):
        return os.path.exists(os.path.join("Savedmodels", self.dataset, "server" + ".pt"))

    def init_loss_fn(self):
        self.loss = nn.CrossEntropyLoss()
        self.ensemble_loss=nn.KLDivLoss(reduction="batchmean")
        self.ce_loss = nn.CrossEntropyLoss()

    def build_metrics(self, metric_ids):
        groups = metric_ids.split(";")
        metric_dict = {}

        for group in groups:
            group_key, group_metrics = group.strip().split("->")
            group_key = group_key.strip()
            metrics = group_metrics.strip().split(",")

            group_dict = {}
            for metric in metrics:
                metric = metric.strip()
                group_dict[metric] = []

            metric_dict[group_key] = group_dict

        return metric_dict

    # average method which zero-out the weights which are not in the top-k before averaging
    def state_dict_avg_ties(self, state_dicts: List[Dict], K=0.2, trim=True):
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

    def state_dict_avg_pcb(self, state_dicts: List[Dict], K=0.2, trim=True, involved_client_ids=None, client_min_max_dict=None):
        """
        Returns the ties average of a list of state dicts.

        Args:
            state_dicts (List[Dict]): A list of state dicts.
            p (List[float], optional): Weights for averaging each state dict.
            K (float, optional): Fraction of values to retain based on magnitude.
            trim (bool, optional): Whether to trim using PCB dropping method.

        Returns:
            Dict: The average of the state dicts.
        """
        task_vectors = [state_dict_to_vector(state_dict) for state_dict in state_dicts]
        task_vectors = torch.stack(task_vectors, dim=0)
        merged_task_vector = pcb_merging(task_vectors, K, trim, involved_client_ids, client_min_max_dict)
        merged_state_dict = vector_to_state_dict(merged_task_vector, state_dicts[0])
        return merged_state_dict

    def state_dict_avg_emr(self, state_dicts: List[Dict]):
        """
        Returns the emr unified task vector and masks.

        Args:
            state_dicts (List[Dict]): A list of state dicts.

        Returns:
            Dict: The emr average of the state dicts.
            List[Dict]: The list of masks for each task vector.
        """
        merged_state_dict, masks = emr_merge(state_dicts)
        return merged_state_dict, masks

    def state_dict_consensus(self, state_dicts: List[Dict], merged_state_dict: Dict, consensus_lambda=0.3):
        """
        Returns the list of masks for each task vector.
        
        Args:
            state_dicts (List[Dict]): A list of state dicts.
            merged_state_dict (Dict): The merged state dict.
        Returns:
            List[Dict]: The list of masks for each task vector.
        """
        task_vectors = [state_dict_to_vector(state_dict) for state_dict in state_dicts]
        merged_task_vector = state_dict_to_vector(merged_state_dict)
        masks = []
        for task_vector in task_vectors:
            mask = task_vector.abs() > consensus_lambda * (merged_task_vector - task_vector).abs()
            mask = mask.float()
            print("Density rate: {:.3f}".format(mask.mean()))
            mask = vector_to_state_dict(mask, state_dicts[0])
            masks.append(mask)
        return masks




