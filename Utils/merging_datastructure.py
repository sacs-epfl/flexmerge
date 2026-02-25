from copy import deepcopy
import torch

from Server.mtl_fft_nlp_server import MTL_FFTNLP_server
from Server.mtl_ia3_server import MTL_IA3_server

def tune_lambda(server, merged_group_task_vector, involved_clients, n_batches, lambdas=None):
    if lambdas is None: lambdas = [0.8, 1.0, 1.5, 2.0]
    # check if server is instance of MTL_IA3_server
    if isinstance(server, MTL_IA3_server):
        print("Server is instance of MTL_IA3_server")
        ft_check = server.client_state_dicts[0]
        base_state_dict = {pn: torch.ones_like(pv) for pn, pv in ft_check.items()}
    else:
        base_state_dict = server.model.state_dict()

    ### make copies of involved clients' state dicts
    client_state_dicts = {}
    for i in involved_clients:
        client_state_dicts[i] = {}
        for layer in merged_group_task_vector:
            client_state_dicts[i][layer] = deepcopy(server.client_state_dicts[i][layer])

    client_objects = [server.clients[i] for i in involved_clients]

    best_acc, best_l = 0, None
    for l in lambdas:
        print("="*5, f"Lambda = {l}", "="*5)
        model_state_dict = server.state_dict_mul(merged_group_task_vector, l)
        model_state_dict = server.state_dict_add(model_state_dict, base_state_dict, strict=False)
        for i in involved_clients:
            for layer in model_state_dict:
                server.client_state_dicts[i][layer] = model_state_dict[layer]
        if isinstance(server, MTL_IA3_server):
            _, avg_acc, _ = server.test('clientside', client_objects, ptm_check=base_state_dict, is_val=True, n_batches=n_batches)
        elif isinstance(server, MTL_FFTNLP_server):
            _, avg_acc, _ = server.test('clientside', client_objects, ptm_check=base_state_dict, is_val=True, n_batches=n_batches)
        else:
            _, avg_acc, _ = server.test('clientside', client_objects, is_val=True, n_batches=n_batches)
        if avg_acc > best_acc:
            best_acc = avg_acc
            best_l = l
    
    ### reset the state dicts of the involved clients
    for i in involved_clients:
        for layer in merged_group_task_vector:
            server.client_state_dicts[i][layer] = client_state_dicts[i][layer]

    assert best_l is not None
    print("="*5, f"Best Lambda: {best_l}", "="*5)

    return best_l

class DisjointSetUnionWithTaskVectors:
    
    def __init__(self, n, task_vectors, sparsified_task_vectors, server, merge_method, 
                 K, n_batches, default_lambda, consensus_lambda):
        self.n = n
        self.parent = list(range(n))
        self.rank = [1] * n
        ### each cluster is a client initially so task vector for each cluster is the task vector of the client
        self.group_task_vector = {i: task_vectors[i] for i in range(n)}
        self.individual_task_vectors = task_vectors   ### task vector of each client, used for similarity computation
        self.sparsified_task_vectors = sparsified_task_vectors ### sparsified task vector of each client, used for merging
        self.server = server ### to use methods from the server class
        self.merge_method = merge_method
        self.K = K ### top K values to keep if merge_method is ties
        self.n_batches = n_batches ### how many batches to use for validation
        self.default_lambda = default_lambda ### TA, TIES and PCB lambda when not using validation set
        self.consensus_lambda = consensus_lambda ### lambda for finding consensus masks

    def find(self, x):
        # Path compression
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y, tune=False):
        # Find the root of each element
        rootX = self.find(x)
        rootY = self.find(y)
        all_task_vectors = []
        involved_clients = []
        for i in range(self.n):
            if self.find(i) == rootX or self.find(i) == rootY:
                all_task_vectors.append(self.sparsified_task_vectors[i]) ### use sparsified task vectors for merging
                involved_clients.append(i)
        
        if self.merge_method == 'avg':
            merged_group_task_vector = self.server.state_dict_avg(all_task_vectors)

        elif self.merge_method == 'ties':
            ### we use global trimming hence trim is False here
            merged_group_task_vector = self.server.state_dict_avg_ties(all_task_vectors, K=self.K, trim=False)
            if tune:
                best_coeff = tune_lambda(self.server, merged_group_task_vector, involved_clients, self.n_batches)
                merged_group_task_vector = self.server.state_dict_mul(merged_group_task_vector, best_coeff)
            else:
                merged_group_task_vector = self.server.state_dict_mul(merged_group_task_vector, self.default_lambda)

        elif self.merge_method == 'ta':
            merged_group_task_vector = self.server.state_dict_avg(all_task_vectors)
            if tune:
                best_coeff = tune_lambda(self.server, merged_group_task_vector, involved_clients, self.n_batches)
                merged_group_task_vector = self.server.state_dict_mul(merged_group_task_vector, best_coeff)
            else:
                merged_group_task_vector = self.server.state_dict_mul(merged_group_task_vector, self.default_lambda)
        
        elif self.merge_method == 'pcb':
            merged_group_task_vector = self.server.state_dict_avg_pcb(all_task_vectors, K=self.K, trim=False, \
                        involved_client_ids=involved_clients, client_min_max_dict=self.server.client_min_max_dict)
            if tune:
                best_coeff = tune_lambda(self.server, merged_group_task_vector, involved_clients, self.n_batches)
                merged_group_task_vector = self.server.state_dict_mul(merged_group_task_vector, best_coeff)
            else:
                merged_group_task_vector = self.server.state_dict_mul(merged_group_task_vector, self.default_lambda)
        
        elif self.merge_method == 'consensus_ta':
            if len(involved_clients) > 2: ### size benefit only for > 2 clients
                merged_group_task_vector = self.server.state_dict_avg(all_task_vectors)
                if tune:
                    best_coeff = tune_lambda(self.server, merged_group_task_vector, involved_clients, self.n_batches)
                    merged_group_task_vector = self.server.state_dict_mul(merged_group_task_vector, best_coeff)
                else:
                    merged_group_task_vector = self.server.state_dict_mul(merged_group_task_vector, self.default_lambda)
                masks = self.server.state_dict_consensus(all_task_vectors, merged_group_task_vector, self.consensus_lambda)
                for i in range(len(involved_clients)):
                    for layer in masks[i]:
                        client_idx = involved_clients[i]
                        self.server.client_state_masks[client_idx][layer] = masks[i][layer]
        
        elif self.merge_method == 'consensus_ties':
            if len(involved_clients) > 2: ### size benefit only for > 2 clients
                merged_group_task_vector = self.server.state_dict_avg_ties(all_task_vectors, K=self.K, trim=False)
                if tune:
                    best_coeff = tune_lambda(self.server, merged_group_task_vector, involved_clients, self.n_batches)
                    merged_group_task_vector = self.server.state_dict_mul(merged_group_task_vector, best_coeff)
                else:
                    merged_group_task_vector = self.server.state_dict_mul(merged_group_task_vector, self.default_lambda)
                masks = self.server.state_dict_consensus(all_task_vectors, merged_group_task_vector, self.consensus_lambda)
                for i in range(len(involved_clients)):
                    for layer in masks[i]:
                        client_idx = involved_clients[i]
                        self.server.client_state_masks[client_idx][layer] = masks[i][layer]
        
        elif self.merge_method == 'emr':
            if len(involved_clients) > 2: ### size benefit only for > 2 clients
                merged_group_task_vector, masks = self.server.state_dict_avg_emr(all_task_vectors)
                for i in range(len(involved_clients)):
                    for layer in masks[i]:
                        client_idx = involved_clients[i]
                        self.server.client_state_masks[client_idx][layer] = masks[i][layer]
        
        else:
            raise ValueError("Invalid merge method")

        if rootX != rootY:
            # Union by rank
            if self.rank[rootX] > self.rank[rootY]:
                self.parent[rootY] = rootX
                if not ((self.merge_method in ['consensus_ties', 'consensus_ta', 'emr']) and len(involved_clients) <= 2):
                    self.group_task_vector[rootX] =  merged_group_task_vector 
            elif self.rank[rootX] < self.rank[rootY]:
                self.parent[rootX] = rootY
                if not ((self.merge_method in ['consensus_ties', 'consensus_ta', 'emr']) and len(involved_clients) <= 2):
                    self.group_task_vector[rootY] =  merged_group_task_vector  
            else:
                self.parent[rootY] = rootX
                self.rank[rootX] += 1
                if not ((self.merge_method in ['consensus_ties', 'consensus_ta', 'emr']) and len(involved_clients) <= 2):
                    self.group_task_vector[rootX] = merged_group_task_vector 

    def get_group_task_vector(self, x):
        # Get the value associated with the group that x belongs to
        rootX = self.find(x)
        return self.group_task_vector[rootX]
    
    def get_constituent_task_vectors(self, x):
        # Get the task vectors of the clients in the group that x belongs to
        rootX = self.find(x)
        constituent_task_vectors = []
        for i in range(self.n):
            if self.find(i) == rootX:
                constituent_task_vectors.append(self.individual_task_vectors[i])
        return constituent_task_vectors

    def find_unique_groups(self):
        # Find unique roots to identify unique groups
        unique_roots = set(self.find(x) for x in range(len(self.parent)))
        return unique_roots

    def get_group_size(self, x):
        # Get the size of the group that x belongs to
        rootX = self.find(x)
        return sum(1 for i in range(self.n) if self.find(i) == rootX)