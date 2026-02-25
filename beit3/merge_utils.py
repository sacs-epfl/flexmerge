from collections import OrderedDict
import os
import copy
import torch


# Source: https://github.com/harveyhuang18/EMR_Merging/blob/main/merge_beit3/merge_beit.py
#   thanks
def filt_param_to_merge(pt_weight, ft_weights):
    names = []
    for n in pt_weight:
        flag = True
        for ft_weight in ft_weights:
            if n not in ft_weight or  pt_weight[n].shape != ft_weight[n].shape:
                flag = False
                break
        if flag:
            names.append(n)
    return names

def state_dict_to_vector(state_dict, selected_keys):
    shared_state_dict = copy.deepcopy(state_dict)
    for key in list(shared_state_dict.keys()):
        if key not in selected_keys:
            del shared_state_dict[key]
    sorted_shared_state_dict = OrderedDict(sorted(shared_state_dict.items()))
    return torch.nn.utils.parameters_to_vector(
        [value.reshape(-1) for key, value in sorted_shared_state_dict.items()]
    )

def vector_to_state_dict(vector, state_dict, selected_keys):
    reference_dict = copy.deepcopy(state_dict)
    for key in list(reference_dict.keys()):
        if key not in selected_keys:
            del reference_dict[key]
    sorted_reference_dict = OrderedDict(sorted(reference_dict.items()))

    # create a shared state dict using the refence dict
    torch.nn.utils.vector_to_parameters(vector, sorted_reference_dict.values())
    return sorted_reference_dict

def check_state_dicts_subset(state_dict1, state_dict2):
    if not set(state_dict1.keys()).issubset(set(state_dict2.keys())):
        return False
    for key in state_dict1.keys():
        if not torch.equal(state_dict1[key], state_dict2[key]):
            return False
    return True


def load_beit3(server):
    
    ft_checks = [client.model.state_dict() for client in server.clients]
    ptm_check = torch.load(os.path.join(server.checkpoint_dir, "beit3_base_patch16_224.pth"))["model"]
    
    keys_to_merge = filt_param_to_merge(ptm_check, ft_checks)
    
    flat_ft = torch.vstack(
        [state_dict_to_vector(check, keys_to_merge) for check in ft_checks]
    )
    flat_ptm = state_dict_to_vector(ptm_check, keys_to_merge)
    
    tv_flat_checks = flat_ft - flat_ptm
    
    
    assert check_state_dicts_subset(
        vector_to_state_dict(flat_ptm, ft_checks[0], keys_to_merge), ptm_check
    )
    assert check_state_dicts_subset(
        vector_to_state_dict(flat_ptm, ptm_check, keys_to_merge), ptm_check
    )
    assert all(
        [
            check_state_dicts_subset(
                vector_to_state_dict(flat_ft[i], ptm_check, keys_to_merge), ft_checks[i]
            )
            for i in range(len(ft_checks))
        ]
    )
    
    return (
        tv_flat_checks,
        flat_ptm,
        flat_ft,
        ft_checks,
        ptm_check,
        keys_to_merge,
    )