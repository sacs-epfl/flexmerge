import torch
import os
import time
import copy
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from Utils.server_utils import create_server
from Utils.args_utils import get_args

n_batches=50

def build_model_from_snapshot(model, snapshot, client_id):
    state_dict = {}
    for block in snapshot:
        mapping = snapshot[block][0]
        merged_task_vector = snapshot[block][1][mapping[client_id]]
        state_dict.update(merged_task_vector)

    # Clone the model
    model_copy = copy.deepcopy(model).cuda()
    keys = set(state_dict.keys())
    for name, param in model_copy.named_parameters():
        if name in keys:
            param.data = state_dict[name]  # Shared GPU tensor — no duplication
    torch.cuda.empty_cache() 
    return model_copy

def run_client_inference(client, client_model):
    return client.test(client_model, False, n_batches=n_batches)

def run_client_inference2(client, model):
    return client.test(model, False, n_batches=n_batches)

def run_client_inference3(client, client_model):
    return client.test(client_model, False, n_batches=n_batches)

def make_balanced_batch_generator(clients, n_batches):
    loaders = [iter(client.testloader) for client in clients]
    i = 0
    while i < n_batches:
        i += 1
        try:
            batch = [next(loader) for loader in loaders]
            # Merge batches into one big batch (you may need to adjust for (X, y) pairs)
            merged_inputs = torch.cat([b[0] for b in batch], dim=0)
            merged_labels = torch.cat([b[1] for b in batch], dim=0)
            yield merged_inputs, merged_labels
        except StopIteration:
            break

def make_precomputed_balanced_batches(clients, n_batches):
    loaders = [iter(client.testloader) for client in clients]
    balanced_batches = []

    for _ in range(n_batches):
        try:
            # Pull one batch per client
            batch = [next(loader) for loader in loaders]
            # Merge inputs and labels
            merged_inputs = torch.cat([b[0] for b in batch], dim=0)
            merged_labels = torch.cat([b[1] for b in batch], dim=0)
            balanced_batches.append((merged_inputs, merged_labels))
        except StopIteration:
            break

    return balanced_batches

def run_client_inference4(clients, model):
    heads = [client.classification_head for client in clients]
    loaders = [iter(client.testloader) for client in clients]
    n = len(clients)
    with torch.no_grad():
        corrects = [0.0] * n
        counts = [0.0] * n
        B, C, H, W = 32*n, 3, 224, 224
        merged_data = torch.empty((B, C, H, W), device='cuda')
        merged_labels = torch.empty((B,), device='cuda')
        for i in tqdm(range(n_batches), total=n_batches):
            start = 0
            for b in [next(loader) for loader in loaders]:
                data, labels = b
                end = start + data.size(0)
                merged_data[start:end].copy_(data, non_blocking=True)
                merged_labels[start:end].copy_(labels, non_blocking=True)
                start = end
            outputs = model(merged_data)
            
            # split outputs by first dimension into n clients
            outputs = outputs.view(n, -1, outputs.shape[-1])
            labels = merged_labels.view(n, -1)
            
            for i in range(n):
                # apply the classification head
                preds = heads[i](outputs[i])
                preds = torch.argmax(preds, dim=-1)
                # compare with labels
                correct = (preds == labels[i]).float().sum()
                corrects[i] += correct
                counts[i] += labels[i].shape[0]

        return corrects, counts

def run(args, run_type):

    # check if args.save_dir exists
    full_save_dir = os.path.join(args.SAVEDIR, args.SAVENAME)
    size_dirs = os.listdir(full_save_dir)
    for size_dir in size_dirs:
        print(size_dir)
        if not os.path.isdir(os.path.join(full_save_dir, size_dir)):
            # print("==> not a dir, continuing")
            continue
        
        if not size_dir.startswith('s_4.77'):
            # print('==> skipping')
            continue

        snapshot = torch.load(os.path.join(full_save_dir, size_dir, 'snapshot.pt'), weights_only=True)

    n = args.NUM_CLIENTS

    # I have the snapshot now
    server = create_server(args)
    base_state_dict = server.model.state_dict().copy()
    model = server.model

    clients = server.clients
    
    if run_type == 1:
        for block in snapshot:
            merged_task_vectors = snapshot[block][1]
            for merged_task_vector in merged_task_vectors:
                for k in merged_task_vector.keys():
                    # move all the tensors into GPU
                    merged_task_vector[k] += base_state_dict[k]
                    merged_task_vector[k] = merged_task_vector[k].cuda()
        model_copies = [build_model_from_snapshot(model, snapshot, i) for i in range(n)]
    elif run_type == 2:
        model = model.cuda()
    elif run_type == 3:
        model_copies = [copy.deepcopy(model).cuda() for _ in range(n)]
    elif run_type == 4:
        model = model.cuda()
        for client in clients:
            client.classification_head = client.classification_head.cuda()

    if run_type <= 3:    
        start = time.time()
        with ThreadPoolExecutor(max_workers=8) as executor:
            if run_type == 1:
                run_fn = run_client_inference
                futures = [executor.submit(run_fn, clients[i], model_copies[i]) for i in range(n)]
            elif run_type == 2:
                run_fn = run_client_inference2
                futures = [executor.submit(run_fn, clients[i], model) for i in range(n)]
            elif run_type == 3:
                run_fn = run_client_inference3
                futures = [executor.submit(run_fn, clients[i], model_copies[i]) for i in range(n)]
            
            results = [f.result() for f in futures]
        
        end = time.time()
        time_taken = end - start
        print("Time taken for inference: ", time_taken)

        avg_acc = 0.0
        for i, (c_crt, c_ns, _) in enumerate(results):
            client_acc = c_crt / c_ns
            avg_acc += client_acc
            print("Accuracy on client {} ({}) is {:.3f}".format(i, clients[i].dataset, client_acc))

        avg_acc /= len(clients)
        print("Across clients: average accuracy {:.3f}".format(avg_acc))
    elif run_type == 4:
        start = time.time()
        corrects, counts = run_client_inference4(clients, model)
        end = time.time()
        time_taken = end - start
        print("Time taken for inference: ", time_taken)

        accs = [corrects[i] / counts[i] for i in range(n)]
        for i in range(n):
            print("Accuracy on client {} ({}) is {:.3f}".format(i, clients[i].dataset, accs[i]))
        avg_acc = sum(accs) / len(accs)
        print("Across clients: average accuracy {:.3f}".format(avg_acc))


if __name__ == '__main__':
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # used to disable warnings
    
    args = get_args() 
    run_type = 2 # 1: flexmerge, 2: single, 3: all finetuned, 4: global batch single
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    run(args, run_type)
