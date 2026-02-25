import os
import numpy as np
import torch 
from torch.utils.data import Subset, ConcatDataset 
from torch.utils.data import random_split 
from torch.utils.data import DataLoader 
import torchvision.transforms as transforms 
import torchvision.datasets as datasets 
from torchvision.datasets import CIFAR10, MNIST, CIFAR100, EMNIST, SVHN, StanfordCars, GTSRB, ImageFolder, DTD
from Utils.utils import rand_mask
from Data.labels_for_datasets import get_labels
# from Data.resisc45 import get_resisc_dataset
import matplotlib.pyplot as plt
import time

transform_cifar100 = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                        std=[0.2675, 0.2565, 0.2761])])  

def get_public_dataloaders(args):
    if args.SEED is None:
        args.SEED = np.random.randint(0, 2**32-1)
    np.random.seed(args.SEED)
    torch.manual_seed(args.SEED)    

    print("Using public dataset, DATAPATH is: {}".format(args.PUBLIC_DATAPATH))
    trainset = CIFAR100(args.PUBLIC_DATAPATH, train=True, download=args.DOWNLOAD, transform=transform_cifar100)
    
    # sample a public training set with 2000 data points
    num_samples = 2000
    indices = np.random.choice(len(trainset), num_samples, replace=False)

    subset = Subset(trainset, indices)

    num_workers=4
    p_dataloader = DataLoader(subset, batch_size=args.BATCH_SIZE, shuffle=True, num_workers=num_workers, drop_last=True)
    
    return p_dataloader

def get_dataloaders(args):
    if args.SEED is None:
        args.SEED = np.random.randint(0, 2**32-1)
    np.random.seed(args.SEED)
    torch.manual_seed(args.SEED)
    print("Preparing dataset, DATAPATH is: {}".format(args.DATAPATH))
    
    trainset, testset, args = get_datasets(args)

    clustertruths = dict()
    clustertruths[0] = [i for i in range(args.NUM_CLIENTS)]
    clustersets = None

    if args.DLTYPE == "equal": 
        print('using equal split')
        datasets, data_inds = balanced_split(trainset, args)
        testsets, data_inds = balanced_split(testset, args)
        labels = []

    elif args.DLTYPE == "dirichlet":
        alpha = args.ALPHA
        print("Using dirichlet split, alpha is {}".format(alpha))
        datasets, data_inds = dirichlet_split(trainset, args, alpha)
        labels = []
        if(args.DATATYPE == 'stanford_cars' or args.DATATYPE == 'gtsrb' or args.DATATYPE == 'dtd'):
            for inds in data_inds:
                labels.append([args.trainset_labels[i] for i in inds])
        elif(args.DATATYPE == 'svhn'):
            for inds in data_inds:
                labels.append([trainset.labels[i] for i in inds])
        else:
            for inds in data_inds:
                labels.append([trainset.targets[i] for i in inds])
        label_dists = []
        weights = []
        for (i,label) in enumerate(labels):
            label_counts = np.bincount(label)
            label_distribution = label_counts / len(label)
            label_dists.append(label_distribution)
            weights.append(len(label)/ len(trainset))
            # print ("Label distribution at client {}  is ".format(i), label_distribution)
        args.WEIGHTS = weights
        testsets = []
        for label_dist, weight in zip(label_dists, weights):
            testsets.append(create_testing_subset(args, testset, label_dist, weight)) 
    
    elif args.DLTYPE == "class_split":

        beta = args.cs_BETA
        print("Using class split, beta is {}".format(beta))
        datasets, testsets = class_split(trainset, testset, args, beta)
    
    elif args.DLTYPE == "cluster_dirichlet":
        alpha = args.ALPHA
        n_clusters = args.N_CLUSTER
        n_class_per_cluster = args.N_CLASS_PER_CLUSTER
        uniform_ratio = args.UNIFORM_RATIO
        overlapping = args.OVERLAP

        print("Using cluster based dirichlet split, alpha ia {}".format(alpha))
        
        label_groups = generate_label_groups(args, n_clusters, n_class_per_cluster, overlapping)
        print("Label groups: {}, overlapping: {}".format(label_groups, overlapping))

        datasets, testsets, clustersets, dicts = cluster_dirichlet_split(trainset, testset, \
        args, alpha, label_groups, n_clusters, n_class_per_cluster, uniform_ratio)
        clustertruths = dicts
    
    else:
        raise NotImplementedError(f"Dataset split {args.DLTYPE} not implemented as it is not supported, check config.yaml!")

    # Visualize the distribution of the generated federated data
    # if args.DLTYPE != "equal" and args.SAVE_DATA:
    #     visualize_distribution(datasets, args)
    # print_flag = False
    # if print_flag:
    #     visualize_distribution(datasets, args)
        # pdb.set_trace()
    # visualize_distribution(datasets, args)
    
    print("Total number of federated data: {}".format(sum(len(datasets[i]) for i in range(len(datasets)))))
    trainloaders = []
    valloaders = []
    localloaders = []

    for i, (ds ,ts) in enumerate(zip(datasets, testsets)):
        v_max, v_min = 1.0, 0.2
        # current_value = v_max - (v_max - v_min) * i / len(datasets)
        current_value = 1.0

        L = len(ds)
        len_val = round(L*args.VALRATIO)
        len_train = L - len_val    
        len_local = len(ts)

        lengths = [len_train, len_val]
        sidx1 = rand_mask(len_train, args.SAMPLE_RATIO*current_value) 
        sidx2 = rand_mask(len_val, args.SAMPLE_RATIO*current_value) 
        sidxt = rand_mask(len_local, max(args.SAMPLE_RATIO, 1.0)) #0.2

        ds_train, ds_val = random_split(ds, lengths, torch.Generator().manual_seed(int(args.SEED)))
        train_sampled = Subset(ds_train, sidx1)
        val_sampled = Subset(ds_val, sidx2)
        test_sampled = Subset(ts, sidxt)

        if(i == 0):
            global_val_sampled = val_sampled
        else:
            global_val_sampled = ConcatDataset([global_val_sampled, val_sampled])

        if len(train_sampled)==0 or len(val_sampled)==0:
            raise NotImplementedError("Dataloader not implemented, as len_train or len_test equals is 0.")

        num_worker = 1
        trainloaders.append(DataLoader(train_sampled, batch_size=args.BATCH_SIZE, shuffle=True, num_workers=num_worker, persistent_workers=True))
        valloaders.append(DataLoader(val_sampled, batch_size=args.BATCH_SIZE, num_workers=num_worker))
        localloaders.append(DataLoader(test_sampled, batch_size=args.BATCH_SIZE, num_workers=num_worker))
    
    sidx_test = rand_mask(len(testset), max(args.SAMPLE_RATIO, 0.5))
    test_sampled = Subset(testset, sidx_test)
    testloaders = []
    if clustersets is not None:
        for cs in clustersets:
            sidxc = rand_mask(len(cs), max(args.SAMPLE_RATIO,0.2)) #0.2
            cluster_sampled = Subset(cs, sidxc)
            testloaders.append(DataLoader(cluster_sampled, batch_size=args.BATCH_SIZE, num_workers=num_worker))
    testloaders.append(DataLoader(test_sampled, batch_size=args.BATCH_SIZE, num_workers=num_worker))

    sidx1 = rand_mask(len(global_val_sampled), 0.3)
    global_val_subsampled = Subset(global_val_sampled, sidx1)
    print("Size of valiation data is", len(global_val_subsampled))
    valloaders.append(DataLoader(global_val_subsampled, batch_size=args.BATCH_SIZE, num_workers=num_worker))

    print("We got dataloaders...")

    args.trainloaders, args.valloaders, args.ctestloaders, args.stestloaders, args.gids_star = \
        trainloaders, valloaders, localloaders,  testloaders, clustertruths

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict

# helpers - get data
def get_datasets(args):
    
    trainset, testset = None, None
    
    if args.DATATYPE == "cifar10":
        # prepare for potential augmentation purpose
        if(args.MODEL != 'vit-b-32'):   
            if args.IMG_SIZE == 32: # To match gFed's data augmentation pipeline
                transform_test_cifar10 = transforms.Compose(
                [transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
                transform_aug = transform_test_cifar10
            else:
                transform_aug = transforms.Compose([
                transforms.RandomResizedCrop(args.IMG_SIZE),
                transforms.RandomHorizontalFlip(),
                # transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
                ]) 
                transform_test_cifar10 = transforms.Compose(
                [transforms.CenterCrop(args.IMG_SIZE),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),]) 
        else:
            transform_aug = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),])
            
            transform_test_cifar10 = transform_aug
            

        trainset = CIFAR10(args.DATAPATH, train=True, download=args.DOWNLOAD , transform=transform_aug)
        testset = CIFAR10(args.DATAPATH, train=False, download=args.DOWNLOAD, transform=transform_test_cifar10)
        args.NUM_CLASS = 10
        args.class_names = unpickle(os.path.join(args.DATAPATH,'cifar-10-batches-py','batches.meta'))['label_names']
        args.trainset_labels = trainset.targets
        args.testset_labels = testset.targets

    elif args.DATATYPE == "cifar100":
        transform_aug = transforms.Compose([
            transforms.RandomCrop(args.IMG_SIZE, padding=4) if args.IMG_SIZE==32 else transforms.RandomResizedCrop(args.IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            # transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        transform_test_cifar100 = transforms.Compose(
            [transforms.CenterCrop(args.IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),]) # [0.4914, 0.4822, 0.4465], ([0.2470, 0.2435, 0.2616]

        trainset = CIFAR100(args.DATAPATH, train=True, download=args.DOWNLOAD, transform=transform_aug)
        testset = CIFAR100(args.DATAPATH, train=False, download=args.DOWNLOAD, transform=transform_test_cifar100)
        args.NUM_CLASS = 100
        args.class_names =  unpickle(os.path.join(args.DATAPATH,'cifar-100-python','meta'))['fine_label_names']
        args.trainset_labels = trainset.targets.tolist()
        args.testset_labels = testset.targets.tolist()

    elif args.DATATYPE == "tiny-imagenet":
        raise NotImplementedError("Tiny-ImageNet is not supported yet.")
        train_dir = args.DATAPATH +'train'
        test_dir = '/fs/scratch/rng_cr_pj_ai_r24_gpu_user_c_lf/datasets/tinyimagenet_val/val/images'
        transform = transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        trainset = ImageFolder(train_dir, transform=transform)
        testset = ImageFolder(test_dir, transform=transform)    
        args.NUM_CLASS = 200

    elif args.DATATYPE == "mnist":
        if(args.MODEL != 'vit-b-32'):  
            transform_mnist = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        else: ### ViT-B-16 requires images to be of size 224x224x3
             transform_mnist = transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
                
        trainset = MNIST(args.DATAPATH,  train=True, download=True, transform=transform_mnist)
        testset = MNIST(args.DATAPATH, train=False, download=True, transform=transform_mnist)
        args.NUM_CLASS = 10
        args.trainset_labels = trainset.targets.tolist()
        args.testset_labels = testset.targets.tolist()

    elif args.DATATYPE == "stanford_cars":
        if(args.MODEL != 'vit-b-32'):  
            raise ValueError(f"Specify transformation for Cars dataset")
        else: ### ViT-B-16 requires images to be of size 224x224x3
             transform_cars = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
        trainset = StanfordCars(args.DATAPATH,  split="train", download=False, transform=transform_cars)
        testset = StanfordCars(args.DATAPATH, split = "test", download=False, transform=transform_cars)
        args.NUM_CLASS = 196
        args.trainset_labels, args.testset_labels = get_labels("stanford_cars")

    elif args.DATATYPE == "sun397":
        raise NotImplementedError("SUN397 is not supported yet.")
        if(args.MODEL != 'vit-b-32'):  
            raise ValueError(f"Specify transformation for SUN dataset")
        else: ### ViT-B-16 requires images to be of size 224x224x3
             transform_sun397 = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])

        trainset = datasets.ImageFolder("Data/SUN397/train", transform=transform_sun397)
        testset = datasets.ImageFolder("Data/SUN397/test", transform=transform_sun397)
        args.NUM_CLASS = 397
        args.trainset_labels, args.testset_labels = get_labels("sun397")

    elif args.DATATYPE == "eurosat":
        raise NotImplementedError("EuroSAT is not supported yet.")
        if(args.MODEL != 'vit-b-32'):  
            raise ValueError(f"Specify transformation for SUN dataset")
        else: ### ViT-B-16 requires images to be of size 224x224x3
             transform_eurosat = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
        trainset = datasets.ImageFolder("Data/EuroSAT_splits/train", transform=transform_eurosat)
        testset = datasets.ImageFolder("Data/EuroSAT_splits/test", transform=transform_eurosat)
        args.NUM_CLASS = 10
        args.trainset_labels = trainset.targets
        args.testset_labels = testset.targets

    elif args.DATATYPE == "dtd":
        raise NotImplementedError("DTD is not supported yet.")
        if(args.MODEL != 'vit-b-32'):  
            raise ValueError(f"Specify transformation for DTD dataset")
        else: ### ViT-B-16 requires images to be of size 224x224x3
             transform_dtd = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
             
        trainset = datasets.ImageFolder("Data/dtd/train", transform=transform_dtd)
        testset = datasets.ImageFolder("Data/dtd/val", transform=transform_dtd)
        args.NUM_CLASS = 47
        args.trainset_labels = trainset.targets
        args.testset_labels = testset.targets
     
    elif args.DATATYPE == "gtsrb":
        if(args.MODEL != 'vit-b-32'):  
            raise ValueError(f"Specify transformation for GTSRB dataset")
        else: ### ViT-B-16 requires images to be of size 224x224x3
             transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])

        trainset = GTSRB(args.DATAPATH, split="train", download=True, transform=transform)
        testset = GTSRB(args.DATAPATH, split = "test", download=True, transform=transform)
        args.NUM_CLASS = 43
        args.trainset_labels, args.testset_labels = get_labels("gtsrb")
      
    elif args.DATATYPE == "svhn":
        if(args.MODEL != 'vit-b-32'):  
            transform_svhn = transforms.Compose(
            [transforms.CenterCrop(args.IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),])
        else: ### ViT-B-16 requires images to be of size 224x224x3
             transform_svhn = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
                
        trainset = SVHN(args.DATAPATH,  split = 'train', download=True, transform=transform_svhn)
        testset = SVHN(args.DATAPATH,  split = 'test', download=True, transform=transform_svhn)
        args.NUM_CLASS = 10
        args.trainset_labels = trainset.labels.tolist()
        args.testset_labels = testset.labels.tolist()
        
    elif args.DATATYPE == 'emnist':
        transform_emnist = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.174,), (0.331,))
        ])

        trainset = EMNIST(args.DATAPATH, split='balanced', train=True, \
                          download=args.DOWNLOAD, transform=transform_emnist)
        testset = EMNIST(args.DATAPATH, split='balanced', train=False, \
                         download=args.DOWNLOAD, transform=transform_emnist)
        args.NUM_CLASS = 47

    else:
        raise NotImplementedError(f"Incorrect dataset {args.DATATYPE} or not supported yet.")
    
    return trainset, testset, args

def balanced_split(dataset, args):
    data_inds = None
    partition_size = len(dataset) // args.NUM_CLIENTS
    lengths = [partition_size] * args.NUM_CLIENTS
    idx = range(sum(lengths))
    dataset_s = Subset(dataset, idx)
    return random_split(dataset_s, lengths, torch.Generator().manual_seed(int(args.SEED))), data_inds

def dirichlet_split(dataset, args, alpha):
    
    num_classes = args.NUM_CLASS
    num_clients = args.NUM_CLIENTS
    samples_per_class = len(dataset) // num_classes # only works if each class has the same number of samples
    min_samples = args.MIN_SAMPLES_PER_CLIENT

    if(args.DATATYPE == 'stanford_cars' or args.DATATYPE == 'gtsrb' or args.DATATYPE == 'dtd'):
        class_inds = [np.random.permutation(np.nonzero(args.trainset_labels == i)[0]) for i in np.arange(num_classes)] # [num_classes][samples_per_class]

    elif(args.DATATYPE == 'svhn'):
        if torch.is_tensor(dataset.labels):
            dataset.labels = dataset.labels.tolist()

        class_inds = [np.random.permutation(np.nonzero(dataset.labels == i)[0]) for i in np.arange(num_classes)] # [num_classes][samples_per_class]
    else:
        if torch.is_tensor(dataset.targets):
            dataset.targets = dataset.targets.tolist()
        class_inds = [np.random.permutation(np.nonzero(dataset.targets == i)[0]) for i in np.arange(num_classes)] # [num_classes][samples_per_class]
    # print(class_inds)

    #step 1. Allocate min_samples to each client. Distribute those randomly regardless of class.
    # assert (min_samples * num_clients) % num_classes == 0, f"(min_samples * num_clients) needs to be divisible by num_classes."
    init_samples_per_class = (min_samples * num_clients) // num_classes
    initial_inds = np.random.permutation(np.concatenate([inds[:init_samples_per_class] for inds in class_inds]))
    class_inds = [inds[init_samples_per_class:] for inds in class_inds]
    samples_per_class -= init_samples_per_class

    data_inds = [[initial_inds[i*min_samples:(i+1)*min_samples]] for i in range(num_clients)]

    # step 2. Distribute the rest of the samples via a Dirichlet distribution
    sample_fractions = np.random.dirichlet(alpha=[alpha]*num_clients, size=num_classes) # [num_classes, num_clients]
    sample_counts = np.round(sample_fractions * samples_per_class).astype(int) # [num_classes, num_clients]

    # adjust the number of samples for the biggest client to make the total number of samples correct (it can be incorrect due to the float -> int conversion)
    sample_counts[np.arange(num_classes),np.argmax(sample_counts, axis=1)] += samples_per_class - sample_counts.sum(axis=1)

    sample_count_cumsum = np.concatenate([np.zeros((num_classes, 1), dtype=int), np.cumsum(sample_counts, axis=1)], axis=1) # [num_classes, num_clients + 1]
    # assert np.all(sample_count_cumsum[:,-1] == samples_per_class) # check that counts for each class add up to the total number of sampels per class
    
    for class_n in range(num_classes):
        for client_n in range(num_clients):
            if sample_count_cumsum[class_n, client_n] < sample_count_cumsum[class_n, client_n+1]:
                data_inds[client_n].append(class_inds[class_n][sample_count_cumsum[class_n, client_n]:sample_count_cumsum[class_n, client_n+1]])

    data_inds = [np.concatenate(x)  for x in data_inds]

    # print(f"dataset sizes: {[len(x) for x in data_inds]}")

    return [Subset(dataset, inds) for inds in data_inds], data_inds

def class_split(dataset, testset, args, beta):

    classes_per_client = beta
    num_classes = args.NUM_CLASS
    num_clients = args.NUM_CLIENTS
    samples_per_client = len(dataset) // num_clients
    samples_per_client_per_class = samples_per_client // classes_per_client

    samples_test_per_client = len(testset) // num_clients
    samples_test_per_client_per_class = samples_test_per_client // classes_per_client
    assert (num_clients * classes_per_client) % num_classes == 0

    class_inds = [np.random.permutation(np.nonzero(dataset.targets == i)[0]) for i in np.arange(num_classes)]
    class_test_inds = [np.random.permutation(np.nonzero(testset.targets == i)[0]) for i in np.arange(num_classes)] 

    client_datasets = []
    client_test_datasets = []


    for _ in range(num_clients):
        labels_used = []
        client_inds = []
        client_test_inds = []
        for _ in range(classes_per_client):
            # randomly picks a class from the classes that have the most samples left
            max_len = max(len(class_inds[i]) for i in range(len(class_inds)) if i not in labels_used)
            label = np.random.choice([i for i in range(len(class_inds)) if (len(class_inds[i]) == max_len) and (i not in labels_used)])
            labels_used.append(label)
            client_inds.append(class_inds[label][:samples_per_client_per_class])
            class_inds[label] = class_inds[label][samples_per_client_per_class:]
            client_test_inds.append(class_test_inds[label][:samples_test_per_client_per_class])
            class_test_inds[label] = class_test_inds[label][samples_test_per_client_per_class:]

        
        client_inds = np.concatenate(client_inds)
        client_test_inds = np.concatenate(client_test_inds)
        client_datasets.append(Subset(dataset, client_inds))
        client_test_datasets.append(Subset(testset, client_test_inds))

    # checking that all classes are used equally
    #assert np.all(np.unique(all_labels_used, return_counts=True)[1] == (num_clients * classes_per_client) // num_classes), np.unique(all_labels_used, return_counts=True)
    return client_datasets, client_test_datasets

def generate_label_groups(args, num_clusters, classes_per_cluster, overlapping):
    num_labels = args.NUM_CLASS
    if overlapping:
        assert classes_per_cluster * num_clusters >= num_labels, "Number of clusters * classes per cluster should be greater than or equal to the number of labels."
    else:
        assert num_clusters * classes_per_cluster == num_labels, "Number of clusters * classes per cluster should be equal to the number of labels."

    labels = np.arange(num_labels)
    np.random.shuffle(labels)

    if not overlapping:
        return [list(labels[i * classes_per_cluster:(i + 1) * classes_per_cluster]) for i in range(num_clusters)]

    label_groups = [set() for _ in range(num_clusters)]

    # Assign each label to a random cluster at least once
    for label in labels:
        random_cluster = np.random.randint(num_clusters)
        label_groups[random_cluster].add(label)

    remaining_slots = classes_per_cluster * num_clusters - num_labels

    # Ensure at least one label in each cluster
    for i, group in enumerate(label_groups):
        if not group:
            random_label = np.random.choice(labels)
            label_groups[i].add(random_label)
            remaining_slots -= 1

    # Assign remaining labels randomly to the clusters
    for _ in range(remaining_slots):
        random_label = np.random.choice(labels)
        random_cluster = np.random.randint(num_clusters)
        label_groups[random_cluster].add(random_label)

    label_groups = [list(group) for group in label_groups]

    return label_groups

def create_clusters(dataset, args, n_clusters, label_groups, uniform_fraction=0.1):
    class_indices = [[] for _ in range(args.NUM_CLASS)]

    if torch.is_tensor(dataset.targets):
        dataset.targets = dataset.targets.tolist()

    for idx, target in enumerate(dataset.targets):
        class_indices[target].append(idx)

    clusters = []

    for label_group in label_groups:
        major_label_indices = []
        other_label_indices = []
        for label in range(args.NUM_CLASS):
            if label in label_group:
                major_label_indices.extend(class_indices[label])
            else:
                other_label_indices.extend(class_indices[label])

        np.random.shuffle(other_label_indices)
        num_uniform_samples = int(len(major_label_indices) * uniform_fraction / (1 - uniform_fraction))
        cluster_indices = major_label_indices + other_label_indices[:num_uniform_samples]

        np.random.shuffle(cluster_indices)
        clusters.append(Subset(dataset, cluster_indices))

    return clusters

def dirichlet_split_cluster(cluster_data, label_group, num_clients, alpha, uniform_ratio):
    num_classes = len(np.unique(cluster_data.dataset.targets))
    main_classes = len(label_group)
    uniform_classes = num_classes - main_classes
 
    main_class_inds = [np.random.permutation([i for i in cluster_data.indices if cluster_data.dataset.targets[i] == c]) for c in label_group if len([i for i in cluster_data.indices if cluster_data.dataset.targets[i] == c]) > 0]
    uniform_class_inds = [np.random.permutation([i for i in cluster_data.indices if cluster_data.dataset.targets[i] == c]) for c in range(num_classes) if c not in label_group ]
    

    main_class_samples = np.random.dirichlet(alpha=[alpha] * num_clients, size=main_classes) 
    v_min, v_max = main_classes / (num_clients*20), 1.0
    main_class_samples = (v_max - v_min) * main_class_samples + v_min
    main_class_samples_counts = np.round(main_class_samples * np.array([len(inds) for inds in main_class_inds]).reshape(-1,1)).astype(int)
    main_class_samples_counts[:, np.argmax(main_class_samples_counts, axis=1)] += [len(inds) for inds in main_class_inds] - main_class_samples_counts.sum(axis=1)


    cumsum = np.cumsum(main_class_samples_counts, axis=1)
    indices = np.argsort(cumsum, axis=1)
    cumsum_sorted = np.take_along_axis(cumsum, indices, axis=1)

    sample_count_cumsum_main = np.concatenate([np.zeros((main_classes, 1), dtype=int), cumsum_sorted], axis=1)
    
    # pdb.set_trace()

    uniform_num = 0
    for i in range(len(uniform_class_inds)):
        uniform_num += len(uniform_class_inds[i])
    uniform_samples_per_client = uniform_num // (uniform_classes * num_clients)


    data_inds = [[] for _ in range(num_clients)]
    for client_idx in range(num_clients):
        for class_idx in range(main_classes):
            if sample_count_cumsum_main[class_idx, client_idx] < sample_count_cumsum_main[class_idx, client_idx + 1]:
                data_inds[client_idx].extend(main_class_inds[class_idx][sample_count_cumsum_main[class_idx, client_idx]:sample_count_cumsum_main[class_idx, client_idx + 1]])
                

        for class_idx in range(uniform_classes):
            num_samples = uniform_samples_per_client
            data_inds[client_idx].extend(uniform_class_inds[class_idx][:num_samples])
            uniform_class_inds[class_idx] = uniform_class_inds[class_idx][num_samples:]

    return [Subset(cluster_data.dataset, inds) for inds in data_inds]

def cluster_dirichlet_split(dataset, testset, args, alpha, label_groups, num_clusters, classes_per_cluster, uniform_fraction):
    num_labels = args.NUM_CLASS

    clusters = create_clusters(dataset, args, num_clusters, label_groups, uniform_fraction)
    federated_data = []

    labels = []
    for cluster in clusters:
        labels.append([dataset.targets[i] for i in cluster.indices])

    label_dists = []
    weights= []
    for label in labels:
        label_counts = np.bincount(label)
        label_distribution = label_counts / len(label)
        label_dists.append(label_distribution)
        weights.append(len(label) / len(dataset))

    clutsertests = []
    for label_dist, weight in zip(label_dists, weights):
        clutsertests.append(create_testing_subset(testset, label_dist, weight))

    m_clients = args.NUM_CLIENTS // num_clusters
    for cluster, label_group in zip(clusters, label_groups):
        client_data = dirichlet_split_cluster(cluster, label_group, m_clients, alpha, uniform_fraction)
        federated_data.extend(client_data)

    test_data = [cluster_test for cluster_test in clutsertests for _ in range(m_clients)]

    dicts = {i:[j for j in range(m_clients*i, m_clients*(i+1))] for i in range(num_clusters)}
    

    return federated_data, test_data, clutsertests, dicts

# this function is used to visulize the generated data distribution 
def visualize_distribution(federated_data, args):
    num_clients = args.NUM_CLIENTS
    num_classes = args.NUM_CLASS
    distribution_matrix = np.zeros((num_clients, num_classes), dtype=int)

    for client_idx, client_data in enumerate(federated_data):
        for idx in client_data.indices:
            target = client_data.dataset.targets[idx]
            distribution_matrix[client_idx, target] += 1

    plt.figure(figsize=(12, 8))
    num_clients, num_classes = distribution_matrix.shape
    x, y = np.meshgrid(range(num_classes), range(num_clients))
    sizes = distribution_matrix / distribution_matrix.max() * 1000  # Normalize the matrix values and scale to control the point sizes

    scatter = plt.scatter(x, y, s=sizes, alpha=0.5)
    # sns.heatmap(distribution_matrix, annot=True, cmap="YlGnBu", fmt='d', cbar=False)
    plt.xlabel('Class Labels')
    plt.ylabel('Client Index')
    # plt.title('Distribution of Samples per Client and Class')
    plt.title('Distribution of Samples per Client and Class, alpha=0.01')
    # plt.show()
    figure_name = 'figures/'+"visual_current_run_data_{}_{}.pdf".format(args.DLTYPE, str(int(time.time())))   
    plt.savefig(figure_name)

def create_testing_subset(args, dataset, label_dist, weight):

    
    # if(args.DATATYPE == 'stanford_cars' or args.DATATYPE == 'gtsrb'):
    test_labels = np.unique(args.testset_labels)
    label_counts = np.round(label_dist * len(args.testset_labels) * weight).astype(int)
    inds = []
    
    for label, target_count in zip(test_labels, label_counts):
        
        label_idx = np.nonzero(args.testset_labels == label)[0]
        target_count = min(target_count, len(label_idx))
        selected_idx = np.random.choice(label_idx, size=target_count, replace=False)
        inds.extend(selected_idx)
    
    # elif(args.DATATYPE == 'svhn'):
    #     test_labels = np.unique(dataset.labels)
    #     label_counts = np.round(label_dist * len(dataset.labels) * weight).astype(int)
    #     inds = []
        
    #     for label, target_count in zip(test_labels, label_counts):

            
    #         label_idx = np.nonzero(dataset.labels == label)[0]
    #         target_count = min(target_count, len(label_idx))
    #         selected_idx = np.random.choice(label_idx, size=target_count, replace=False)
    #         inds.extend(selected_idx)

    # else:
    #     test_labels = np.unique(dataset.targets)
    #     label_counts = np.round(label_dist * len(dataset.targets) * weight).astype(int)
    #     inds = []
        
    #     for label, target_count in zip(test_labels, label_counts):
            
    #         if(args.DATATYPE == 'mnist'):
    #             label_idx = np.nonzero(dataset.targets == label).flatten()  # for mnist
    #         else:
    #             label_idx = np.nonzero(dataset.targets == label)[0]        # for cifar10
    #         # 
    #         # print (label_idx)
    #         selected_idx = np.random.choice(label_idx, size=target_count, replace=False)
    #         inds.extend(selected_idx)
        
    np.random.shuffle(inds)
    
    return Subset(dataset, inds)

