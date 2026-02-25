from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--METHOD', type=str, default='FedAvg', help='Input methods') ### for server averaging
    parser.add_argument('--DEVICE', type=str, default='cuda:0', help='Device to run the experiment')
    parser.add_argument('--DOWNLOAD', type = bool, default= False, help='Download the dataset')
    parser.add_argument('--DATATYPE', type=str, default='svhn', help='dataset used in the experiment')
    parser.add_argument('--DATAPATH', type=str, default='Data', help='path to store the dataset')
    parser.add_argument('--VALRATIO', type=float, default=0.3, help='Validation ratio')
    parser.add_argument('--SAMPLE_RATIO', type=float, default=1.0, help='Sample ratio')
    parser.add_argument('--MIN_SAMPLES_PER_CLIENT', type=int, default=8, help='Minimum number of samples per client')
    parser.add_argument('--SAVE_DECIMAL', type=int, default=5, help='Decimal to save the results')
    parser.add_argument('--MODEL', type=str, default='vit-b-32', help='Model used in the experiment')
    parser.add_argument('--SEED', type=int, default=42, help='seed for data partitioning')
    parser.add_argument('--SEED2', type=int, default=43, help='seed for randomness in merging algorithm')
    parser.add_argument('--RUN_TYPE', type=str, default='train', choices=['train', 'merge'], help='Type of run')
    parser.add_argument('--ALG', type=str, default='greedy', choices=['left-right', 'right-left', 'greedy', 'cluster-merge', 'random', 'channel-merge'], 
                        help='Algorithm for merging the models')

    # path args
    parser.add_argument('--SAVEDIR', type=str, default=None, help='folder to store .txt results')
    parser.add_argument('--SAVENAME', type=str, default=None, help='name of the folder to store the results')
    parser.add_argument('--MODELDIR', type=str, default=None, help='folder where trained local models are stored')
    parser.add_argument('--IA3_CONFIGFILE', type=str, default=None, help='config file for IA3')
    parser.add_argument('--FFT_CONFIGFILE', type=str, default=None, help='config file for NLP models trained with full-fine tuning')

    # data args FL
    parser.add_argument('--NUM_CLIENTS', type=int, default=20, help='The number of models to be trained/fused')
    parser.add_argument('--DLTYPE', type=str, default='dirichlet', help="equal, dirichlet, class_split, cluster_dirichlet")
    parser.add_argument('--ALPHA', type=float, default=0.3, help='Dirichlet alpha, cluster_dirichlet alpha')
    parser.add_argument('--IMG_SIZE', type=int, default=32, help="Size of input image")

    # common args
    parser.add_argument('--MERGE_ITER', type=int, default=100, help='Number of iterations to merge the models')
    parser.add_argument('--SCALE', type=float, default=1.0, help='Scaling applied to the client updates')
    parser.add_argument('--LOCAL_EPOCH', type=int, default=20, help='Number of local epoch')
    parser.add_argument('--ROUNDS', type=int, default=0, help='Number of global rounds of merging')
    parser.add_argument('--BATCH_SIZE', type=int, default=32, help='Number of batch size')
    parser.add_argument('--SAMRATE_CLIENTS', type=float, default=1.0, help="Percentage of clients participate in each round")
    parser.add_argument('--MIN_CLIENTS', type=int, default=1, help='Number of min client participate in each round')
    parser.add_argument('--LEARNING_RATE', type=float, default=1e-5, help='Learning step of optimizer')
    parser.add_argument('--LR_DECAY', type=float, default=0.99, help='Learning rate decay')
    parser.add_argument('--SAVE_DATA', action='store_true', help='Save data visualization')
    parser.add_argument('--SAVE_CLIENT_MODELS', type = bool, default = True, help='Save client models')
    parser.add_argument('--GPU_MERGE', action='store_true', help='Use GPU for merging')
    parser.add_argument('--NO_EVAL', action='store_true', help='Do not evaluate the merged model')

    # merge args
    parser.add_argument('--BLOCK_GRANULARITY', type=str, default='transformer', help='granularity of blocks in the model')
    parser.add_argument('--MERGE_METHOD', type=str, default='avg', help='How to merge model blocks - avg, ta or ties')
    parser.add_argument('--K', type=float, default=1.0, help='Top K values to keep in ties, specify between 0 and 1')
    parser.add_argument('--USE_VAL', action='store_true', help='Use validation set for merging')
    parser.add_argument('--TUNE_START', type=float, default=0, help='After what fraction of merge iterations to start tuning from [0, 1]')
    parser.add_argument('--N_VAL_BATCHES', type=int, default=-1, help='How many validation batches per dataset to use for tuning, use -1 for all')
    parser.add_argument('--LAMBDA', type=float, default=1.0, help='default value of lambda to use when not using validation set')
    parser.add_argument('--EVAL_AFTER', type=float, default=0.01, help='After what fraction of merge iterations to evaluate the model')
    parser.add_argument('--DISTANCE', type=str, default='cosine', choices=['cosine', 'euclidean', 'sign'], help='Distance metric to use for merging')
    parser.add_argument('--USE_CONSTITUENTS_DISTANCE', action='store_true', help='Compute distance between constituents instead of the merged parameters')
    parser.add_argument('--DARE', action='store_true', help='Use DARE pre-processing before for merging')
    parser.add_argument('--DARE_P', type=float, default=0.9, help='DARE drop rate')

    # wandb args
    parser.add_argument('--WANDB', action='store_true', help='Use wandb')
    parser.add_argument('--WANDB_PROJECT', type=str, default='yourwanbproject', help='wandb project name')
    parser.add_argument('--WANDB_ENTITY', type=str, default='yourentity', help='wandb entity name')
    
    # splitted block args
    parser.add_argument('--N_MLP_GROUPS', type=int, default=2, help='Number of groups to divide MLP blocks into')

    # local training MTL
    parser.add_argument("--wd", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--port", type=int, default=12345, help="Port for distributed training")
    parser.add_argument("--ls", type=float, default=0.0, help="Label smoothing")
    parser.add_argument("--warmup_length", type=int, default=500, help="Warmup length")
    parser.add_argument("--num_grad_accumulation", type=int, default=1, help="Number of gradient accumulation steps")
    parser.add_argument("--world_size", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--checkpoint-every", type=int, default=-1, help="How often to checkpoint the model.")

    # consensus
    parser.add_argument('--CONSENSUS_LAMBDA', type=float, default=0.3, choices=[0.2, 0.3, 0.4, 0.5, 0.6], \
                        help='Lambda for finding consensus masks')
    parser.add_argument('--SNAPSHOT', action='store_true', help='Store snapshots of the merged model and the masks')
    parser.add_argument('--MEMORY_EFFICIENT_SNAPSHOT', action='store_true', help='Store snapshots of the DSU data structure')

    # for cluster-merge
    parser.add_argument('--LINKAGE', type=str, default='average', choices=['average', 'single', 'complete'], \
                        help='Linkage method for clustering')
    parser.add_argument('--LAYERWISE_DISTANCE', action='store_true', help='Use layerwise distance for clustering')
    
    # for beit3
    parser.add_argument('--BEIT3_EVAL_TASKS', type=str, default='', 
                        help='Tasks to use for evaluation of BEiT3 models. They should be separated by commas (Example: coco_captioning,nlvr2)')
    parser.add_argument('--BEIT3_CHECKPOINT_DIR', type=str, default='',
                        help='Path to the checkpoint and sentencepiece model directory of BEiT3 models. The checkpoints names must match names specified in default_args.py')
    parser.add_argument('--BEIT3_EVAL_CONFIG_FILE', type=str, default='',
                        help='Path to the config file for BEiT3 evaluation. The config file should contains batch size and num_workers for each task and dist_eval flag value')

    args = parser.parse_args() 
    return args