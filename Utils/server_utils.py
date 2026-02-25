from Server.fedavg_server import FedAvg_server
from Server.mtl_server import MTL_server
from Server.mtl_ia3_server import MTL_IA3_server
from Server.mtl_fft_nlp_server import MTL_FFTNLP_server
from Server.mtl_beit3_server import MTL_BEIT3_server
from model.vit import vit_b_32
from model.mtl_models import ImageEncoder
from Utils.variables_and_paths import MODELS
from Utils.get_dataloaders import get_dataloaders
from NLP.train.TrainingConfig import TrainingConfig
from NLP.train.ModelConfig import ModelConfig
from NLP.model.load_model import construct_model
from beit3.default_args import is_supported_beit3_task

import os

SERVERS = {
    'FedAvg': FedAvg_server,
    'MTL': MTL_server,
    'MTL_IA3': MTL_IA3_server,
    'MTL_FFTNLP': MTL_FFTNLP_server,
    'MTL_BEIT3': MTL_BEIT3_server,
}

def create_server(args):

    kwargs = {
        "split": "validation",
        "should_save_to_gcp": False,
        "world_size": None,
        "eval_template_idx": -1,
        "experiment_name": args.SAVENAME,
        "pretrained_model": "bigscience/T0_3B" if (args.MODEL == 'T0_3B' or args.MODEL is None) else args.MODEL,
        "max_datapoints_per_dataset_without_templates": 1000
    }

    algo = args.METHOD
    print("Algo is: ", algo)
    if algo == 'FedAvg':
        get_dataloaders(args) 

    if args.MODEL == 'vit-b-32':
        model = vit_b_32(num_classes=args.NUM_CLASS, rank = 0, full_ft = True)
    elif args.MODEL in MODELS:
        model = ImageEncoder(args.MODEL)
    elif args.MODEL == 'T0_3B':
        if args.IA3_CONFIGFILE is None:
            raise FileNotFoundError("IA3 config file not found")
        config_toInit = TrainingConfig(
            config_filepaths=[args.IA3_CONFIGFILE], kwargs=kwargs, create_expDir=True
        )
        model_config = ModelConfig(
            configDict_toInitializeFrom=config_toInit.get_dict(),
        )
        model, tokenizer, _ = construct_model(
            model_config.pretrained_model,
            model_config.peft_method,
            model_config.max_seq_len,
            device="cpu",
            model_config=model_config,
        )
    elif args.MODEL in ["t5-base", "t5-large"]:
        if args.FFT_CONFIGFILE is None:
            raise FileNotFoundError("Full-fine tuning config file not found")
        for col in ["eval_template_idx", "max_datapoints_per_dataset_without_templates"]:
            if col in kwargs:
                del kwargs[col]
        config_toInit = TrainingConfig(
            config_filepaths=[args.FFT_CONFIGFILE], kwargs=kwargs, create_expDir=True
        )
        model_config = ModelConfig(
            configDict_toInitializeFrom=config_toInit.get_dict(),
        )
        kwargs["eval_template_idx"] = config_toInit.eval_template_idx
        model, tokenizer, _ = construct_model(
            model_config.pretrained_model,
            model_config.peft_method,
            model_config.max_seq_len,
            device="cpu",
            model_config=model_config,
        )
    elif args.MODEL == "beit3":
        eval_tasks = list(map(lambda s: s.strip(), args.BEIT3_EVAL_TASKS.split(",")))
        checkpoint_dir = os.path.abspath(args.BEIT3_CHECKPOINT_DIR.strip())
        base_data_path = os.path.abspath(args.DATAPATH.strip())
        eval_config_path = os.path.abspath(args.BEIT3_EVAL_CONFIG_FILE.strip())
    else:
        raise ValueError(f"Invalid model name: {args.MODEL}")
        
    if algo not in SERVERS:
        raise ValueError(f"Invalid algo name: {algo}")
    elif algo == 'MTL_IA3':
        return SERVERS['MTL_IA3'](args, model, tokenizer, config_toInit, model_config, kwargs)
    elif algo == 'MTL_FFTNLP':
        return SERVERS['MTL_FFTNLP'](args, model, tokenizer, config_toInit, model_config, kwargs)
    elif algo == 'MTL_BEIT3':
        return SERVERS['MTL_BEIT3'](args, eval_tasks, checkpoint_dir, base_data_path, eval_config_path)
    else:
        return SERVERS[algo](args, model)