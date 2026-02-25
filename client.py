import torch
from typing import Tuple, Union
import torch.nn.functional as F
import time
import torch.nn as nn
import copy
import numpy as np
import time
from tqdm import tqdm
import sys
import os
import wandb
import datetime
sys.path.append('../')

from model.mtl_models import ImageClassifier, ImageEncoder
from model.heads import get_classification_head
from Data.my_datasets.common import maybe_dictionarize
from Utils.mtl_datasets import get_dataset
from Utils.distributed import cleanup_ddp, distribute_loader, is_main_process, setup_ddp
from Utils.utils import LabelSmoothing, cosine_lr
from Utils.variables_and_paths import get_finetuned_path, get_zeroshot_path

from timm.models import create_model
import beit3.modeling_finetune
from beit3.engine_for_finetuning import get_handler
from beit3.engine_for_finetuning import evaluate as evaluate_beit3
from beit3.datasets import create_downstream_dataset
import beit3.utils as beit3_utils


def _warmup_lr(base_lr: float, warmup_length: int, step_idx: int):
    return base_lr * (step_idx + 1) / warmup_length

def _cos_lr(base_lr: float, max_steps: int, step_idx: int):
    lr = 0.5 * (1 + np.cos(np.pi * step_idx / max_steps)) * base_lr
    return lr

class CosineAnnealingWithWarmup:
    R"""
    a `max_steps`-step cosine annealing learning rate schedule with `warmup_steps` warm-up steps.
    The `step(step_idx)` method should be called every update step.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        base_lrs: Union[float, Tuple[float]],
        warmup_steps: int,
        max_steps: int,
    ):
        super().__init__()
        self.optimizer = optimizer
        if isinstance(base_lrs, (float, int)):
            base_lrs = tuple(base_lrs for _ in optimizer.param_groups)
        self.base_lrs = base_lrs
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {
            key: value for key, value in self.__dict__.items() if key != "optimizer"
        }

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_lrs(self):
        return [group["lr"] for group in self.optimizer.param_groups]

    def step(self, step_idx: int = 0):
        warmup_length = self.warmup_steps
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            if step_idx < warmup_length:
                lr = _warmup_lr(base_lr, warmup_length, step_idx)
            else:
                lr = _cos_lr(
                    base_lr, self.max_steps - warmup_length, step_idx - warmup_length
                )
            param_group["lr"] = lr  # assign learning rate

        self._last_lr = [
            param_group["lr"] for param_group in self.optimizer.param_groups
        ]
            
class Client:
    
    def __init__(self, args, id, trainloader, valloader, testloader):

        self.device = args.DEVICE
        self.seed = args.SEED
        self.args = args

        self.dataset = args.DATATYPE
        self.decimal = args.SAVE_DECIMAL


        self.local_epochs = args.LOCAL_EPOCH
        self.num_rounds = args.ROUNDS
        self.batch_size = args.BATCH_SIZE
        self.learning_rate = args.LEARNING_RATE
        

        self.algo = args.METHOD

        self.id = id 
        self.train_samples = len(trainloader.dataset)
        self.test_samples = len(valloader.dataset)

        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader

        self.max_steps = self.local_epochs*len(self.trainloader)
        self.init_loss_fn()

    def init_loss_fn(self):
        self.loss = nn.CrossEntropyLoss() 
        self.dist_loss = nn.MSELoss()
        self.ensemble_loss=nn.KLDivLoss(reduction="batchmean")
        self.ce_loss = nn.CrossEntropyLoss()
            
    def train(self, base_model, text_embeds, iter, personalized=False):


        self.scheduler = 'cosine_annealing'
        model = copy.deepcopy(base_model) 
        
        # Overwriting the default optimizer with that used in the PETA paper
        self.optimizer = torch.optim.AdamW(
        [p for p in model.vision_model.parameters() if p.requires_grad],
        lr= self.args.LEARNING_RATE,
        weight_decay = 0.1)

        self.lr_scheduler = CosineAnnealingWithWarmup(
            self.optimizer,
            base_lrs= self.args.LEARNING_RATE,
            warmup_steps= 0,
            max_steps= self.max_steps,
        )

        # print ("Learning rate is ", self.args.LEARNING_RATE)

        # finetuning
        step_idx = 0
        model.to(self.device)
        model.vision_model.train()  # Training only the vision model part of the Vision Transformer
        
            
        print ("Training on Client ", self.id)
        start = time.time()

        if personalized:
            n_epochs = 1
        else:
            n_epochs = self.args.LOCAL_EPOCH


        for _ in tqdm(range(n_epochs)):
            correct, total_eg, batch_loss = 0, 0, 0.0
            epoch_loss = []
            epoch_acc = []  
            for images, labels in self.trainloader:
                
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                image_embeds = model.get_image_features(pixel_values=images)
            

                # normalized features
                image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)

                # cosine similarity as logits
                logit_scale = model.logit_scale.exp().item()
                logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
                logits_per_image = logits_per_text.t()

                loss = F.cross_entropy(logits_per_image, labels)
                
                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step(step_idx)
                step_idx += 1
                
                batch_loss += loss.detach()
                total_eg += labels.size(0)
                
                pred = logits_per_image.argmax(dim=-1)
                correct += (pred == labels).sum().item()


                epoch_loss.append(batch_loss.detach().cpu().item()/len(self.trainloader))
                epoch_acc.append(correct/total_eg)
    
         
        epoch_loss = round(sum(epoch_loss) / len(epoch_loss), self.decimal)
        epoch_acc = round(sum(epoch_acc) / len(epoch_acc), self.decimal)
        print ("Final Epoch Loss is", epoch_loss)
        print ("Final Epoch Accuracy is", epoch_acc)
        end = time.time()
        print ("Time taken for training is ", end-start)

        client_model = dict(
                    (k, p.detach().cpu())
                    for k, p in model.named_parameters()
                    if p.requires_grad)

#         # Clearing GPU cache
        del self.optimizer
        del model
        torch.cuda.empty_cache() 

        # print (client_lora_model.keys())
      
        

        return epoch_loss, epoch_acc, client_model

class MTL_Client:
    
    def __init__(self, args, id, train_preprocess, val_preprocess, dataset):

        self.device = args.DEVICE
        self.seed = args.SEED
        self.args = args

        self.dataset = dataset
        self.decimal = args.SAVE_DECIMAL

        self.local_epochs = args.LOCAL_EPOCH
        self.num_rounds = args.ROUNDS
        self.batch_size = args.BATCH_SIZE
        self.learning_rate = args.LEARNING_RATE
        
        self.algo = args.METHOD
        self.id = id 

        ## Provides the testloader, hence using val_preprocess
        dset = get_dataset(self.dataset, val_preprocess, args.DATAPATH, batch_size=args.BATCH_SIZE, num_workers=4)
        ## Provides the trainloader and valoader, hence using train_preprocess
        dset_val = get_dataset(self.dataset + 'Val', train_preprocess, args.DATAPATH, batch_size=args.BATCH_SIZE, num_workers=4)
        self.classification_head = get_classification_head(args.MODEL, args.MODELDIR, args.DATAPATH, dataset)

        self.train_samples = len(dset_val.train_dataset)
        self.val_samples = len(dset_val.test_dataset)
        self.test_samples = len(dset.test_dataset)

        self.trainloader = dset_val.train_loader
        self.valloader = dset_val.test_loader_shuffle
        self.testloader = dset.test_loader

        self.max_steps = self.local_epochs*len(self.trainloader)
        self.init_loss_fn()
        print("="*5, f"Created client {self.id} for dataset {self.dataset}")
        print("="*5, f"Client {self.id} has {self.train_samples} training samples")
        print("="*5, f"{self.val_samples} validation samples and {self.test_samples} test samples")

    def init_loss_fn(self):
        self.loss = nn.CrossEntropyLoss() 
        self.dist_loss = nn.MSELoss()
        self.ensemble_loss=nn.KLDivLoss(reduction="batchmean")
        self.ce_loss = nn.CrossEntropyLoss()
        
    def finetune(self, rank, args):
        setup_ddp(rank, args.world_size, port=args.port)

        if is_main_process():
            # initialize wandb
            wandb.init(
                config=args,
                project=args.WANDB_PROJECT, 
                entity=args.WANDB_ENTITY,
                name=args.SAVENAME + "_" + datetime.datetime.now().strftime("%d/%m/%Y-%H:%M:%S"),
            )

        train_dataset = args.DATATYPE

        ft_path = get_finetuned_path(args.MODELDIR, train_dataset, args.MODEL)
        zs_path = get_zeroshot_path(args.MODELDIR, train_dataset, args.MODEL)

        if os.path.exists(zs_path) and os.path.exists(ft_path):
            if is_main_process():
                print(f"Skipping fine-tuning because {ft_path} exists.")
            return

        # create a new model for training
        image_encoder = ImageEncoder(args.MODEL)
        model = ImageClassifier(image_encoder, copy.deepcopy(self.classification_head))

        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"The total number of trainable parameters is {num_params/1e6:.2f}M")

        model.freeze_head()
        model = model.cuda()

        print_every = 100

        data_loader = self.trainloader
        num_batches = len(data_loader)

        # Distribute the data and model across the GPUs.
        ddp_loader = distribute_loader(data_loader)
        ddp_model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[rank], find_unused_parameters=True, output_device=rank
        )

        print("Hello from process", rank)

        if args.ls > 0:
            loss_fn = LabelSmoothing(args.ls)
        else:
            loss_fn = torch.nn.CrossEntropyLoss()

        params = [p for p in ddp_model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=args.LEARNING_RATE, weight_decay=args.wd)
        scheduler = cosine_lr(
            optimizer, args.LEARNING_RATE, args.warmup_length, args.LOCAL_EPOCH * num_batches // args.num_grad_accumulation
        )

        # Saving zero-shot model
        if is_main_process():
            model_path = get_zeroshot_path(args.MODELDIR, train_dataset, args.MODEL)
            ckpdir = os.path.dirname(model_path)
            print("Dir to save the model: ", ckpdir)
            os.makedirs(ckpdir, exist_ok=True)
            # check if the model exists
            if not os.path.exists(model_path):
                ddp_model.module.image_encoder.save(model_path)

        for epoch in range(args.LOCAL_EPOCH):
            ddp_model.train()

            torch.distributed.barrier()

            for i, batch in enumerate(ddp_loader):
                start_time = time.time()

                step = i // args.num_grad_accumulation + epoch * num_batches // args.num_grad_accumulation

                batch = maybe_dictionarize(batch)
                inputs = batch["images"].cuda()
                labels = batch["labels"].cuda()
                data_time = time.time() - start_time

                logits = ddp_model(inputs)
                loss = loss_fn(logits, labels)
                loss.backward()

                if (i + 1) % args.num_grad_accumulation == 0:
                    scheduler(step)

                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                batch_time = time.time() - start_time

                if args.checkpoint_every > 0 and step % args.checkpoint_every == 0 and is_main_process():
                    print("Saving checkpoint.")
                    model_path = get_finetuned_path(args.MODELDIR, train_dataset, args.MODEL).replace(
                        ".pt", f"_{step}.pt"
                    )
                    ddp_model.module.image_encoder.save(model_path)

                if step % print_every == 0 and ((i + 1) % args.num_grad_accumulation == 0) and is_main_process():
                    percent_complete = 100 * i / len(ddp_loader)
                    print(
                        f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{num_batches}]\t"  # noqa: E501
                        f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}\t",  # noqa: E501
                        flush=True,
                    )
                    wandb.log(
                        {
                            f"{train_dataset}/train/loss": loss.item(),
                            "train/data_time": data_time,
                            "train/batch_time": batch_time,
                        }
                    )

            torch.distributed.barrier()

            if is_main_process():
                # We only need to evaluate the model on the first GPU.
                sd = ddp_model.module.image_encoder.state_dict()
                # independent copy for testing, avoiding ddp issues, FIX later
                image_encoder = ImageEncoder(args.MODEL)
                image_encoder.load_state_dict(sd)
                nc, tc, avg_loss = self.test(image_encoder, is_val=False)
                test_acc = nc / tc
                print(f"Test accuracy: {test_acc:.2%}, test loss: {avg_loss:.4f}")
                wandb.log(
                    {
                        f"{train_dataset}/test/accuracy": test_acc, 
                        f"{train_dataset}/test/loss": avg_loss
                    }
                )

                # empty cache
                torch.cuda.empty_cache()

        torch.distributed.barrier()

        if is_main_process():
            # get directory to save the model from ft_path
            ckpdir = os.path.dirname(ft_path)
            print("Dir to save the model: ", ckpdir)
            os.makedirs(ckpdir, exist_ok=True)

            image_encoder.save(ft_path)
            return

        cleanup_ddp()

    def finetune_simple(self, args):
        # Initialize wandb if enabled
        wandb.init(
            config=args,
            project=args.WANDB_PROJECT,
            entity=args.WANDB_ENTITY,
            name=args.SAVENAME + "_" + datetime.datetime.now().strftime("%d/%m/%Y-%H:%M:%S"),
        )

        train_dataset = args.DATATYPE
        ft_path = get_finetuned_path(args.MODELDIR, train_dataset, args.MODEL)
        zs_path = get_zeroshot_path(args.MODELDIR, train_dataset, args.MODEL)

        if os.path.exists(zs_path) and os.path.exists(ft_path):
            print(f"Skipping fine-tuning because {ft_path} exists.")
            return

        # Create and configure the model
        image_encoder = ImageEncoder(args.MODEL)
        model = ImageClassifier(image_encoder, copy.deepcopy(self.classification_head))

        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"The total number of trainable parameters is {num_params/1e6:.2f}M")

        model.freeze_head()
        model = model.cuda()

        print_every = 100
        data_loader = self.trainloader
        num_batches = len(data_loader)

        if args.ls > 0:
            loss_fn = LabelSmoothing(args.ls)
        else:
            loss_fn = torch.nn.CrossEntropyLoss()

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=args.LEARNING_RATE, weight_decay=args.wd)
        scheduler = cosine_lr(
            optimizer,
            args.LEARNING_RATE,
            args.warmup_length,
            args.LOCAL_EPOCH * num_batches // args.num_grad_accumulation,
        )

        # Save the zero-shot model
        model_path = get_zeroshot_path(args.MODELDIR, train_dataset, args.MODEL)
        ckpdir = os.path.dirname(model_path)
        print("Dir to save the model: ", ckpdir)
        os.makedirs(ckpdir, exist_ok=True)
        if not os.path.exists(model_path):
            model.image_encoder.save(model_path)

        for epoch in range(args.LOCAL_EPOCH):
            model.train()
            for i, batch in enumerate(data_loader):
                start_time = time.time()
                step = i // args.num_grad_accumulation + epoch * num_batches // args.num_grad_accumulation

                batch = maybe_dictionarize(batch)
                inputs = batch["images"].cuda()
                labels = batch["labels"].cuda()
                data_time = time.time() - start_time

                logits = model(inputs)
                loss = loss_fn(logits, labels)
                loss.backward()

                if (i + 1) % args.num_grad_accumulation == 0:
                    scheduler(step)
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                batch_time = time.time() - start_time

                if step % print_every == 0 and ((i + 1) % args.num_grad_accumulation == 0):
                    percent_complete = 100 * i / len(data_loader)
                    print(
                        f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{num_batches}]\t"
                        f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}\t",
                        flush=True,
                    )
                    wandb.log(
                        {
                            f"{train_dataset}/train/loss": loss.item(),
                            "train/data_time": data_time,
                            "train/batch_time": batch_time,
                        }
                    )

            # Test the model after each epoch
            model.eval()
            with torch.no_grad():
                sd = model.image_encoder.state_dict()
                image_encoder = ImageEncoder(args.MODEL)
                image_encoder.load_state_dict(sd)
                nc, tc, avg_loss = self.test(image_encoder, is_val=False)
                test_acc = nc / tc
                print(f"Test accuracy: {test_acc:.2%}, test loss: {avg_loss:.4f}")
                wandb.log(
                    {
                        f"{train_dataset}/test/accuracy": test_acc,
                        f"{train_dataset}/test/loss": avg_loss,
                    }
                )

            # Clear cache
            torch.cuda.empty_cache()

        # Save the fine-tuned model
        ckpdir = os.path.dirname(ft_path)
        print("Dir to save the model: ", ckpdir)
        os.makedirs(ckpdir, exist_ok=True)
        image_encoder.save(ft_path)

    def train(self):
        print("=" * 100)
        print(f"Finetuning {self.args.MODEL} on {self.dataset}")
        print("=" * 100)
        if self.dataset == 'PCAM': ### there is some issue with pickling the PCAM dataset during distributed process launch
            self.finetune_simple(self.args)
        else:
            torch.multiprocessing.spawn(self.finetune, args=(self.args,), nprocs=self.args.world_size)

    @torch.no_grad()
    def test(self, base_model, is_val=False, n_batches=-1):
        model = ImageClassifier(base_model, self.classification_head)
        model.to(self.device)
        model.eval()

        nc, tc, avg_loss = 0, 0, 0.0
        if is_val:
            data_loader = self.valloader
        else:
            data_loader = self.testloader
        
        idx = 0
        for images, labels in tqdm(data_loader):
            images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
            # start = time.time()
            logits = model(images)
            # end = time.time()
            # print(f"Time taken for inference: {end-start:.4f} seconds")
            loss = self.ce_loss(logits, labels)
            pred = logits.argmax(dim=-1)
            correct = (pred == labels).sum().item()
            avg_loss += loss.detach().cpu().item()
            nc += correct
            tc += labels.size(0)
            
            idx += 1
            if n_batches > 0 and idx >= n_batches - 1:
                break

        avg_loss /= idx
        
        # model.to("cpu")
        # torch.cuda.empty_cache() 
        
        return nc, tc, avg_loss

### Minimalistic definition to keep the pattern, most of the logic is in the server
class MTL_IA3_Client:
    def __init__(self, id, dataset):
        self.id = id
        self.dataset = dataset

class MTL_FFTNLP_Client:
    def __init__(self, id, dataset):
        self.id = id
        self.dataset = dataset

class MTL_BEIT3_Client:
    def __init__(self, id, args, model_config):
        self.id = id
        self.dataset = args.task
        self.args = args
        self._task_handler = get_handler(self.args)
        
        self.model = create_model(
            model_config,
            pretrained=False,               
            vocab_size=args.vocab_size,           
            checkpoint_activations=False 
        )
        beit3_utils.load_model_and_may_interpolate(self.args.finetune, self.model, "model|module", "")
    
    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict, strict=False)
        self.model = self.model.half()
    
    def eval(self, device):
        self.model.to(device)
        data_loader_test = create_downstream_dataset(self.args, is_eval=True)
        if self.args.task in ["nlvr2", "flickr30k", "coco_retrieval", "imagenet"]:
            ext_test_stats, task_key = evaluate_beit3(data_loader_test, self.model, device, self._task_handler)
            print(f"Task {self.args.task}: {ext_test_stats}")
            # print(f"External test stats: {ext_test_stats}")
            return ext_test_stats
        elif self.args.task == "vqav2":
            raise ValueError("VQAv2 is not supported.")
            # result, _ = evaluate_beit3(data_loader_test, self.model, device, self._task_handler)
            # return result
        elif self.args.task in ["coco_captioning", "nocaps"]:
            predictions, _ = evaluate_beit3(data_loader_test, self.model, device, self._task_handler)
            prediction_file = beit3_utils.dump_predictions(self.args, predictions, "{}_test".format(self.args.task))
            if beit3_utils.is_main_process() and self.args.task == "coco_captioning":
                captioning_result = beit3_utils.coco_caption_eval(self.args.data_path, prediction_file, "{}_test".format(self.args.task))
                print("Captioning result: ", captioning_result)
                return captioning_result
