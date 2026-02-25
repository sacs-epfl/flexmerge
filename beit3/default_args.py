"""
This module defines the default command-line arguments for the BEiT-3 tutorials, as shown in the 
“get_started” examples on the Microsoft UNILM GitHub repository 
(https://github.com/microsoft/unilm/tree/c837c5073154f8c61d6c1929bcc4accc57b0f2c2/beit3/get_started).
"""

class BasicArgs:
	model = 'beit_base_patch16_224'
	input_size = 224
	drop_path = 0.1
	checkpoint_activations = None
	vocab_size = 64010
	num_max_bpe_tokens = 64
	model_ema = False
	model_ema_decay = 0.9999
	model_ema_force_cpu = False
	opt = 'adamw'
	opt_eps = 1e-08
	opt_betas = [0.9, 0.999]
	clip_grad = None
	momentum = 0.9
	weight_decay = 0.05
	lr = 0.0005
	layer_decay = 0.9
	task_head_lr_weight = 0
	warmup_lr = 1e-06
	min_lr = 1e-06
	warmup_epochs = 5
	warmup_steps = -1
	batch_size = 64
	eval_batch_size = None
	epochs = 20
	update_freq = 1
	save_ckpt_freq = 5
	randaug = False
	train_interpolation = 'bicubic'
	finetune = ''
	model_key = 'model|module'
	model_prefix = ''
	data_path = '/datasets01/imagenet_full_size/061417/'
	output_dir = ''
	log_dir = None
	device = 'cuda'
	seed = 0
	resume = ''
	auto_resume = True
	save_ckpt = True
	start_epoch = 0
	eval = False
	dist_eval = False
	num_workers = 10
	pin_mem = True
	world_size = 1
	local_rank = -1
	dist_on_itp = False
	dist_url = 'env://'
	task_cache_path = None
	nb_classes = 1000
	mixup = 0
	cutmix = 0
	cutmix_minmax = None
	mixup_prob = 1.0
	mixup_switch_prob = 0.5
	mixup_mode = 'batch'
	color_jitter = 0.4
	aa = 'rand-m9-mstd0.5-inc1'
	smoothing = 0.1
	crop_pct = None
	reprob = 0.25
	remode = 'pixel'
	recount = 1
	resplit = False
	captioning_mask_prob = 0.6
	drop_worst_ratio = 0.2
	drop_worst_after = 12000
	num_beams = 3
	length_penalty = 0.6
	label_smoothing = 0.1
	enable_deepspeed = False
	initial_scale_power = 16
	zero_stage = 0

class CocoCaptioningArgs(BasicArgs):
	model = 'beit3_base_patch16_480'
	task = 'coco_captioning'
	input_size = 480
	sentencepiece_model = '/your_beit3_model_path/beit3.spm'
	batch_size = 16
	finetune = '/your_beit3_model_path/beit3_base_patch16_480_coco_captioning.pth'
	data_path = '/path/to/your_data'
	output_dir = '/path/to/save/your_prediction'
	eval = True
	dist_eval = True

class ImageNet1KArgs(BasicArgs):
	model = 'beit3_base_patch16_224'
	task = 'imagenet'
	sentencepiece_model = '/your_beit3_model_path/beit3.spm'
	batch_size = 128
	finetune = '/your_beit3_model_path/beit3_base_patch16_224_in1k.pth'
	data_path = '/path/to/your_data'
	eval = True
	dist_eval = True

class NLVR2Args(BasicArgs):
	model = 'beit3_base_patch16_224'
	task = 'nlvr2'
	sentencepiece_model = '/your_beit3_model_path/beit3.spm'
	batch_size = 32
	finetune = '/your_beit3_model_path/beit3_base_patch16_224_nlvr2.pth'
	data_path = '/path/to/your_data'
	eval = True
	dist_eval = True

class CocoRetrievalArgs(BasicArgs):
	model = 'beit3_base_patch16_384'
	task = 'coco_retrieval'
	input_size = 384
	sentencepiece_model = '/your_beit3_model_path/beit3.spm'
	batch_size = 16
	finetune = '/your_beit3_model_path/beit3_base_patch16_384_coco_retrieval.pth'
	data_path = '/path/to/your_data'
	eval = True
	dist_eval = True

class VQAv2Args(BasicArgs):
	model = 'beit3_base_patch16_480'
	task = 'vqav2'
	input_size = 480
	sentencepiece_model = '/your_beit3_model_path/beit3.spm'
	batch_size = 16
	finetune = '/your_beit3_model_path/beit3_base_patch16_480_vqa.pth'
	data_path = '/path/to/your_data'
	output_dir = '/path/to/save/your_prediction'
	eval = True
	dist_eval = True


def get_default_args(task_name):
	"""
	Returns the default arguments for the specified task.

	Args:
		task_name (str): The name of the task.

	Returns:
		BasicArgs: The default arguments for the specified task.
	"""
	if task_name == 'coco_captioning':
		return CocoCaptioningArgs()
	elif task_name == 'imagenet':
		return ImageNet1KArgs()
	elif task_name == 'nlvr2':
		return NLVR2Args()
	elif task_name == 'coco_retrieval':
		return CocoRetrievalArgs()
	elif task_name == 'vqav2':
		return VQAv2Args()
	else:
		raise ValueError(f"Unknown task name: {task_name}")

def create_personalized_args(task_name, sentencepiece_model_path, finetuned_model_path, 
                            data_path, output_dir, batch_size=None, num_workers=None,
                            dist_eval=False):
	args = get_default_args(task_name)
	args.sentencepiece_model = sentencepiece_model_path
	args.finetune = finetuned_model_path
	args.data_path = data_path
	args.output_dir = output_dir
	args.dist_eval = dist_eval
	if batch_size is not None:
		args.batch_size = batch_size
	if num_workers is not None:
		args.num_workers = num_workers
	if args.task_cache_path is None:
		args.task_cache_path = args.output_dir
	return args

def is_supported_beit3_task(task_name):
	"""
	Checks if the specified task is supported by BEiT-3.

	Args:
		task_name (str): The name of the task.

	Returns:
		bool: True if the task is supported, False otherwise.
	"""
	supported_tasks = ['coco_captioning', 'imagenet', 'nlvr2', 'coco_retrieval', 'vqav2']
	return task_name in supported_tasks