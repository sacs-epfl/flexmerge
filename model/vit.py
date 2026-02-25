from peft import LoraConfig, get_peft_model
from transformers import CLIPModel, CLIPProcessor
import os
from transformers import AutoConfig, AutoModel

def load_clip_model(model_name_or_path: str):

    processor = CLIPProcessor.from_pretrained(
        model_name_or_path
    )
    clip_model: CLIPModel = CLIPModel.from_pretrained(
        model_name_or_path
    )
    return processor, clip_model

def freeze_unless_image_model(clip_model: CLIPModel):
    """
    Freezes the parameters in a CLIP model instance other than image model.

    Args:
        clip_model (CLIPModel): A CLIP model instance.

    Returns:
        CLIPModel: The same CLIP model instance with the text model parameters frozen.
    """
    for param in clip_model.parameters():
        param.requires_grad = False
    for param in clip_model.vision_model.parameters():
        param.requires_grad = True
    for param in clip_model.visual_projection.parameters():
        param.requires_grad = False
    return clip_model

def load_clip_processor_and_model(
    model_name_or_path: str,
    lora_config=None,
    linearized_lora: bool = False,
    local_files_only=False,
    random_seed: int = 42,
):
    # L.seed_everything(random_seed)
    
    processor, clip_model = load_clip_model(model_name_or_path)
    clip_model = freeze_unless_image_model(clip_model)
    clip_vision_model = clip_model.vision_model
    clip_text_model = clip_model.text_model

    if lora_config is not None:
        # lora_config = instantiate(lora_config)
        lora_vision_model = get_peft_model(clip_vision_model, lora_config)
        # if linearized_lora:
        #     lora_vision_model = linearize_lora_model(lora_vision_model)  # Not needed for now
        lora_vision_model.print_trainable_parameters()
        clip_vision_model = lora_vision_model

    clip_model.vision_model = clip_vision_model
    clip_model.text_model = clip_text_model
    return processor, clip_model, clip_vision_model, clip_text_model

def vit_b_32(rank = 16, full_ft = False, **kwargs):
    
    if(full_ft == False):
        lora_config = LoraConfig(
                                r=rank,
                                lora_alpha = 2*rank,
                                target_modules=["q_proj", "v_proj"],
                                lora_dropout=0,
                                bias="none",
                                )   
    else:
        lora_config = None 
    
    (
        clip_processor,
        clip_model,
        clip_vision_model,
        clip_text_model,
    ) = load_clip_processor_and_model(
        'openai/clip-vit-base-patch32',
        lora_config,
        linearized_lora=False,
    )
    
    return clip_model