from typing import Optional, Union, Dict, Tuple, Any
import os
import glob
import importlib
import torch
from torch import nn
import transformers
from transformers import PretrainedConfig, PreTrainedModel, AutoModel, AutoModelForCausalLM, ProcessorMixin

from smallvlm.utils.model_utils import load_state_dict_file
from smallvlm.utils.loggings import logger
from .utils import set_requires_grad, last_layer_as_adaptor


def build_model(config: Dict[str, Any], processor: ProcessorMixin, device: Optional[Union[str, torch.device]] = None)-> PreTrainedModel:
    config_ = config
    config = config.copy()

    torch_dtype = getattr(torch, config.pop('torch_dtype'))

    # Build Vision Language Model & Config.
    pretrained_path = config.pop('pretrained_path', None)
    skip_weight_loading = pretrained_path is not None

    # Build Language Model & Config.
    language_model_config = config.pop('language_model').copy()
    language_model_config.pop('freeze', None)
    language_model_config.pop('trainable_params', None)
    language_model, language_model_config = build_lm(language_model_config, torch_dtype=torch_dtype, device=device, skip_weight_loading=skip_weight_loading)
    if len(processor.tokenizer) > language_model.vocab_size:
        old_vocab_size = language_model.vocab_size
        language_model.resize_token_embeddings(len(processor.tokenizer), pad_to_multiple_of=64)
        logger.warning("The vocabulary size of tokenizer is larger than that of language model. "
                       "The original token_embeddings of the model is resized to match the vocabulary size: "
                       f"{old_vocab_size} -> {language_model.vocab_size}")

    # Build Vision Model & Config.
    vision_model_config = config.pop('vision_model').copy()
    vision_model_config.pop('freeze', None)
    vision_model_config.pop('trainable_params', None)
    vision_model_config.pop('last_layer_as_adaptor', None)
    vision_model, vision_model_config = build_vision(vision_model_config, torch_dtype=torch_dtype, device=device, skip_weight_loading=skip_weight_loading)

    # Build Vision Abstractor Config.
    vision_abstractor_config_ = config.pop('vision_abstractor', None)
    if vision_abstractor_config_:
        vision_abstractor_config = vision_abstractor_config_.copy()
        vision_abstractor_config.pop('freeze', None)
        vision_abstractor_config.pop('trainable_params', None)

    # Get Vision Language Model & Config Class.
    architecture = config.pop('architecture', None)
    if architecture is None:
        raise ValueError("Model architecture is not given.")
    VLM = getattr(importlib.import_module(f"smallvlm.models.{'.'.join(architecture.split('.')[:-1])}"), architecture.split('.')[-1])
    VLMConfig = VLM.config_class

    # Build Vision Language Model & Config.
    lora_configs = config.pop('lora', None)
    prompt_numbers=config.pop('prompt_numbers', 15)
    roi_output_size=config.pop('roi_output_size', None)

    model_config: PretrainedConfig = VLMConfig(language_model_config=language_model_config,
                                               vision_model_config=vision_model_config,
                                               vision_abstractor_config=vision_abstractor_config,
                                               torch_dtype=torch_dtype,
                                               image_token_id=processor.image_token_id,
                                               prompt_numbers=prompt_numbers,
                                               roi_output_size=roi_output_size,
                                               **config)

    if 'pdmllm' in architecture:
        model: PreTrainedModel = VLM(model_config, language_model=language_model, vision_model=vision_model, processor=processor)
    else:
        model: PreTrainedModel = VLM(model_config, language_model=language_model, vision_model=vision_model)
        
    model.generation_config = model.language_model.generation_config

    # Prepare The Model For Training.
    prepare_model_for_training(model, config_)
    # print_trainable_modules(model)

    model.to(device=device, dtype=torch_dtype)

    if pretrained_path is not None:
        load_state_dict_file(pretrained_path, model=model)
        print(f'Loaded Model: {pretrained_path}')

    if lora_configs is not None:
        from .lora import make_lora
        for lora_config in lora_configs:
            model = make_lora(model, lora_config, keep_requires_grad=True)

    torch.cuda.empty_cache()
    return model


def prepare_model_for_training(model: nn.Module, config: dict):
    set_requires_grad(model.language_model,
                      config['language_model'].get('freeze', 0),
                      config['language_model'].get('trainable_params', None))

    set_requires_grad(model.vision_model,
                      config['vision_model'].get('freeze', 0),
                      config['vision_model'].get('trainable_params', None))

    set_requires_grad(model.vision_abstractor,
                      config['vision_abstractor'].get('freeze', 0),
                      config['vision_abstractor'].get('trainable_params', None))

    if config['vision_model'].get('last_layer_as_adaptor', False):
        last_layer_as_adaptor(model.vision_model)


def build_lm(config: Dict, torch_dtype: Optional[str] = None, device: Optional[torch.device] = None, skip_weight_loading: bool = False):
    def getCausalLM(architecture=None):
        if architecture is None:
            Model = AutoModelForCausalLM
        elif isinstance(architecture, str):
            try:
                Model = getattr(importlib.import_module(f"smallvlm.models.{'.'.join(architecture.split('.')[:-1])}"),
                                architecture.split('.')[-1])
            except:
                Model = getattr(transformers, architecture, AutoModelForCausalLM)
        elif isinstance(architecture, PreTrainedModel):
            Model = architecture
        else:
            raise TypeError(f"Invalid architecture type: {architecture}={type(architecture)}")

        return Model

    if config.get('name_or_path', None):
        architecture = config.pop('architecture', None)
        Model = getCausalLM(architecture)
        name_or_path = config.pop('name_or_path')

        print(architecture, name_or_path)
        if skip_weight_loading:
            from transformers import AutoConfig
            model_config = AutoConfig.from_pretrained(name_or_path, trust_remote_code=True, **config)
            with transformers.modeling_utils.no_init_weights():
                model = Model(model_config)
            
            # optionally set torch_dtype? Not entirely needed strictly here, as to(dtype) happens later.
            if torch_dtype is not None:
                model = model.to(dtype=torch_dtype)
        else:
            model = Model.from_pretrained(name_or_path, torch_dtype=torch_dtype, device_map=device, trust_remote_code=True, **config)
            model_config = model.config

    else:
        raise NotImplementedError(config)

    return model, model_config


def build_vision(config: Dict, torch_dtype: Optional[str] = None, device: Optional[torch.device] = None, skip_weight_loading: bool = False):
    def getVisionModel(architecture=None):
        if architecture is None:
            Model = AutoModel
        elif isinstance(architecture, str):
            Model = getattr(transformers, architecture)
        elif isinstance(architecture, PreTrainedModel):
            Model = architecture
        else:
            raise TypeError(f"Invalid architecture type: {architecture}={type(architecture)}")

        return Model

    architecture = config.pop('architecture', None)
    if architecture:
        ModelClass = getattr(importlib.import_module(f"smallvlm.models.{'.'.join(architecture.split('.')[:-1])}"), architecture.split('.')[-1])
        ConfigClass = ModelClass.config_class

        pretrained_path = config.pop('pretrained_path', None)
        model_config = ConfigClass(**config)
        model = ModelClass(model_config)
        if pretrained_path is not None:
            load_state_dict_file(pretrained_path, model=model)
        # model.gradient_checkpointing = True
    else:
        name_or_path = config.get('name_or_path', None)
        assert name_or_path is not None
        if skip_weight_loading:
            from transformers import AutoConfig
            model_config = AutoConfig.from_pretrained(name_or_path, trust_remote_code=True, **config)
            with transformers.modeling_utils.no_init_weights():
                model = AutoModel.from_config(model_config)
            if torch_dtype is not None:
                model = model.to(dtype=torch_dtype)
        else:
            model = AutoModel.from_pretrained(name_or_path, torch_dtype=torch_dtype, trust_remote_code=True, **config)
            model_config = model.config
    return model, model_config

def print_trainable_modules(model: nn.Module):
    logger.info(">>> Trainable Modules:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(f"  {name}: {param.shape}")