import os
import math
import torch
import warnings
from typing import Dict, List, Tuple, Optional, Any
import re
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer
from transformers.trainer import is_sagemaker_mp_enabled

from smallvlm.utils.loggings import logger
from .utils import Timer
from .trainer_callback import ProgressCallback, TimingProgressCallback

def get_parameter_names(model, forbidden_layer_types, forbidden_layer_names=None):
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    """
    forbidden_layer_patterns = (
        [re.compile(pattern) for pattern in forbidden_layer_names] if forbidden_layer_names is not None else []
    )
    result = []
    for name, child in model.named_children():
        child_params = get_parameter_names(child, forbidden_layer_types, forbidden_layer_names)
        result += [
            f"{name}.{n}"
            for n in child_params
            if not isinstance(child, tuple(forbidden_layer_types))
            and not any(pattern.search(f"{name}.{n}".lower()) for pattern in forbidden_layer_patterns)
        ]
    # Add model specific parameters that are not in any child
    result += [
        k for k in model._parameters if not any(pattern.search(k.lower()) for pattern in forbidden_layer_patterns)
    ]

    return result

def get_decay_parameter_names(model) -> list[str]:
        """
        Get all parameter names that weight decay will be applied to.

        This function filters out parameters in two ways:
        1. By layer type (instances of layers specified in ALL_LAYERNORM_LAYERS)
        2. By parameter name patterns (containing 'bias', or variation of 'norm')
        """
        forbidden_name_patterns = [r"bias", r"layernorm", r"rmsnorm", r"(?:^|\.)norm(?:$|\.)", r"_norm(?:$|\.)"]
        decay_parameters = get_parameter_names(model, [nn.LayerNorm], forbidden_name_patterns)
        return decay_parameters

def get_module_param_groups(
    model,
    args,
    default_lr: float,
    weight_decay: float,
) -> List[Dict[str, Any]]:
    """
    Create parameter groups with different learning rates for different modules.
    
    Args:
        model: The model with vision_model, vision_abstractor, and language_model
        args: Training arguments containing module-specific learning rates
        default_lr: Default learning rate to use if module-specific lr is not set
        weight_decay: Weight decay value
    
    Returns:
        List of parameter group dictionaries
    """
    # Get module-specific learning rates, fallback to default if not specified
    vision_model_lr = getattr(args, 'vision_model_lr', None) or default_lr
    vision_abstractor_lr = getattr(args, 'vision_abstractor_lr', None) or default_lr
    language_model_lr = getattr(args, 'language_model_lr', None) or default_lr
    
    # Collect parameters for each module
    vision_model_params = []
    vision_model_params_no_decay = []
    vision_abstractor_params = []
    vision_abstractor_params_no_decay = []
    language_model_params = []
    language_model_params_no_decay = []
    other_params = []
    other_params_no_decay = []
    
    # Get parameter names that should have weight decay applied
    decay_parameters = get_decay_parameter_names(model)
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        # Check if this parameter should skip weight decay
        skip_decay = name not in decay_parameters
        
        if name.startswith("vision_model."):
            if skip_decay:
                vision_model_params_no_decay.append(param)
            else:
                vision_model_params.append(param)
        elif name.startswith("vision_abstractor."):
            if skip_decay:
                vision_abstractor_params_no_decay.append(param)
            else:
                vision_abstractor_params.append(param)
        elif name.startswith("language_model."):
            if skip_decay:
                language_model_params_no_decay.append(param)
            else:
                language_model_params.append(param)
        else:
            if skip_decay:
                other_params_no_decay.append(param)
            else:
                other_params.append(param)
    
    param_groups = []
    
    if vision_model_params:
        param_groups.append({
            "params": vision_model_params,
            "lr": vision_model_lr,
            "weight_decay": weight_decay,
            "name": "vision_model"
        })
    if vision_model_params_no_decay:
        param_groups.append({
            "params": vision_model_params_no_decay,
            "lr": vision_model_lr,
            "weight_decay": 0.0,
            "name": "vision_model_no_decay"
        })
    
    if vision_abstractor_params:
        param_groups.append({
            "params": vision_abstractor_params,
            "lr": vision_abstractor_lr,
            "weight_decay": weight_decay,
            "name": "vision_abstractor"
        })
    if vision_abstractor_params_no_decay:
        param_groups.append({
            "params": vision_abstractor_params_no_decay,
            "lr": vision_abstractor_lr,
            "weight_decay": 0.0,
            "name": "vision_abstractor_no_decay"
        })
    
    if language_model_params:
        param_groups.append({
            "params": language_model_params,
            "lr": language_model_lr,
            "weight_decay": weight_decay,
            "name": "language_model"
        })
    if language_model_params_no_decay:
        param_groups.append({
            "params": language_model_params_no_decay,
            "lr": language_model_lr,
            "weight_decay": 0.0,
            "name": "language_model_no_decay"
        })
    
    if other_params:
        param_groups.append({
            "params": other_params,
            "lr": default_lr,
            "weight_decay": weight_decay,
            "name": "other"
        })
    if other_params_no_decay:
        param_groups.append({
            "params": other_params_no_decay,
            "lr": default_lr,
            "weight_decay": 0.0,
            "name": "other_no_decay"
        })
    
    return param_groups


class MLLMTrainer(Trainer):
    def get_train_dataloader(self):
        dataloader = super().get_train_dataloader()

        if getattr(self.args, 'use_online_length_grouped_dataloader', False):
            from smallvlm.datasets.dataloaders_olg import apply_online_length_grouped_dataloader
            if 'DataLoaderAdapter' in iter(c.__name__ for c in type(dataloader).__mro__):
                dataloader_ = dataloader.base_dataloader
            else:
                dataloader_ = dataloader
            apply_online_length_grouped_dataloader(dataloader_)

        return dataloader

    def create_optimizer(self):
        """
        Create optimizer with different learning rates for different modules.
        Overrides the default Trainer.create_optimizer() method.
        """
        # Check if any module-specific learning rate is set
        has_custom_lr = any([
            getattr(self.args, 'vision_model_lr', None) is not None,
            getattr(self.args, 'vision_abstractor_lr', None) is not None,
            getattr(self.args, 'language_model_lr', None) is not None,
        ])
        
        if not has_custom_lr:
            # Use default optimizer creation if no custom learning rates
            return super().create_optimizer()
        
        if self.optimizer is not None:
            return self.optimizer
        
        # Get optimizer class and kwargs
        optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, self.model)
        
        # Remove 'lr' from optimizer_kwargs since we'll set it per group
        default_lr = optimizer_kwargs.pop('lr', self.args.learning_rate)
        weight_decay = optimizer_kwargs.get('weight_decay', self.args.weight_decay)
        
        # Create parameter groups with different learning rates
        param_groups = get_module_param_groups(
            self.model,
            self.args,
            default_lr=default_lr,
            weight_decay=weight_decay,
        )
        
        # Log the learning rates for each module
        if self.args.local_rank <= 0:
            logger.info("Creating optimizer with module-specific learning rates:")
            for group in param_groups:
                num_params = len(group['params'])
                total_params = sum(p.numel() for p in group['params'])
                logger.info(f"  {group['name']}: lr={group['lr']}, weight_decay={group['weight_decay']}, "
                          f"num_params={num_params}, total_elements={total_params:,}")
        
        # Remove weight_decay from optimizer_kwargs as it's already in param_groups
        optimizer_kwargs.pop('weight_decay', None)
        
        self.optimizer = optimizer_cls(param_groups, **optimizer_kwargs)
        
        return self.optimizer

    def _load_optimizer_and_scheduler(self, checkpoint):
        """If optimizer and scheduler states exist, load them."""
        """Fuck compatibility of transformers"""
        from transformers.integrations.deepspeed import is_deepspeed_available
        from transformers.trainer_pt_utils import reissue_pt_warnings
        if is_deepspeed_available():
            from accelerate.utils import DeepSpeedSchedulerWrapper

        SCHEDULER_NAME = "scheduler.pt"

        if checkpoint is None:
            return

        if self.is_deepspeed_enabled:
            # deepspeed loads optimizer/lr_scheduler together with the model in deepspeed_init
            if not isinstance(self.lr_scheduler, DeepSpeedSchedulerWrapper):
                with warnings.catch_warnings(record=True) as caught_warnings:
                    # check_torch_load_is_safe() # Fuck Here!!!
                    self.lr_scheduler.load_state_dict(
                        torch.load(os.path.join(
                            checkpoint, SCHEDULER_NAME), weights_only=True)
                    )
                reissue_pt_warnings(caught_warnings)
            return
        else:
            raise NotImplementedError