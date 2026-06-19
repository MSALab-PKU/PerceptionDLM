from typing import Optional, Sequence
from collections import OrderedDict
import glob
import re
import gc
import torch
from .loggings import logger

BASIC_TYPES = (int, float, str, bool)


def to_device(entry, device: Optional[torch.device] = None, dtype: torch.dtype = None):
    if isinstance(entry, torch.Tensor):
        if not entry.is_floating_point() or entry.dtype == dtype:
            dtype = None
        if device is None and dtype is None:
            return entry
        return entry.to(device=device, dtype=dtype)
    elif isinstance(entry, Sequence):
        if all(isinstance(x, BASIC_TYPES) for x in entry):
            return entry
        return type(entry)(x if isinstance(x, BASIC_TYPES) else to_device(x, device, dtype) for x in entry)
    elif isinstance(entry, dict):
        return {k: to_device(v, device, dtype) for k, v in entry.items()}
    else:
        return entry


def load_state_dict_file(state_dict_path, model=None, prefix='default'):

    from safetensors.torch import load_file
    import os
    import glob
    
    if isinstance(state_dict_path, str):
        if os.path.isdir(state_dict_path):
            state_dict_path = glob.glob(os.path.join(state_dict_path, "*.safetensors"))
        elif '*' in state_dict_path:
            state_dict_path = glob.glob(state_dict_path)

    state_dict = {}
    if isinstance(state_dict_path, list):
        for p in state_dict_path:
            logger.info(f'Loading state_dict from {p}')
            state_dict_ = load_file(p)
            state_dict.update(state_dict_)
    elif isinstance(state_dict_path, str) and state_dict_path.endswith('.safetensors'):
        logger.info(f'Loading state_dict from {state_dict_path}')
        state_dict = load_file(state_dict_path)

    # Auto-resize token embeddings if checkpoint has a different vocab size
    if model is not None:
        for key in state_dict:
            if 'embed_tokens.weight' in key:
                ckpt_vocab_size = state_dict[key].shape[0]
                # Walk the model attributes to find the corresponding parameter
                try:
                    obj = model
                    for part in key.split('.'):
                        obj = getattr(obj, part)
                    cur_vocab_size = obj.shape[0]
                except Exception:
                    cur_vocab_size = ckpt_vocab_size  # skip resize on lookup failure

                if cur_vocab_size != ckpt_vocab_size:
                    logger.warning(
                        f"Resizing token embeddings from {cur_vocab_size} to "
                        f"{ckpt_vocab_size} to match checkpoint."
                    )
                    if hasattr(model, 'language_model') and hasattr(model.language_model, 'resize_token_embeddings'):
                        model.language_model.resize_token_embeddings(ckpt_vocab_size)
                    elif hasattr(model, 'resize_token_embeddings'):
                        model.resize_token_embeddings(ckpt_vocab_size)
                break

    missing_keys_, unexpected_keys_ = model.load_state_dict(state_dict, strict=False)
    return missing_keys_, unexpected_keys_