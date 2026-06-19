from collections import UserDict
import torch
from torch.nn import functional as F


class Collator:
    def __init__(self, pad_token_id, pad_side='right'):
        self.pad_token_id = pad_token_id
        assert pad_side in ('right', 'left')
        self.pad_side = pad_side

    def __call__(self, batch):
        input_ids = padcat_sequences([inputs.pop('input_ids') for inputs in batch],
                                     value=self.pad_token_id, pad_side=self.pad_side)
        attention_mask = padcat_attention_mask([inputs.pop('attention_mask') for inputs in batch],
                                               pad_side=self.pad_side)
        position_ids = padcat_sequences([inputs.pop('position_ids', None) for inputs in batch],
                                        value=0, pad_side=self.pad_side)
        inputs_batch = {'input_ids': input_ids,
                        'attention_mask': attention_mask,
                        'position_ids': position_ids}
       
        if 'labels' in batch[0]:
            inputs_batch['labels'] = padcat_sequences([inputs.pop('labels') for inputs in batch],
                                                    value=-100, pad_side=self.pad_side)
        
        if 'record' in batch[0]:
            inputs_batch['record'] = [inputs.pop('record') for inputs in batch]

        if 'bboxes' in batch[0]:
            inputs_batch['bboxes'] = [inputs.pop('bboxes') for inputs in batch]

        if 'prompt_tokens' in batch[0]:
            inputs_batch['prompt_tokens'] = sum([inputs.pop('prompt_tokens') for inputs in batch], [])

        others = collate_items(batch)
        inputs_batch.update(others)

        return inputs_batch


def collate_items(batch):
    if not isinstance(batch, list):
        return batch

    item_type = type(batch[0])
    if not all(type(it) is item_type for it in batch[1:]):
        return batch

    if item_type is torch.Tensor:
        return torch.cat(batch)
    elif item_type is list:
        if all(len(it) == 0 for it in batch):
            return None
        if all(len(it) == 0 or all(isinstance(x, torch.Tensor) for x in it) for it in batch):
            batch = torch.stack(sum(batch, []))
        return batch
    elif item_type is tuple:
        return tuple(collate_items(list(b)) for b in zip(*batch))
    elif item_type is dict or issubclass(item_type, UserDict):
        keys = set.intersection(*(set(it.keys()) for it in batch))
        if keys != set.union(*(set(it.keys()) for it in batch)):
            return batch
        return {k: collate_items([it[k] for it in batch]) for k in keys}
    elif item_type is type(None):
        return None
    else:
        return batch


def padcat_attention_mask(attention_masks, pad_side='right'):
    if all(s is None for s in attention_masks):
        return None

    ndim = attention_masks[0].ndim
    if ndim == 2 and all(x.ndim == ndim for x in attention_masks):
        return padcat_sequences(attention_masks, value=False, pad_side=pad_side)

    attention_masks_4d = []
    for x in attention_masks:
        if x.ndim == 4:
            attention_masks_4d.append(x)
        elif x.ndim == 2:
            x_ = torch.ones((1, 1, x.size(1), x.size(1)), dtype=torch.bool)
            x_.tril_()
            x_[0, x == False] = False
            attention_masks_4d.append(x_)
        else:
            raise ValueError(f"Wrong attention_mask dim: {x.ndim}")

    max_l = max(x.size(2) for x in attention_masks_4d)
    attention_masks_ = []
    for x in attention_masks_4d:
        if x.size(2) != max_l:
            pad_len = max_l - x.size(2)
            pad_len = (0, pad_len, 0, pad_len) if pad_side == 'right' else (pad_len, 0, pad_len, 0)
            x = F.pad(x, pad_len, value=False)
        attention_masks_.append(x)

    attention_masks = torch.cat(attention_masks_)

    return attention_masks


def padcat_sequences(sequences, value=0, pad_side='right'):
    if all(s is None for s in sequences):
        return None
    max_l = max(s.size(1) for s in sequences)
    sequences_ = []
    for seq in sequences:
        if seq.size(1) != max_l:
            pad_len = max_l - seq.size(1)
            pad_len = (0, pad_len) if pad_side == 'right' else (pad_len, 0)
            seq = F.pad(seq, pad_len, value=value)
        sequences_.append(seq)

    sequences = torch.cat(sequences_)

    return sequences


def padcat_images(images, value=0):
    sizes = [m.shape[2:] for m in images]
    max_h, max_w = max(sz[0] for sz in sizes), max(sz[1] for sz in sizes)
    images_ = []
    for image, sz in zip(images, sizes):
        if sz[0] != max_h or sz[1] != max_w:
            image = F.pad(image, (0, max_w-sz[1], 0, max_h-sz[0]), value=value)
        images_.append(image)

    images = torch.cat(images_)

    return images
