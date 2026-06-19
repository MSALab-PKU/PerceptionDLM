import base64
import copy
import io
import json
import math
import os
import random
import re
import bisect
import yaml
import numpy as np
import pycocotools.mask as mask_util
import torch
from datasets import load_from_disk
from PIL import Image
from torch.utils.data import Dataset
from transformers.image_utils import PILImageResampling
from typing import Optional, Union, List, Dict, Tuple, Any

from transformers import PretrainedConfig, PreTrainedModel, AutoModel, AutoModelForCausalLM, ProcessorMixin
from smallvlm.models.pdmllm.processing_pdmllm import find_closest_aspect_ratio, dynamic_preprocess

from smallvlm.utils.loggings import logger


def smart_resize(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = 56 * 56,
    max_pixels: int = 768 * 768,
):
    """Rescales the image so that the following conditions are met:
    1. Both dimensions are divisible by 'factor'.
    2. The total number of pixels is within ['min_pixels', 'max_pixels'].
    3. The aspect ratio is preserved as closely as possible.
    """
    if height < factor or width < factor:
        raise ValueError(
            f"height:{height} or width:{width} must be larger than factor:{factor}"
        )
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar

def build_mask_datasets(data_path: str,
                   processor: ProcessorMixin,
                   max_length: int,
                   augmentations: Optional[List[Union[str, dict]]] = None,
                   start_index: int = 0,
                   **kwargs) -> torch.utils.data.Dataset:

    if data_path.endswith('_datasets.yaml'):
        data_sources = _read_multi_datasets(data_path)
    else:
        data_sources = [{'data_file': data_path}]

    datasets = []
    num_samples = []
    for i, data_source in enumerate(data_sources):
        data_kwargs = kwargs.copy()
        data_kwargs.update(data_source)
        datafile = data_kwargs.pop('data_file')
        num_samples.append(data_kwargs.pop('num_samples', -1))
        
        data_type = data_kwargs.pop('data_type')
        if data_type == 'json':
            visual_prompt_nums = data_kwargs.pop('visual_prompt_nums')
            # visual_prompt_tokens = [f"<Prompt{i}>" for i in range(visual_prompt_nums)]
            # visual_prompt_tokens.append("<NO_Prompt>")
            # special_tokens = visual_prompt_tokens

            prompt_augmentation=data_kwargs.pop('prompt_augmentation', False)
            #dynamic_image_size=data_kwargs.pop('dynamic_image_size', False)
            repeats = data_kwargs.pop('repeats', 1)
            # max_num_tiles = data_kwargs.pop('max_num_tiles', 16)
            dataset = GraspAnyRegionDataset(pano_json=datafile,
                                             processor=processor,
                                             prompt_numbers=visual_prompt_nums,
                                             #special_tokens=special_tokens,
                                             #dynamic_image_size=dynamic_image_size,
                                             repeats=repeats,
                                             # max_num_tiles=max_num_tiles,
                                             prompt_augmentation=prompt_augmentation,
                                             **data_kwargs)
            datasets.append(dataset)
    
    dataset = SampleDatasets(datasets, num_samples, start_index=start_index)

    return dataset


def _read_multi_datasets(data_path):
    def _parse_dataset(data_source):
        if isinstance(data_source, dict):
            if 'data_file' in data_source:
                return [data_source]
            else:
                return sum((_parse_dataset(d) for d in data_source.values()), [])
        elif isinstance(data_source, list):
            return sum((_parse_dataset(d) for d in data_source), [])

    with open(data_path) as f:
        data_sources = yaml.load(f, Loader=yaml.FullLoader)

    data_sources = _parse_dataset(data_sources)

    return data_sources

class GraspAnyRegionDataset(Dataset):
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    def __init__(
        self,
        pano_json,
        processor,
        special_tokens=None,
        #dynamic_image_size=True,
        repeats=1,
        #max_num_tiles=16,
        prompt_augmentation=False,
        prompt_numbers=5,
        **kwargs,
    ):
        self._system = ""
        self.repeats = repeats
        #self.dynamic_image_size = dynamic_image_size
        #self.max_num_tiles = max_num_tiles if dynamic_image_size else 1
        self.prompt_augmentation = prompt_augmentation
        self.prompt_numbers = prompt_numbers
        # Epoch counter for deterministic mask partitioning (multi-worker safe)
        self._epoch = 0

        self.pano_json = pano_json

        self.processor = processor
        #image_processor_config = self.processor.image_processor.__dict__
        #image_processor_config.pop("_processor_class", None)

        # self.processor.image_processor = PerceptionLMImageProcessorFast.from_dict(
        #     image_processor_config
        # )
        #self.processor.image_processor.max_num_tiles = self.max_num_tiles

        self.processor_mask = processor
        # self.processor_mask.image_processor = PerceptionLMImageProcessorFast.from_dict(
        #     image_processor_config
        # )
        #self.processor_mask.image_processor.max_num_tiles = self.max_num_tiles
        self.processor_mask.image_processor.resample = PILImageResampling.NEAREST

        if special_tokens is not None:
            self.special_tokens = special_tokens
            self.processor.tokenizer.add_tokens(special_tokens, special_tokens=True)
            self.processor_mask.tokenizer.add_tokens(
                special_tokens, special_tokens=True
            )

        self.datas, self.data_lengths = self.read_pano_json()
        # precompute how many times to sample each image to cover its masks
        self.item_repeat_factors, self._cumulative_items = self._build_repeat_index(self.datas)

        # self.max_length = max_length
        self._max_refetch = 5

        start_text = "<|start_header_id|>assistant<|end_header_id|>\n"
        self.start_tokens = self.processor.tokenizer.encode(start_text, add_special_tokens=False)
        self.start_tokens = torch.tensor(self.start_tokens, dtype=torch.long)
        self.end_token = self.processor.tokenizer.encode("<|eot_id|>", add_special_tokens=False)[0]
        self.end_token = torch.tensor(self.end_token, dtype=torch.long)

    def set_epoch(self, epoch: int):
        """Set current epoch for deterministic mask partitioning."""
        self._epoch = epoch

    @property
    def modality_length(self):
        length_list = []
        for _ in range(sum(self.item_repeat_factors)):
            length_list.append(100)
        return length_list * self.repeats

    def __len__(self):
        return sum(self.item_repeat_factors) * self.repeats

    def read_pano_json(self):
        if self.pano_json.endswith(".json"):
            with open(self.pano_json, "r") as f:
                json_info = json.load(f)
        else:
            json_info = load_from_disk(self.pano_json)

        return json_info, len(json_info)

    def _count_masks(self, ann_info: Dict[str, Any]) -> int:
        if ann_info.get("mask_rles") is not None:
            if isinstance(ann_info["mask_rles"], list):
                return len(ann_info["mask_rles"])
            if isinstance(ann_info["mask_rles"], dict):
                return 1
        if ann_info.get("annotations") is not None:
            return len(ann_info["annotations"])
        return 1

    def _build_repeat_index(self, data_list: List[Dict[str, Any]]):
        repeat_factors = []
        cumulative = []
        total = 0
        for ann in data_list:
            num_masks = self._count_masks(ann)
            # expect 1-self.prompt_numbers masks per sample; use ceil(num_masks/((1+self.prompt_numbers)/2)) to guarantee coverage even with min group size 1
            repeats = max(1, math.ceil(num_masks / ((1 + self.prompt_numbers) / 2)))
            repeat_factors.append(repeats)
            total += repeats
            cumulative.append(total)
        return repeat_factors, cumulative

    def sort_masks_by_area(self, masks):
        areas = []
        for mask in masks:
            area = np.sum(mask)
            areas.append(area)
        indexes = np.argsort(np.array(areas))[
            ::-1
        ]  # sort the mask from large area to small area
        return indexes

    def _select_mask_indices(self, data_idx: int, internal_index: int, num_masks: int) -> List[int]:
        """Return a deterministic batch of mask indices for the given (data_idx, internal_index).

        Uses a seeded RNG to shuffle all mask indices and partition them into
        groups, ensuring full coverage across all internal_indices.  The result
        is purely a function of (data_idx, internal_index, epoch) so it is
        safe with any number of DataLoader workers or DDP ranks.
        """
        num_repeats = self.item_repeat_factors[data_idx]

        # Deterministic shuffle seeded by (data_idx, epoch)
        rng = random.Random(data_idx * 100003 + self._epoch * 999983)
        all_indices = list(range(num_masks))
        rng.shuffle(all_indices)

        # Partition into num_repeats groups as evenly as possible
        groups: List[List[int]] = []
        base_size = num_masks // num_repeats
        remainder = num_masks % num_repeats
        start = 0
        for i in range(num_repeats):
            size = base_size + (1 if i < remainder else 0)
            groups.append(all_indices[start:start + size])
            start += size

        # Defensive clamp (should not happen given _build_repeat_index logic)
        if internal_index >= num_repeats:
            logger.warning(
                f"internal_index ({internal_index}) >= num_repeats ({num_repeats}) "
                f"for data_idx={data_idx}, num_masks={num_masks}. Clamping."
            )
            internal_index = internal_index % num_repeats

        selected = groups[internal_index]
        # Ensure at least one mask is returned
        return selected if selected else [all_indices[0]]

    def _compute_aspect_ratio(self, width: int, height: int, num_tiles: int) -> torch.Tensor:
        """Infer tiling grid (ncw, nch) to align with vision tiles."""
        min_tiles = getattr(self.processor, "min_sub_img", 1)
        max_tiles = getattr(self.processor, "max_sub_img", 6)
        image_size = getattr(self.processor, "image_size", (512, 512))
        image_size = image_size[0] if isinstance(image_size, tuple) else image_size

        aspect_ratio = width / height
        target_ratios = {
            (i, j)
            for n in range(min_tiles, max_tiles + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if min_tiles <= i * j <= max_tiles
        }
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        grid_w, grid_h = find_closest_aspect_ratio(
            aspect_ratio, target_ratios, width, height, image_size
        )

        tiles_without_thumb = grid_w * grid_h
        expected_tiles = tiles_without_thumb if tiles_without_thumb == 1 else tiles_without_thumb + 1
        if num_tiles not in (tiles_without_thumb, expected_tiles):
            logger.warning(
                f"Unexpected tile count ({num_tiles}) for grid {grid_w}x{grid_h}; downstream merge may fail."
            )

        return torch.tensor([grid_w, grid_h], dtype=torch.int)
    
    def sort_masks_by_area(self, masks):
        areas = []
        for mask in masks:
            area = np.sum(mask)
            areas.append(area)
        indexes = np.argsort(np.array(areas))[::-1]  # sort the mask from large area to small area
        return indexes

    def _parse_annotations(self, ann_info, data_idx, internal_index):
        # unify schema: support keys image_name/image_file or legacy image
        image_path = ann_info.get("image_file") or ann_info.get("image") or ann_info.get("image_path")
        image_key = (
            ann_info.get("image_name")
            or ann_info.get("image_id")
            or (image_path if isinstance(image_path, str) else ann_info.get("image_key", f"mem_{id(image_path)}"))
        )

        if image_path is not None:
            if isinstance(image_path, Image.Image):
                image = image_path
            elif isinstance(image_path, str) and image_path.startswith("data:base64,"):
                base64_str = image_path.replace("data:base64,", "")
                image_bytes = base64.b64decode(base64_str)
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            else:
                image = Image.open(image_path).convert("RGB")

            if ann_info.get("mask_rles") is not None:
                mask_caption_data = True
                if isinstance(ann_info["mask_rles"], list):
                    masks = [mask_util.decode(rle_dict) for rle_dict in ann_info["mask_rles"]]
                elif isinstance(ann_info["mask_rles"], dict):
                    masks = [mask_util.decode(ann_info["mask_rles"])]
                else:
                    raise ValueError(
                        f"mask_rles should be list or dict, but got {type(ann_info['mask_rles'])}"
                    )
                captions = ann_info.get("captions", [""] * len(masks))
            elif ann_info.get("annotations") is not None:
                mask_caption_data = True
                masks, captions = [], []
                for ann in ann_info["annotations"]:
                    captions.append(ann.get("caption", ""))
                    seg = ann.get("segmentation")
                    if seg is None:
                        masks.append(np.ones((image.height, image.width), dtype=np.uint8))
                        continue
                    masks.append(mask_util.decode(seg))
            else:
                mask_caption_data = False
                masks = [np.ones((image.height, image.width), dtype=np.uint8)]
                captions = ann_info.get("captions", [""])
        else:
            print("no image, skip.")
            return None

        # pick a subset of masks (1-self.prompt_numbers) from the same image, deterministically partitioned
        selected_indices = self._select_mask_indices(data_idx, internal_index, len(masks))
        selected_masks_np = [masks[i].astype(np.uint8) for i in selected_indices]
        selected_captions = [captions[i] for i in selected_indices]
        num_selected = len(selected_masks_np)

        selected_prompt_img_tokens = [
            f"<|reserved_token_{i}|>" for i in range(num_selected)
        ]

        # ensure mask size matches image and rebuild mask list as numpy arrays
        for mask_id, mask_np in enumerate(selected_masks_np):
            if image.width != mask_np.shape[1] or image.height != mask_np.shape[0]:
                pil_mask = Image.fromarray(mask_np)
                pil_mask = pil_mask.resize(image.size, Image.NEAREST)
                selected_masks_np[mask_id] = np.array(pil_mask).astype(np.uint8)

        bboxes = {}
        for mask_id, mask_np in enumerate(selected_masks_np):
            non_zero_coords = np.argwhere(mask_np)
            if non_zero_coords.size == 0:
                y_min, x_min = 0, 0
                y_max, x_max = 0, 0
            else:
                y_min, x_min = non_zero_coords.min(axis=0)
                y_max, x_max = non_zero_coords.max(axis=0)
            bbox = (
                x_min / image.width,
                y_min / image.height,
                x_max / image.width,
                y_max / image.height,
            )
            bboxes[
                str(
                    self.processor.tokenizer.convert_tokens_to_ids(
                        f"<|reserved_token_{mask_id}|>"
                    )
                )
            ] = bbox

        if not mask_caption_data:
            filled_matrix = np.ones(
               (image.height, image.width), dtype=np.uint8
            )
            ret = {
                "masks": [Image.fromarray((m * 255).astype(np.uint8)) for m in selected_masks_np],
                "bboxes": bboxes,
                "conversations": ann_info["conversations"],
                "image": image,
                "visual_prompt_matrix": Image.fromarray(filled_matrix),
                "mask_caption_data": False,
            }
            return ret

        # sort the mask according to the area (descending) BEFORE building
        # conversation / bboxes so that every downstream structure is consistent.
        indexes = self.sort_masks_by_area(selected_masks_np)
        selected_masks_np = [selected_masks_np[idx] for idx in indexes]
        selected_captions = [selected_captions[idx] for idx in indexes]

        # rebuild bboxes in sorted order
        bboxes = {}
        for mask_id, mask_np in enumerate(selected_masks_np):
            non_zero_coords = np.argwhere(mask_np)
            if non_zero_coords.size == 0:
                y_min, x_min = 0, 0
                y_max, x_max = 0, 0
            else:
                y_min, x_min = non_zero_coords.min(axis=0)
                y_max, x_max = non_zero_coords.max(axis=0)
            bbox = (
                x_min / image.width,
                y_min / image.height,
                x_max / image.width,
                y_max / image.height,
            )
            bboxes[
                str(
                    self.processor.tokenizer.convert_tokens_to_ids(
                        f"<|reserved_token_{mask_id}|>"
                    )
                )
            ] = bbox

        # build single-turn conversation covering all masks (now in sorted order)
        user_prompt_lines = selected_prompt_img_tokens + ["Describe each masked region in detail."]
        user_prompt = "\n".join(user_prompt_lines)
        assistant_replies = [f"<|Mask_Cap_{i}|>{selected_captions[i]}" for i in range(num_selected)]

        conversation = [
            {"from": "human", "value": f"<image>\n{user_prompt}"},
            {"from": "gpt", "value": "\n".join(assistant_replies)},
        ]

        # assign random prompt indexes and sort them consistently
        prompt_indexes = list(range(self.prompt_numbers))
        random.shuffle(prompt_indexes)
        selected_prompt_indexes = prompt_indexes[:len(selected_masks_np)]
        selected_prompt_tokens = [f"<Prompt{i}>" for i in selected_prompt_indexes]

        # build filled matrices: masks are already sorted, so zip directly
        filled_matrices = []
        for prompt_id, mask_np in zip(selected_prompt_indexes, selected_masks_np):
            filled_matrix = np.full((image.height, image.width), 255, dtype=np.uint8)

            assert prompt_id < self.prompt_numbers, (
                f"prompt_id should be less than {self.prompt_numbers}, got {prompt_id}"
            )
            fill_area = (filled_matrix == 255) & mask_np.astype(bool)
            filled_matrix[fill_area] = prompt_id
            filled_matrices.append(filled_matrix)

        masks = [
            Image.fromarray((selected_masks_np[i] * 255).astype(np.uint8))
            for i in range(num_selected)
        ]

        ret = {
            "masks": masks,
            "bboxes": bboxes,
            "conversations": conversation,
            "image": image,
            "visual_prompt_matrices": [Image.fromarray(m) for m in filled_matrices],
            "mask_caption_data": mask_caption_data,
            "prompt_tokens": selected_prompt_tokens
        }
        return ret

    def parse_label(self, labels):
        labels = labels.clone()
        mask = torch.full_like(labels, fill_value=-100)

        i = 0
        while i < len(labels):
            if i + len(self.start_tokens) <= len(labels) and torch.equal(
                labels[i : i + len(self.start_tokens)], self.start_tokens
            ):
                start = i + len(self.start_tokens)
                try:
                    end = (labels[start:] == self.end_token).nonzero(as_tuple=True)[0][
                        0
                    ].item() + start
                except IndexError:
                    break
                # keep [start:end+1]
                if end >= start:
                    mask[start : end + 1] = labels[start : end + 1]
                i = end + 1
            else:
                i += 1

        return mask

    def prepare_data(self, index, **kwargs):
        base_cycle = sum(self.item_repeat_factors)
        index = index % base_cycle

        def find_dataset_index(index, cumulative_sizes):
            pos = bisect.bisect_right(cumulative_sizes, index)
            data_idx = max(0, pos)
            internal_index = index if data_idx == 0 else index - cumulative_sizes[data_idx - 1]
            return data_idx, internal_index

        data_idx, internal_index = find_dataset_index(index, self._cumulative_items)

        data_dict = copy.deepcopy(self.datas[data_idx])
        data_dict = self._parse_annotations(data_dict, data_idx, internal_index)

        if data_dict is None:
            return None

        image = data_dict["image"]
        convs = data_dict["conversations"]
        visual_prompts = data_dict["visual_prompt_matrices"]
        prompt_tokens = data_dict["prompt_tokens"]

        w, h = image.size
        if w < 10 or h < 10:
            return None

        if data_dict["mask_caption_data"]:
            messages, messages_mask = [], []
            for i, conv in enumerate(convs):
                if i == 0:
                    assert conv["from"] == "human"
                    messages.append(
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image},
                                {
                                    "type": "text",
                                    "text": conv["value"].replace("<image>\n", ""),
                                },
                            ],
                        },
                    )
                    # messages_mask.append(
                    #     {
                    #         "role": "user",
                    #         "content": [
                    #             {"type": "image", "image": visual_prompt},
                    #             {
                    #                 "type": "text",
                    #                 "text": conv["value"].replace("<image>\n", ""),
                    #             },
                    #         ],
                    #     },
                    # )
                    continue

                assert "<image>" not in conv["value"]
                if conv["from"] == "human":
                    messages.append(
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": conv["value"]}],
                        }
                    )
                    # messages_mask.append(
                    #     {
                    #         "role": "user",
                    #         "content": [{"type": "text", "text": conv["value"]}],
                    #     }
                    # )
                elif conv["from"] == "gpt":
                    messages.append(
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": conv["value"]}],
                        }
                    )
                    # messages_mask.append(
                    #     {
                    #         "role": "assistant",
                    #         "content": [{"type": "text", "text": conv["value"]}],
                    #     }
                    # )
                else:
                    raise NotImplementedError
        else:
            # keep the same with the original provided conversation
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {
                            "type": "text",
                            "text": data_dict["conversations"][0]["value"].replace(
                                "<image>\n", ""
                            ),
                        },
                    ],
                },
            ]
            messages_mask = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": visual_prompt},
                        {
                            "type": "text",
                            "text": data_dict["conversations"][0]["value"].replace(
                                "<image>\n", ""
                            ),
                        },
                    ],
                },
            ]
            for conv in data_dict["conversations"][1:]:
                assert "<image>" not in conv["value"]
                if conv["from"] == "human":
                    messages.append(
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": conv["value"]}],
                        }
                    )
                    messages_mask.append(
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": conv["value"]}],
                        }
                    )
                elif conv["from"] == "gpt":
                    messages.append(
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": conv["value"]}],
                        }
                    )
                    messages_mask.append(
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": conv["value"]}],
                        }
                    )

        mask_values_list = []
        try:
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=False,
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
            )

            # inputs_mask = self.processor.apply_chat_template(
            #     messages_mask,
            #     add_generation_prompt=False,
            #     tokenize=True,
            #     return_tensors="pt",
            #     return_dict=True,
            # )
            for visual_prompt in visual_prompts:
                visual_prompt_rgb = visual_prompt.convert("RGB")
                sub_masks = dynamic_preprocess(
                    visual_prompt_rgb,
                    min_num=self.processor.min_sub_img,
                    max_num=self.processor.max_sub_img,
                    image_size=self.processor.image_size[0],
                    use_thumbnail=True,
                )
                mask_values = self.processor.image_processor.preprocess(images=sub_masks, return_tensors="pt")["pixel_values"]
                mask_values_list.append(mask_values)

        except:
            print("tokenization failed.")
            return None
        
        

        pixel_values = inputs["pixel_values"]
        num_tiles = pixel_values.shape[0]
        aspect_ratio = self._compute_aspect_ratio(w, h, num_tiles)
        #mask_values = inputs_mask["pixel_values"]
        input_ids = inputs["input_ids"].squeeze(0)
        # try:
        #     assert torch.equal(inputs["input_ids"], inputs_mask["input_ids"])
        #     assert torch.equal(inputs["attention_mask"], inputs_mask["attention_mask"])
        # except:
        #     print("inputs are different, skip")
        #     return None

        labels = inputs["input_ids"].squeeze(0).clone()
        labels = self.parse_label(labels)
        attention_mask = inputs["attention_mask"].squeeze(0)

        ret = dict(
            input_ids=input_ids.unsqueeze(0),
            labels=labels.unsqueeze(0),
            attention_mask=attention_mask.unsqueeze(0),
            pixel_values=pixel_values,
            global_mask_values_list=mask_values_list,
            aspect_ratios=aspect_ratio.unsqueeze(0),
            bboxes=data_dict["bboxes"],
            prompt_tokens=prompt_tokens,
        )
        return ret

    def _rand_another(self):
        idx = random.randint(0, max(0, len(self) - 1))
        return idx

    def __getitem__(self, index):
        for _ in range(self._max_refetch + 1):
            try:
                data = self.prepare_data(index, padding=False, return_tensors="pt")
            except:
                data = None

            if data is None:
                index_old = index
                index = self._rand_another()
                print(f"[WARNING] data {index_old} is None, use {index}!")
                continue
            return data

class SampleDatasets(torch.utils.data.Dataset):
    def __init__(self, datasets: List[GraspAnyRegionDataset], num_samples: List[int], start_index: int = 0):
        super().__init__()
        self.datasets = datasets
        assert len(self.datasets) > 0, 'datasets should not be an empty iterable'
        for d in self.datasets:
            assert not isinstance(d, torch.utils.data.IterableDataset), "ConcatDataset does not support IterableDataset"
        self.num_samples = num_samples
        self.cumulative_sizes = []
        self._samples = {}
        self._nth_epoch = -1
        
        self.start_index = start_index 
        
        self.set_epoch(0)

        total_len = self.cumulative_sizes[-1]

        if self.start_index >= total_len:
            logger.warning(f"Start index ({self.start_index}) is larger than dataset length ({total_len}). Dataset will be empty.")
        
        self._max_refetch = 5

    def __len__(self):
        return max(0, self.cumulative_sizes[-1] - self.start_index)

    def _rand_another(self):
        idx = random.randint(0, len(self) - 1)
        return idx

    def _get_data(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
            
        real_idx = idx + self.start_index
        
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, real_idx)

        sample_idx = real_idx if dataset_idx == 0 else real_idx - self.cumulative_sizes[dataset_idx - 1]
        if dataset_idx in self._samples:
            sample_idx = self._samples[dataset_idx][sample_idx]

        return self.datasets[dataset_idx][sample_idx]

    def __getitem__(self, index):
        for _ in range(self._max_refetch + 1):
            try:
                data = self._get_data(index)
            except Exception as e:
                logger.warning(f"Data error at index {index}: {e}")
                data = None

            if data is None:
                index_old = index
                index = self._rand_another()
                logger.warning(f"[WARNING] data {index_old} is None, use {index}!")
                continue
            return data
        raise RuntimeError(f"Failed to fetch data after {self._max_refetch} retries.")

    def set_epoch(self, nth_epoch):
        if nth_epoch != self._nth_epoch:
            self._sample_indices(nth_epoch)
            # Propagate epoch to child datasets for deterministic mask partitioning
            for ds in self.datasets:
                if hasattr(ds, 'set_epoch'):
                    ds.set_epoch(nth_epoch)
        self._nth_epoch = nth_epoch

    def _sample_indices(self, nth_epoch):
        logger.info(f'resample datasets in epoch_{nth_epoch}')
        random.seed(281 + nth_epoch)
        g = torch.Generator()
        g.manual_seed(281 + nth_epoch)
        cumulative_sizes = [0]
        for i, (dataset, num_samples) in enumerate(zip(self.datasets, self.num_samples)):
            if num_samples >= 0:
                indices = torch.randperm(len(dataset), generator=g)[:num_samples % len(dataset) if num_samples % len(dataset) != 0 else len(dataset)].tolist()
                if num_samples > len(dataset):
                    logger.warning(f"num_samples ({num_samples}) exceed dataset size ({len(dataset)}), "
                                   f"duplicating the dataset.")
                    indices = list(range(len(dataset))) * (num_samples // len(dataset)) + indices
                self._samples[i] = indices
                cumulative_sizes.append(cumulative_sizes[-1] + num_samples)
            else:
                cumulative_sizes.append(cumulative_sizes[-1] + len(dataset))
        self.cumulative_sizes = cumulative_sizes[1:]