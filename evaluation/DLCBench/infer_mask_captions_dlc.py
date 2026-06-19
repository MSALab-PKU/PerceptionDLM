import argparse
import os
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from pycocotools.coco import COCO
from typing import Dict, List
import sys
import yaml

sys.path.append("../../")
from smallvlm.models.pdmllm.modeling_pdmllm import PDMLLM
from smallvlm.models.pdmllm.processing_pdmllm import find_closest_aspect_ratio, PDMLLMProcessor, dynamic_preprocess
from smallvlm.models.build_processors import build_processor

def sort_masks_by_area(masks: List[np.ndarray]):
    """Return indices that sort masks from large area to small area."""
    areas = [np.sum(m) for m in masks]
    return np.argsort(np.array(areas))[::-1]

def build_visual_prompt_matrices(masks: List[np.ndarray], prompt_numbers: int) -> tuple:
    """Build per-mask visual prompt matrices matching the dataset implementation."""
    if len(masks) > prompt_numbers:
        raise ValueError(f"Number of masks ({len(masks)}) exceeds prompt_numbers ({prompt_numbers}).")
    height, width = masks[0].shape

    prompt_indexes = list(range(prompt_numbers))
    selected_prompt_indexes = prompt_indexes[: len(masks)]
    selected_prompt_tokens = [f"<Prompt{i}>" for i in selected_prompt_indexes]

    filled_matrices = []
    for prompt_id, mask in zip(selected_prompt_indexes, masks):
        filled_matrix = np.full((height, width), 255, dtype=np.uint8)
        fill_area = (filled_matrix == 255) & mask.astype(bool)
        filled_matrix[fill_area] = prompt_id
        filled_matrices.append(filled_matrix)

    visual_prompt_images = [Image.fromarray(m) for m in filled_matrices]
    return visual_prompt_images, selected_prompt_tokens, selected_prompt_indexes

def build_bboxes(masks: List[np.ndarray], tokenizer) -> Dict[str, tuple[float, float, float, float]]:
    height, width = masks[0].shape
    bboxes: Dict[str, tuple[float, float, float, float]] = {}
    for idx, mask in enumerate(masks):
        coords = np.argwhere(mask > 0)
        if coords.size == 0:
            continue
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        token_id = tokenizer.convert_tokens_to_ids(f"<|reserved_token_{idx}|>")
        bboxes[str(token_id)] = (
            x_min / width,
            y_min / height,
            x_max / width,
            y_max / height,
        )
    return bboxes

def compute_aspect_ratio(image: Image.Image, processor, num_tiles: int) -> torch.Tensor:
    min_tiles = getattr(processor, "min_sub_img", 1)
    max_tiles = getattr(processor, "max_sub_img", 6)
    if hasattr(processor, "image_size"):
        image_size = processor.image_size[0] if isinstance(processor.image_size, tuple) else processor.image_size
    else:
        size = getattr(processor, "size", 512)
        if isinstance(size, dict):
            image_size = size.get("height", size.get("shortest_edge", 512))
        else:
            image_size = size
    aspect_ratio = image.width / image.height
    target_ratios = {
        (i, j)
        for n in range(min_tiles, max_tiles + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_tiles <= i * j <= max_tiles
    }
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    grid_w, grid_h = find_closest_aspect_ratio(aspect_ratio, target_ratios, image.width, image.height, image_size)
    print(f"aspect_ratio: {aspect_ratio}, grid_w: {grid_w}, grid_h: {grid_h}, image_size: {image_size}")
    return torch.tensor([[grid_w, grid_h]], dtype=torch.int64)

def build_prompt_text(tokenizer, num_image_token: int, num_tiles: int, questions: List[str], gen_len: int, num_masks: int) -> str:
    img_ctx = "".join(["<IMG_CONTEXT>"] * (num_image_token * num_tiles))
    parts = ["<|start_header_id|>system<|end_header_id|>\nYou are a helpful assistant.<|eot_id|>\n"]
    parts.append("<|start_header_id|>user<|end_header_id|>\n")
    parts.append(f"<img>{img_ctx}</img>" + "\n".join([f"<|reserved_token_{i}|>" for i in range(num_masks)]) + f"\n{questions[0]}<|eot_id|>\n")
    parts.append("<|start_header_id|>assistant<|end_header_id|>\n")
    mask_seq = "<|mdm_mask|>" * gen_len
    #parts.append("\n".join([f"<|Mask_Cap_{i}|>{mask_seq}" for i in range(num_masks)]))
    # specific for pfdmllm
    parts.append("\n".join([f"<Prompt{i}>:{mask_seq}" for i in range(num_masks)]))
    return "".join(parts) + "<|eot_id|>"

def split_assistant_blocks(text: str) -> str:
    blocks = text.split("<|start_header_id|>assistant<|end_header_id|>\n")
    return blocks[-1] if blocks else ""

cache_base = "dlc_model_outputs_cache"

def cache_model_outputs(cache_name, cache_values, overwrite=False):
    os.makedirs(cache_base, exist_ok=True)
    with open(os.path.join(cache_base, cache_name + ".json"), 'w' if overwrite else 'x') as f:
        json.dump(cache_values, f, indent=4)

def parse_key(k):
    try:
        return int(k)
    except ValueError:
        return k

def load_cached_model_outputs(cache_name):
    cache_path = os.path.join(cache_base, cache_name + ".json")
    if not os.path.exists(cache_path):
        return {}
    print("Loading cache from", cache_path)
    with open(cache_path, 'r') as f:
        model_outputs = json.load(f)
    model_outputs = {parse_key(k): v for k, v in model_outputs.items()}
    return model_outputs

def select_ann(coco, img_id, area_min=None, area_max=None):
    cat_ids = coco.getCatIds()
    ann_ids = coco.getAnnIds(imgIds=[img_id], catIds=cat_ids, iscrowd=None)
    if area_min is not None:
        ann_ids = [ann_id for ann_id in ann_ids if coco.anns[ann_id]['area'] >= area_min]
    if area_max is not None:
        ann_ids = [ann_id for ann_id in ann_ids if coco.anns[ann_id]['area'] <= area_max]
    return ann_ids

def load_processor_from_config(config_path: str):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    if "PROCESSOR_CONFIG" not in cfg:
        raise ValueError(f"PROCESSOR_CONFIG not found in {config_path}")
    return build_processor(cfg["PROCESSOR_CONFIG"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate PDMLLM on DLC-Bench with Multi-Mask Generator')
    
    parser.add_argument("--model-path", required=True, help="Path to PDMLLM checkpoint.")
    parser.add_argument("--gen-length", type=int, default=128)
    parser.add_argument("--steps", type=int, default=128, help="Number of generation steps.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--prompt", default="Describe each masked region in detail.", help="Prompt instruction.")

    parser.add_argument("--data-root", type=str, help="Data root", default="./")
    parser.add_argument("--suffix", type=str, help="Suffix to the saved json", default="")
    parser.add_argument("--cache_base", default=None, type=str, help="Override the cache base")
    parser.add_argument("--cache-name-override", type=str, help="Override the cache name", default=None)

    args = parser.parse_args()

    if args.cache_base is not None:
        cache_base = args.cache_base

    print(f"Loading model from {args.model_path}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    processor = PDMLLMProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer = processor.tokenizer
    
    model = PDMLLM.from_pretrained(args.model_path, torch_dtype=dtype, trust_remote_code=True)
    model.processor = processor
    model.to(device)
    model.eval()
    print("Model loaded.")

    ann_file = os.path.join(args.data_root, 'annotations.json')
    print(f"Loading annotations from {ann_file}...")
    coco = COCO(ann_file)
    img_ids = list(coco.imgs.keys())
    num_anns = len(coco.anns)

    cache_name = args.cache_name_override if args.cache_name_override is not None else "pdmllm_multi_bench_output" + args.suffix
    model_outputs = load_cached_model_outputs(cache_name)
    print(f"Results will be cached to {os.path.join(cache_base, cache_name + '.json')}")

    pbar = tqdm(total=num_anns)
    
    # Pre-calculate already processed to update pbar correctly
    pre_processed = len([ann for img_id in img_ids for ann in select_ann(coco, img_id) if ann in model_outputs])
    pbar.update(pre_processed)

    for img_id in img_ids:
        ann_ids = select_ann(coco, img_id)
        if not ann_ids:
            continue
            
        remaining_ann_ids = [ann_id for ann_id in ann_ids if ann_id not in model_outputs]
        if not remaining_ann_ids:
            continue

        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(args.data_root, "images", img_info['file_name'])
        try:
            pil_image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Skip if image fails to load, maybe update pbar?
            pbar.update(len(remaining_ann_ids))
            continue

        sub_images = dynamic_preprocess(
            pil_image,
            min_num=processor.min_sub_img,
            max_num=processor.max_sub_img,
            image_size=processor.image_size[0],
            use_thumbnail=True,
        )
        pixel_values = processor.image_processor.preprocess(images=sub_images, return_tensors="pt")["pixel_values"].to(device).to(dtype)
        aspect_ratio = compute_aspect_ratio(pil_image, processor, num_tiles=pixel_values.shape[0]).to(device)
        print(f"Image ID: {img_id} / aspect_ratio: {aspect_ratio} / pixel_values: {pixel_values.shape}")

        # MAX_MASKS = getattr(model.config, 'prompt_numbers', 10)  # Default fallback if not found
        MAX_MASKS =1 # DLC-Bench eval

        for i in range(0, len(remaining_ann_ids), MAX_MASKS):
            batch_ann_ids = remaining_ann_ids[i:i + MAX_MASKS]
            
            masks_list = []
            for ann_id in batch_ann_ids:
                anns = coco.loadAnns([ann_id])
                mask_np = coco.annToMask(anns[0]).astype(np.uint8)
                masks_list.append(mask_np)

            # Sort masks by area descending
            sort_idx = sort_masks_by_area(masks_list)
            sorted_masks_list = [masks_list[idx] for idx in sort_idx]
            
            sorted_batch_ann_ids = [batch_ann_ids[idx] for idx in sort_idx]

            bboxes = build_bboxes(sorted_masks_list, tokenizer)
            visual_prompt_images, prompt_tokens, _ = build_visual_prompt_matrices(
                sorted_masks_list, prompt_numbers=model.config.prompt_numbers
            )

            mask_values_list = []
            for vp_img in visual_prompt_images:
                vp_rgb = vp_img.convert("RGB")
                sub_masks = dynamic_preprocess(
                    vp_rgb,
                    min_num=processor.min_sub_img,
                    max_num=processor.max_sub_img,
                    image_size=processor.image_size[0],
                    use_thumbnail=True,
                )
                mv = processor.image_processor.preprocess(images=sub_masks, return_tensors="pt")["pixel_values"].to(device).to(dtype)
                mask_values_list.append(mv)

            questions = [args.prompt] 
            prompt_text = build_prompt_text(
                tokenizer=tokenizer,
                num_image_token=model.config.num_image_token,
                num_tiles=pixel_values.shape[0],
                questions=questions,
                gen_len=args.gen_length,
                num_masks=len(sorted_masks_list),
            )

            model_inputs = tokenizer(prompt_text, return_tensors="pt")
            input_ids = model_inputs["input_ids"].to(device)

            with torch.no_grad():
                outputs = model.generate(
                    pixel_values=pixel_values,
                    global_mask_values_list=mask_values_list,
                    aspect_ratios=aspect_ratio,
                    bboxes=[bboxes],
                    input_ids=input_ids,
                    steps=args.steps,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    tokenizer=tokenizer,
                    prompt_tokens=prompt_tokens,
                )
            
            decoded = tokenizer.decode(outputs[0], skip_special_tokens=False)
            ans_str = split_assistant_blocks(decoded)
            
            for mask_idx in range(len(sorted_masks_list)):
                #tag = f"<|Mask_Cap_{mask_idx}|>"
                #next_tag = f"<|Mask_Cap_{mask_idx+1}|>" if mask_idx + 1 < len(sorted_masks_list) else "<|eot_id|>"
                tag = f"<Prompt{mask_idx}>:"
                next_tag = f"<Prompt{mask_idx+1}>:" if mask_idx + 1 < len(sorted_masks_list) else "<|eot_id|>"
                
                start_idx = ans_str.find(tag)
                if start_idx != -1:
                    start_idx += len(tag)
                    end_idx = ans_str.find(next_tag, start_idx)
                    if end_idx == -1:
                        end_idx = ans_str.find("<|eot_id|>", start_idx)
                    
                    if end_idx == -1:
                        cap = ans_str[start_idx:].strip()
                    else:
                        cap = ans_str[start_idx:end_idx].strip()
                else:
                    cap = ""
                
                model_outputs[sorted_batch_ann_ids[mask_idx]] = cap
                print(f"Ann ID: {sorted_batch_ann_ids[mask_idx]}, Caption: {cap}")
            
            pbar.update(len(batch_ann_ids))
            
            if len(model_outputs) % 50 == 0:
                cache_model_outputs(cache_name, model_outputs, overwrite=True)

    pbar.close()
    cache_model_outputs(cache_name, model_outputs, overwrite=True)
    print(f"Finished. Results saved to {cache_name}")

"""
python infer_mask_captions_dlc.py \
  --model-path /path/to/PerceptionDLM \
  --prompt "Describe the masked region in detail." \
  --gen-length 32 --steps 32 --temperature 0.0 --top-p 1.0 \
  --cache-name-override dlc_output
"""
