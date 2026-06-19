import argparse
import os
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Dict, List, Optional

sys.path.append("../../")
from smallvlm.models.pdmllm.modeling_pdmllm import PDMLLM
from smallvlm.models.pdmllm.processing_pdmllm import (
    PDMLLMProcessor,
    find_closest_aspect_ratio,
    dynamic_preprocess,
)

from pycocotools import mask as mask_utils

def decode_rle(rle_dict: dict, h: int, w: int) -> np.ndarray:
    """Decode COCO-style RLE into a binary mask (H, W)."""
    rle = {"size": [h, w], "counts": rle_dict["counts"]}
    if isinstance(rle["counts"], list):
        rle = mask_utils.frPyObjects(rle, h, w)
    return mask_utils.decode(rle).astype(np.uint8)

def sort_masks_by_area(masks: List[np.ndarray]):
    """Return indices that sort masks from large area to small area."""
    areas = [np.sum(m) for m in masks]
    return np.argsort(np.array(areas))[::-1]


def build_visual_prompt(masks: List[np.ndarray], prompt_numbers: int) -> Image.Image:
    height, width = masks[0].shape
    visual_prompt = np.full((height, width), 255, dtype=np.uint8)
    for idx, mask in enumerate(masks):
        if idx >= prompt_numbers:
            raise ValueError(f"Mask index {idx} exceeds prompt_numbers {prompt_numbers}.")
        fill_area = (visual_prompt == 255) & (mask > 0)
        visual_prompt[fill_area] = idx
    return Image.fromarray(visual_prompt)


def build_visual_prompt_matrices(
    masks: List[np.ndarray],
    prompt_numbers: int,
) -> tuple:
    """Build per-mask visual prompt matrices for parallel inference.

    Each mask gets its own filled_matrix (starting from all-255) with only its
    mask region filled by its prompt_id.  Returns
    (visual_prompt_images, prompt_tokens, selected_prompt_indexes).
    """
    if len(masks) > prompt_numbers:
        raise ValueError(
            f"Number of masks ({len(masks)}) exceeds prompt_numbers ({prompt_numbers})."
        )
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


def build_bboxes(masks: List[np.ndarray], tokenizer) -> Dict[str, tuple]:
    height, width = masks[0].shape
    bboxes: Dict[str, tuple] = {}
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
    grid_w, grid_h = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, image.width, image.height, image_size
    )
    return torch.tensor([[grid_w, grid_h]], dtype=torch.int64)


def build_prompt_text(
    tokenizer, num_image_token: int, num_tiles: int,
    questions: List[str], gen_len: int, num_masks: int = 1,
) -> str:
    """Build parallel prompt text with all masks in a single user/assistant turn."""
    img_ctx = "".join(["<IMG_CONTEXT>"] * (num_image_token * num_tiles))
    # system
    parts = [
        "<|start_header_id|>system<|end_header_id|>\nYou are a helpful assistant.<|eot_id|>\n",
    ]
    # single user turn with image + all mask tokens
    parts.append("<|start_header_id|>user<|end_header_id|>\n")
    parts.append(
        f"<img>{img_ctx}</img>"
        + "\n".join([f"<|reserved_token_{i}|>" for i in range(num_masks)])
        + f"\n{questions[0]}<|eot_id|>\n"
    )
    # single assistant turn with all Mask_Cap blocks
    parts.append("<|start_header_id|>assistant<|end_header_id|>\n")
    mask_seq = "<|mdm_mask|>" * gen_len
    parts.append("\n".join([f"<|Mask_Cap_{i}|>{mask_seq}" for i in range(num_masks)]))

    return "".join(parts) + "<|eot_id|>"


def split_assistant_blocks(text: str, num_masks: int) -> List[str]:
    """Parse parallel output with <|Mask_Cap_i|> markers into per-mask captions."""
    # Extract the last assistant block
    blocks = text.split("<|start_header_id|>assistant<|end_header_id|>\n")
    assistant_text = blocks[-1].split("<|eot_id|>")[0] if len(blocks) > 1 else text

    captions = []
    for i in range(num_masks):
        start_tag = f"<|Mask_Cap_{i}|>"
        next_tag = f"<|Mask_Cap_{i + 1}|>"
        start_pos = assistant_text.find(start_tag)
        if start_pos == -1:
            captions.append("")
            continue
        content_start = start_pos + len(start_tag)
        end_pos = assistant_text.find(next_tag, content_start) if i < num_masks - 1 else len(assistant_text)
        if end_pos == -1:
            end_pos = len(assistant_text)
        captions.append(assistant_text[content_start:end_pos].strip())
    return captions


cache_base = "parallel_model_outputs_cache"

def cache_model_outputs(cache_name, cache_values, overwrite=False):
    os.makedirs(cache_base, exist_ok=True)
    with open(os.path.join(cache_base, cache_name + ".json"), "w" if overwrite else "x") as f:
        json.dump(cache_values, f, indent=4, ensure_ascii=False)

def load_cached_model_outputs(cache_name):
    cache_path = os.path.join(cache_base, cache_name + ".json")
    if not os.path.exists(cache_path):
        return {}
    print(f"Loading cache from {cache_path}")
    with open(cache_path, "r") as f:
        return json.load(f)

def load_anno_data(anno_json_path: str) -> dict:
    """Load annotation JSON and build lookup structures.

    Returns:
        anno_lookup: {image_id: {
            "file_name": str,
            "height": int,
            "width": int,
            "annotations": {ann_index: ann_dict}
        }}
    """
    with open(anno_json_path) as f:
        data = json.load(f)

    lookup = {}
    for item in data:
        image_id = str(item["image_id"])
        ann_by_index = {}
        for ann in item["annotations"]:
            ann_by_index[str(ann["index"])] = ann
        lookup[image_id] = {
            "file_name": item["file_name"],
            "height": item["height"],
            "width": item["width"],
            "annotations": ann_by_index,
        }
    return lookup

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run PDMLLM inference on anno-style data with RLE masks"
    )

    # Model arguments
    parser.add_argument("--model-path", required=True, help="Path to PDMLLM checkpoint.")
    parser.add_argument("--gen-length", type=int, default=128)
    parser.add_argument("--steps", type=int, default=128, help="Number of generation steps.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--prompt",
        default="Describe the masked region in detail.",
        help="Prompt suffix for each mask.",
    )

    # Data arguments
    parser.add_argument("--image-root", type=str, help="Image root")
    parser.add_argument(
        "--anno-json",
        required=True,
        help="Path to anno-style JSON file (with RLE masks).",
    )
    parser.add_argument(
        "--qa-json",
        default=None,
        help="Path to qa.json (keyed by ann_id). If provided, only infer annotations present in qa.json.",
    )
    parser.add_argument(
        "--class-names-json",
        default=None,
        help="Path to class_names.json (keyed by ann_id). If provided, only infer annotations present.",
    )

    # Cache arguments
    parser.add_argument("--suffix", type=str, default="", help="Suffix for cache name.")
    parser.add_argument("--cache-base", default=None, type=str, help="Override cache base dir.")
    parser.add_argument("--cache-name-override", type=str, default=None, help="Override cache name.")
    parser.add_argument("--save-interval", type=int, default=20, help="Save every N annotations.")

    args = parser.parse_args()

    if args.cache_base is not None:
        cache_base = args.cache_base

    # --- 1. Load Model ---
    print(f"Loading model from {args.model_path}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    #processor = load_processor_from_config(args.processor_config)
    processor = PDMLLMProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer = processor.tokenizer

    model = PDMLLM.from_pretrained(args.model_path, torch_dtype=dtype, trust_remote_code=True)
    model.to(device)
    model.eval()
    print("Model loaded.")

    # --- 2. Load anno data ---
    print(f"Loading anno data from {args.anno_json}...")
    anno_lookup = load_anno_data(args.anno_json)
    print(f"Loaded {len(anno_lookup)} images from anno JSON.")

    # --- 3. Determine which annotations to process ---
    # ann_id format: "{image_id}_{ann_index}"
    target_ann_ids = set()

    if args.qa_json and os.path.exists(args.qa_json):
        with open(args.qa_json) as f:
            qa_data = json.load(f)
        target_ann_ids.update(qa_data.keys())
        print(f"Loaded {len(qa_data)} annotation IDs from qa.json")
    elif args.class_names_json and os.path.exists(args.class_names_json):
        with open(args.class_names_json) as f:
            cn_data = json.load(f)
        target_ann_ids.update(cn_data.keys())
        print(f"Loaded {len(cn_data)} annotation IDs from class_names.json")
    else:
        # Process all annotations in the anno data
        for image_id, img_data in anno_lookup.items():
            for ann_index in img_data["annotations"]:
                target_ann_ids.add(f"{image_id}_{ann_index}")
        print(f"No qa.json/class_names.json provided; processing all {len(target_ann_ids)} annotations.")

    # --- 4. Group by image_id for efficient loading ---
    # {image_id: [ann_index, ...]}
    image_to_anns: Dict[str, List[str]] = {}
    for ann_id in target_ann_ids:
        # import pdb; pdb.set_trace()
        parts = ann_id.rsplit("_", 1)
        if len(parts) != 2:
            print(f"[WARN] Skipping malformed ann_id: {ann_id}")
            continue
        image_id, ann_index = parts
        if image_id not in anno_lookup:
            continue
        if ann_index not in anno_lookup[image_id]["annotations"]:
            continue
        image_to_anns.setdefault(image_id, []).append(ann_index)

    total_anns = sum(len(v) for v in image_to_anns.values())
    print(f"Will process {total_anns} annotations across {len(image_to_anns)} images.")

    # --- 5. Cache setup ---
    cache_name = (
        args.cache_name_override
        if args.cache_name_override
        else "pdmllm_anno_output" + args.suffix
    )
    model_outputs = load_cached_model_outputs(cache_name)
    print(f"Cache: {os.path.join(cache_base, cache_name + '.json')} "
          f"({len(model_outputs)} existing entries)")

    # --- 6. Inference loop ---
    import time
    total_generate_time = 0.0
    total_generated_tokens = 0
    pbar = tqdm(total=total_anns, desc="Inference")
    processed_count = 0

    for image_id, ann_indices in image_to_anns.items():
        img_data = anno_lookup[image_id]
        file_name = img_data["file_name"]
        image_name= img_data['file_name'].split('/')[-1] ##!
        file_name = os.path.join(args.image_root, image_name)
        img_h = img_data["height"]
        img_w = img_data["width"]

        # Check if all annotations for this image are already done
        pending = [idx for idx in ann_indices if f"{image_id}_{idx}" not in model_outputs]
        if not pending:
            pbar.update(len(ann_indices))
            continue

        # Load image
        if not os.path.exists(file_name):
            print(f"[WARN] Image not found: {file_name}, skipping.")
            pbar.update(len(ann_indices))
            continue

        try:
            pil_image = Image.open(file_name).convert("RGB")
        except Exception as e:
            print(f"[ERROR] Loading image {file_name}: {e}")
            pbar.update(len(ann_indices))
            continue

        # Preprocess image (once per image)
        sub_images = dynamic_preprocess(
            pil_image,
            min_num=processor.min_sub_img,
            max_num=processor.max_sub_img,
            image_size=processor.image_size[0],
            use_thumbnail=True,
        )
        pixel_values = (
            processor.image_processor.preprocess(images=sub_images, return_tensors="pt")["pixel_values"]
            .to(device)
            .to(dtype)
        )
        aspect_ratio = compute_aspect_ratio(pil_image, processor, num_tiles=pixel_values.shape[0]).to(device)

        MAX_MASKS = getattr(model.config, 'prompt_numbers', 6)  # Default fallback if not found

        for i in range(0, len(pending), MAX_MASKS):
            batch_pending = pending[i:i + MAX_MASKS]

            # Decode all pending masks and track mapping to ann_ids
            masks_list = []
            ann_id_order = []  # tracks which ann_id each mask index maps to
            for ann_index in batch_pending:
                ann = img_data["annotations"][ann_index]
                mask_np = decode_rle(ann["mask_rle"], img_h, img_w)
                masks_list.append(mask_np)
                ann_id_order.append(f"{image_id}_{ann_index}")
    
            # Sort masks by area (descending) — consistent with dataset training
            sort_idx = sort_masks_by_area(masks_list)
            masks_list = [masks_list[i] for i in sort_idx]
            ann_id_order = [ann_id_order[i] for i in sort_idx]
            num_masks = len(masks_list)
    
            # Build bboxes AFTER sorting so reserved_token_i matches the i-th sorted mask
            bboxes = build_bboxes(masks_list, tokenizer)
    
            # Build per-mask visual prompt matrices for parallel inference
            visual_prompt_images, prompt_tokens, _ = build_visual_prompt_matrices(
                masks_list, prompt_numbers=model.config.prompt_numbers
            )
    
            # Process each visual prompt separately → list of tensors
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
                mv = (
                    processor.image_processor.preprocess(images=sub_masks, return_tensors="pt")["pixel_values"]
                    .to(device)
                    .to(dtype)
                )
                mask_values_list.append(mv)
    
            # Build parallel text prompt
            questions = [args.prompt]
            prompt_text = build_prompt_text(
                tokenizer=tokenizer,
                num_image_token=model.config.num_image_token,
                num_tiles=pixel_values.shape[0],
                questions=questions,
                gen_len=args.gen_length,
                num_masks=num_masks,
            )
            model_inputs = tokenizer(prompt_text, return_tensors="pt")
            input_ids = model_inputs["input_ids"].to(device)
    
            # Generate (parallel for all masks)
            with torch.no_grad():
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
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
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                total_generate_time += (t1 - t0)
                print(f"  Generated batch of {len(batch_pending)} masks in {t1 - t0:.2f} seconds.")
                gen_len = args.gen_length * num_masks  # Approximate tokens generated (parallel masks)
                total_generated_tokens += gen_len
    
            decoded = tokenizer.decode(outputs[0], skip_special_tokens=False)
            captions = split_assistant_blocks(decoded, num_masks=num_masks)
    
            # Map captions back to ann_ids
            for idx, ann_id in enumerate(ann_id_order):
                caption = captions[idx] if idx < len(captions) else ""
                model_outputs[ann_id] = caption
                ann_index = ann_id.rsplit("_", 1)[1]
                cat_name = img_data["annotations"][ann_index].get("category_name", "?")
                print(f"  [{ann_id}] {cat_name}: {caption}")
    
            processed_count += len(batch_pending)
            pbar.update(len(batch_pending))
    
            # Periodic save
            if processed_count % args.save_interval < len(batch_pending):
                cache_model_outputs(cache_name, model_outputs, overwrite=True)

        # Also update for already-done annotations
        already_done = len(ann_indices) - len(pending)
        if already_done > 0:
            pbar.update(already_done)

    pbar.close()

    # Final save
    cache_model_outputs(cache_name, model_outputs, overwrite=True)
    print(f"\nFinished. {processed_count} new annotations processed.")
    print(f"Total cached: {len(model_outputs)} entries in {cache_name}")
    print(f"Total generate time: {total_generate_time:.2f} seconds")
    print(f"Total generated tokens: {total_generated_tokens}")
    print(f"TPS (tokens per second): {total_generated_tokens / total_generate_time:.2f}")


"""
python infer_mask_captions_paradlc.py \
    --model-path /path/to/PerceptionDLM \
    --image-root annotations/images \
    --anno-json annotations/annotations.json \
    --qa-json annotations/qa.json \
    --prompt "Describe the masked region in detail." \
    --gen-length 32 --steps 32 --temperature 0.0 --top-p 1.0 \
    --cache-name-override paradlc_outputs
"""