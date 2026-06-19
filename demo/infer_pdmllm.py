
import argparse
import os
import json
import random
import torch
import numpy as np
from PIL import Image
from typing import Dict, List
import sys
import yaml
from transformers import AutoModel, AutoProcessor

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=6, image_size=512, use_thumbnail=True):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def sort_masks_by_area(masks: List[np.ndarray]):
    """Return indices that sort masks from large area to small area."""
    areas = [np.sum(m) for m in masks]
    return np.argsort(np.array(areas))[::-1]

def build_visual_prompt_matrices(
    masks: List[np.ndarray],
    prompt_numbers: int,
) -> tuple:
    """Build per-mask visual prompt matrices matching the dataset implementation.

    Each mask gets its own filled_matrix (starting from all-255) with only its
    mask region filled by a randomly-assigned prompt_id.  Returns
    (visual_prompt_images, prompt_tokens, selected_prompt_indexes).
    """
    if len(masks) > prompt_numbers:
        raise ValueError(
            f"Number of masks ({len(masks)}) exceeds prompt_numbers ({prompt_numbers})."
        )
    height, width = masks[0].shape

    # Random prompt-id assignment (same as dataset training code)
    prompt_indexes = list(range(prompt_numbers))
    ## random.shuffle(prompt_indexes)
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

    return torch.tensor([[grid_w, grid_h]], dtype=torch.int64)

def build_prompt_text(tokenizer, num_image_token: int, num_tiles: int, questions: List[str], gen_len: int, num_masks: int) -> str:
    img_ctx = "".join(["<IMG_CONTEXT>"] * (num_image_token * num_tiles))
    # system
    parts = ["<|start_header_id|>system<|end_header_id|>\nYou are a helpful assistant.<|eot_id|>\n"]
    # single user turn with image + all mask tokens
    parts.append("<|start_header_id|>user<|end_header_id|>\n")
    parts.append(f"<img>{img_ctx}</img>" + "\n".join([f"<|reserved_token_{i}|>" for i in range(num_masks)]) + f"\n{questions[0]}<|eot_id|>\n")
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate PDMLLM on DLC-Bench')
    parser.add_argument("--model-path", required=True, help="Path to PDMLLM checkpoint.")
    parser.add_argument("--gen-length", type=int, default=32, help="Number of tokens to generate for each mask.")
    parser.add_argument("--steps", type=int, default=32, help="Number of generation steps.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--prompt", default="Describe the masked region in detail.", help="Prompt suffix for each mask.")
    parser.add_argument("--image", default=None, help="Path to a single RGB image for direct inference.")
    parser.add_argument("--masks", nargs="+", default=None, help="Paths to binary mask images (>=1) for direct inference.")
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer = processor.tokenizer

    model = AutoModel.from_pretrained(args.model_path, torch_dtype=dtype, trust_remote_code=True)
    model.processor = processor
    
    model.to(device)
    model.eval()
    print("Model loaded.")

    if args.image is not None and args.masks is not None:
        print(f"Running inference on {args.image} with {len(args.masks)} mask(s)...")
        pil_image = Image.open(args.image).convert("RGB")

        # Load masks from file paths
        masks_list = []
        for m_path in args.masks:
            m_arr = np.array(Image.open(m_path).convert("L").resize(pil_image.size, Image.NEAREST))
            masks_list.append((m_arr > 0).astype(np.uint8))

        # Preprocess image
        sub_images = dynamic_preprocess(
            pil_image,
            min_num=processor.min_sub_img,
            max_num=processor.max_sub_img,
            image_size=processor.image_size[0],
            use_thumbnail=True,
        )
        pixel_values = processor.image_processor.preprocess(images=sub_images, return_tensors="pt")["pixel_values"].to(device).to(dtype)
        aspect_ratio = compute_aspect_ratio(pil_image, processor, num_tiles=pixel_values.shape[0]).to(device)

        # Sort masks by area (descending) — consistent with dataset training
        sort_idx = sort_masks_by_area(masks_list)
        masks_list = [masks_list[i] for i in sort_idx]

        # Build bboxes AFTER sorting so reserved_token_i matches the i-th sorted mask
        bboxes = build_bboxes(masks_list, tokenizer)

        # Build per-mask visual prompt matrices (matching dataset implementation)
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
            mv = processor.image_processor.preprocess(images=sub_masks, return_tensors="pt")["pixel_values"].to(device).to(dtype)
            mask_values_list.append(mv)

        prompt = "Describe each masked region in detail."
        questions = [prompt for _ in masks_list]
        prompt_text = build_prompt_text(
            tokenizer=tokenizer,
            num_image_token=model.config.num_image_token,
            num_tiles=pixel_values.shape[0],
            questions=questions,
            gen_len=args.gen_length,
            num_masks=len(masks_list),
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
        assistant_blocks = split_assistant_blocks(decoded, num_masks=len(masks_list))
        for i, caption in enumerate(assistant_blocks):
            print(f"[Caption {i}]: {caption}")


"""
python demo/infer_pdmllm.py \
  --model-path MSALab/PerceptionDLM \
  --image assets/demo.jpg \
  --masks assets/demo_mask_0.jpg \
          assets/demo_mask_1.jpg \
          assets/demo_mask_2.jpg \
  --gen-length 32 --steps 32 --temperature 0.0 --top-p 1.0
"""