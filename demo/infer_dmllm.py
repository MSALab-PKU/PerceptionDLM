import torch
import argparse
from PIL import Image
from transformers import AutoModel, AutoProcessor
from smallvlm.models.dmllm.modeling_dmllm import DMLLM

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate DMLLM on Single Image')
    parser.add_argument("--model-path", required=True, help="Path to DMLLM checkpoint.")
    parser.add_argument("--image", required=True, help="Path to a single RGB image for direct inference.")
    parser.add_argument("--prompt", default="Describe the image in detail.", help="Question or prompt for the model.")
    parser.add_argument("--gen-length", type=int, default=128, help="Number of tokens to generate.")
    parser.add_argument("--block-length", type=int, default=128, help="Block length for generation.")
    parser.add_argument("--steps", type=int, default=128, help="Number of generation steps.")
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer = processor.tokenizer
    image_processor = processor.image_processor

    model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).eval().to(device)
    
    image = Image.open(args.image).convert('RGB')
    image_size = processor.image_size[0] if isinstance(processor.image_size, tuple) else processor.image_size
    images = dynamic_preprocess(
        image,
        min_num=getattr(processor, "min_sub_img", 1),
        max_num=getattr(processor, "max_sub_img", 6),
        image_size=image_size,
        use_thumbnail=True
    )
    pixel_values = processor.image_processor.preprocess(
        images=images, return_tensors="pt"
    )["pixel_values"].to(device).to(dtype)
    
    num_patches_list = [pixel_values.size(0)]

    # Format prompt
    formatted_prompt = f"<image>\n{args.prompt}"
    
    # Generation config mapping normal training args
    generation_config = {
        "steps": args.steps,
        "gen_length": args.gen_length,
        "block_length": args.block_length,
        "temperature": 0.0,
        "top_k": 0,
        "top_p": 1.0,
        "cfg_scale": 0.,
        "remasking": 'low_confidence',
    }

    with torch.no_grad():
        result = model.chat(
            tokenizer,
            pixel_values=pixel_values,
            num_patches_list=num_patches_list,
            question=formatted_prompt,
            generation_config=generation_config,
            verbose=False,
        )

    response = result.replace("<think>\n\n</think>", "").strip()
    
    print(f"Image: {args.image}")
    print(f"Question: {args.prompt}")
    print(f"Response: {response}")

"""
python demo/infer_dmllm.py \
  --model-path MSALab/PerceptionDLM-Base \
  --image assets/demo.jpg \
  --prompt "What color shirt is the man in the picture wearing?" \
  --gen-length 64 --block-length 64 --steps 64
"""
