"""Gradio visualization for PDMLLM (Parallel Diffusion MLLM) region captioning.

This app provides a dynamic alternative to the previous two-step static HTML
visualization (see demo/old_visualization/). It runs the same inference pipeline
as demo/infer_pdmllm.py, but uses the model's `generate_replace_noise` API to
capture the per-step denoising history so the diffusion decoding process can be
played back as an animation, step by step.

Run (activate the inference env first):
    source ~/dllm/bin/activate
    python demo/gradio/app.py --model-path MSALab/PerceptionDLM

Then open the printed URL (use SSH port forwarding if running on a remote box).
"""

import argparse
import html as html_lib
import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from typing import Dict, List, Tuple

import gradio as gr
from transformers import AutoModel, AutoProcessor

# Reuse the exact preprocessing helpers from the inference script so the Gradio
# pipeline stays consistent with demo/infer_pdmllm.py.
_DEMO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _DEMO_DIR not in sys.path:
    sys.path.insert(0, _DEMO_DIR)

from infer_pdmllm import (  # noqa: E402
    dynamic_preprocess,
    sort_masks_by_area,
    build_visual_prompt_matrices,
    build_bboxes,
    compute_aspect_ratio,
    build_prompt_text,
    split_assistant_blocks,
)

MASK_ID = 126336  # [MASK] token id for the LLaDA diffusion backbone.
# Internal sentinel marking a not-yet-revealed token. Rendered as an animated
# pending dot in render_caption_html (uses a Unicode private-use char so it
# never collides with real decoded text).
MASK_PLACEHOLDER = "\ue000"

# Globals populated by load_model().
MODEL = None
PROCESSOR = None
TOKENIZER = None
DEVICE = None
DTYPE = torch.bfloat16

# Default prompt mirrors infer_pdmllm.py.
DEFAULT_PROMPT = "Describe each masked region in detail."

# Overlay colors for the mask regions (R, G, B).
OVERLAY_COLORS = [
    (239, 68, 68),    # red
    (16, 185, 129),   # green
    (59, 130, 246),   # blue
    (245, 158, 11),   # amber
    (236, 72, 153),   # pink
    (139, 92, 246),   # violet
    (6, 182, 212),    # cyan
    (132, 204, 22),   # lime
]


def load_model(model_path: str):
    """Load the PDMLLM model + processor once and keep them on the GPU."""
    global MODEL, PROCESSOR, TOKENIZER, DEVICE
    print(f"Loading model from {model_path} ...")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PROCESSOR = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    TOKENIZER = PROCESSOR.tokenizer
    MODEL = AutoModel.from_pretrained(model_path, torch_dtype=DTYPE, trust_remote_code=True)
    MODEL.processor = PROCESSOR
    MODEL.to(DEVICE)
    MODEL.eval()
    print("Model loaded.")


def _to_binary_mask(mask_img: Image.Image, target_size: Tuple[int, int]) -> np.ndarray:
    """Convert an arbitrary mask image to a binary {0,1} array at target size."""
    arr = np.array(mask_img.convert("L").resize(target_size, Image.NEAREST))
    return (arr > 0).astype(np.uint8)


def make_overlay(pil_image: Image.Image, masks: List[np.ndarray], max_side: int = 768):
    """Build inputs for gr.AnnotatedImage.

    The base image (and masks) are resized so the longest side is at most
    max_side; this keeps the rendered AnnotatedImage from overflowing its
    container. Returns (base_image, [(mask_bool_array, "Region i"), ...]).
    AnnotatedImage renders each mask as a colored region and highlights it on
    hover, giving the interactive "hover to float/highlight" behavior.
    """
    base = pil_image.convert("RGB")
    w, h = base.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        base = base.resize(new_size, Image.BILINEAR)
    annotations = []
    for idx, mask in enumerate(masks):
        m = mask.astype(np.uint8)
        if scale < 1.0:
            m = np.array(
                Image.fromarray(m * 255).resize(base.size, Image.NEAREST)
            ) > 0
            m = m.astype(np.uint8)
        annotations.append((m, f"Region {idx}"))
    return (base, annotations)


def make_preset_thumbnail(image_path: str, mask_paths: List[str]) -> Image.Image:
    """Render a small preview: original image with mask regions overlaid in color."""
    img = Image.open(image_path).convert("RGB")
    base = np.array(img).astype(np.float32)
    for idx, mp in enumerate(mask_paths):
        m = _to_binary_mask(Image.open(mp), img.size).astype(bool)
        color = np.array(OVERLAY_COLORS[idx % len(OVERLAY_COLORS)], dtype=np.float32)
        base[m] = 0.45 * base[m] + 0.55 * color
    out = Image.fromarray(base.astype(np.uint8))
    out.thumbnail((320, 320))
    return out


def decode_step_captions(step_tokens: torch.Tensor, num_masks: int) -> List[str]:
    """Decode a single denoising step's token state into per-mask captions.

    Not-yet-revealed positions hold MASK_ID; we swap them for a visible
    placeholder so the animation shows tokens being filled in progressively.
    """
    ids = step_tokens[0].tolist()
    # Replace mask ids with a token we can render, decode the rest normally.
    pieces = []
    for tid in ids:
        if tid == MASK_ID:
            pieces.append(MASK_PLACEHOLDER)
        else:
            pieces.append(TOKENIZER.decode([tid], skip_special_tokens=False))
    raw = "".join(pieces)
    # The recorded region starts at <|Mask_Cap_0|>; split into per-mask blocks.
    captions = []
    for i in range(num_masks):
        start_tag = f"<|Mask_Cap_{i}|>"
        next_tag = f"<|Mask_Cap_{i + 1}|>"
        start_pos = raw.find(start_tag)
        if start_pos == -1:
            captions.append("")
            continue
        content_start = start_pos + len(start_tag)
        end_pos = raw.find(next_tag, content_start) if i < num_masks - 1 else len(raw)
        if end_pos == -1:
            end_pos = len(raw)
        text = raw[content_start:end_pos]
        # Strip remaining structural special tokens but keep mask placeholders.
        for tok in ("<|eot_id|>", "<|mdm_mask|>"):
            text = text.replace(tok, "")
        captions.append(text.strip())
    return captions


def _render_caption_body(cap: str, prev_cap: str, color: tuple, highlight: bool = True) -> str:
    """Render a single caption string into animated HTML.

    Not-yet-revealed tokens (MASK_PLACEHOLDER) become pulsing dots colored like
    the region; tokens newly revealed vs the previous step are briefly
    highlighted using that same region color. On the final frame (highlight=
    False) no token is highlighted.
    """
    rgb = f"rgb{color}"
    out = []
    # Compare aligned non-placeholder content to detect newly revealed chars.
    prev_revealed = prev_cap.replace(MASK_PLACEHOLDER, "") if prev_cap else ""
    seen_real = 0
    for ch in cap:
        if ch == MASK_PLACEHOLDER:
            out.append(
                f'<span class="tok-pending" style="background:{rgb};"></span>'
            )
        else:
            seen_real += 1
            is_new = highlight and seen_real > len(prev_revealed)
            esc = html_lib.escape(ch)
            if is_new:
                out.append(f'<span class="tok-new" style="background:{rgb};">{esc}</span>')
            else:
                out.append(esc)
    if not cap:
        return '<span class="tok-empty">…</span>'
    return "".join(out)


def render_caption_html(
    captions: List[str],
    prev_captions: List[str],
    step_idx: int,
    total_steps: int,
) -> str:
    """Render all per-mask captions for a step as a styled HTML block.

    step_idx is 0-based; the number of denoising steps shown is total_steps - 1
    (history holds the initial all-mask state plus one entry per step).
    """
    last_step = max(total_steps - 1, 1)
    is_final = step_idx >= total_steps - 1
    pct = int(round(step_idx / last_step * 100))
    css = """
<style>
.dec-wrap { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
.dec-head { display:flex; align-items:center; gap:12px; margin-bottom:14px; }
.dec-step { font-size:0.95em; font-weight:600; color:#475569; white-space:nowrap; }
.dec-progress { flex:1; height:6px; background:#e2e8f0; border-radius:3px; overflow:hidden; }
.dec-progress-fill { height:100%; background:linear-gradient(90deg,#6366f1,#a855f7); border-radius:3px; transition:width 0.25s ease; }
.cap-card { border:1px solid #e2e8f0; border-radius:12px; padding:14px 16px; margin-bottom:12px; background:#fff; box-shadow:0 1px 3px rgba(0,0,0,0.04); }
.cap-title { display:flex; align-items:center; gap:8px; font-weight:600; font-size:0.9em; margin-bottom:8px; color:#1e293b; }
.cap-dot { width:13px; height:13px; border-radius:50%; flex-shrink:0; }
.cap-body { font-size:0.95em; line-height:1.75; color:#0f172a; word-break:break-word; }
.tok-pending { display:inline-block; width:0.55em; height:0.55em; border-radius:50%; margin:0 1px; opacity:0.35; vertical-align:middle; animation:tokpulse 1.1s ease-in-out infinite; }
@keyframes tokpulse { 0%,100%{opacity:0.18;transform:scale(0.8);} 50%{opacity:0.6;transform:scale(1.05);} }
.tok-new { color:#fff; border-radius:4px; padding:0 2px; animation:tokreveal 0.45s ease-out; }
@keyframes tokreveal { from{opacity:0;transform:translateY(-3px) scale(0.9);} to{opacity:1;transform:none;} }
.tok-empty { color:#94a3b8; font-style:italic; }
</style>
"""
    parts = [css, '<div class="dec-wrap">']
    parts.append(
        f'<div class="dec-head"><span class="dec-step">Step {step_idx} / {last_step}</span>'
        f'<div class="dec-progress"><div class="dec-progress-fill" style="width:{pct}%;"></div></div></div>'
    )
    for i, cap in enumerate(captions):
        color = OVERLAY_COLORS[i % len(OVERLAY_COLORS)]
        prev = prev_captions[i] if prev_captions and i < len(prev_captions) else ""
        body = _render_caption_body(cap, prev, color, highlight=not is_final)
        parts.append(
            f'<div class="cap-card"><div class="cap-title">'
            f'<span class="cap-dot" style="background:rgb{color};"></span>Region {i}</div>'
            f'<div class="cap-body">{body}</div></div>'
        )
    parts.append("</div>")
    return "".join(parts)


@torch.no_grad()
def run_inference(
    pil_image: Image.Image,
    mask_images: List[Image.Image],
    prompt: str,
    gen_length: int,
    steps: int,
    temperature: float,
    top_p: float,
):
    """Run the full PDMLLM pipeline and return per-step decoding history.

    Returns: (history, num_masks, masks_list) where history is a list of
    per-mask caption lists (one entry per recorded denoising step).
    """
    prompt = prompt or DEFAULT_PROMPT
    target_size = pil_image.size

    masks_list = [_to_binary_mask(m, target_size) for m in mask_images]

    # Image preprocessing (identical to infer_pdmllm.py).
    sub_images = dynamic_preprocess(
        pil_image,
        min_num=PROCESSOR.min_sub_img,
        max_num=PROCESSOR.max_sub_img,
        image_size=PROCESSOR.image_size[0],
        use_thumbnail=True,
    )
    pixel_values = PROCESSOR.image_processor.preprocess(
        images=sub_images, return_tensors="pt"
    )["pixel_values"].to(DEVICE).to(DTYPE)
    aspect_ratio = compute_aspect_ratio(
        pil_image, PROCESSOR, num_tiles=pixel_values.shape[0]
    ).to(DEVICE)

    # Sort masks by area (descending) — consistent with dataset/training.
    sort_idx = sort_masks_by_area(masks_list)
    masks_list = [masks_list[i] for i in sort_idx]

    bboxes = build_bboxes(masks_list, TOKENIZER)
    visual_prompt_images, prompt_tokens, _ = build_visual_prompt_matrices(
        masks_list, prompt_numbers=MODEL.config.prompt_numbers
    )

    mask_values_list = []
    for vp_img in visual_prompt_images:
        vp_rgb = vp_img.convert("RGB")
        sub_masks = dynamic_preprocess(
            vp_rgb,
            min_num=PROCESSOR.min_sub_img,
            max_num=PROCESSOR.max_sub_img,
            image_size=PROCESSOR.image_size[0],
            use_thumbnail=True,
        )
        mv = PROCESSOR.image_processor.preprocess(
            images=sub_masks, return_tensors="pt"
        )["pixel_values"].to(DEVICE).to(DTYPE)
        mask_values_list.append(mv)

    questions = [prompt for _ in masks_list]
    prompt_text = build_prompt_text(
        tokenizer=TOKENIZER,
        num_image_token=MODEL.config.num_image_token,
        num_tiles=pixel_values.shape[0],
        questions=questions,
        gen_len=gen_length,
        num_masks=len(masks_list),
    )

    model_inputs = TOKENIZER(prompt_text, return_tensors="pt")
    input_ids = model_inputs["input_ids"].to(DEVICE)

    # generate_replace_noise returns (final_tokens, all_steps_responses) where
    # all_steps_responses is a list of token tensors over the recorded region,
    # one per denoising step — exactly what we need for the animation.
    _, all_steps = MODEL.generate_replace_noise(
        pixel_values=pixel_values,
        global_mask_values_list=mask_values_list,
        aspect_ratios=aspect_ratio,
        bboxes=[bboxes],
        input_ids=input_ids,
        steps=steps,
        temperature=temperature,
        top_p=top_p,
        tokenizer=TOKENIZER,
        prompt_tokens=prompt_tokens,
    )

    num_masks = len(masks_list)
    history = [decode_step_captions(step_tok, num_masks) for step_tok in all_steps]
    return history, num_masks, masks_list


def build_demo(preset_cases: Dict[str, dict], play_delay: float = 0.35):
    # Color map so AnnotatedImage region colors match the caption card dots.
    preset_keys = list(preset_cases.keys())
    color_map = {
        f"Region {i}": "#%02x%02x%02x" % OVERLAY_COLORS[i % len(OVERLAY_COLORS)]
        for i in range(len(OVERLAY_COLORS))
    }

    custom_css = """
.region-anno { overflow:hidden; }
.region-anno img, .region-anno canvas { max-width:100%; height:auto; object-fit:contain; }
"""

    with gr.Blocks(title="PerceptionDLM Region Captioning", theme=gr.themes.Soft(), css=custom_css) as demo:
        gr.Markdown(
            "# 🎯 PerceptionDLM Region Captioning\n"
            "A diffusion multimodal LLM that captions any region of an image **in parallel**. "
            "Pick a preset case below or upload your own image and binary mask(s), then run inference — "
            "hover over a region to highlight it, and replay the diffusion decoding to watch each "
            "caption emerge token by token."
        )

        # Per-session state holding the decoding history.
        history_state = gr.State([])
        num_masks_state = gr.State(0)

        with gr.Row():
            with gr.Column(scale=1):
                if preset_cases:
                    gr.Markdown("**Preset cases** — click a thumbnail to load")
                    preset_gallery = gr.Gallery(
                        value=[
                            (make_preset_thumbnail(c["image"], c["masks"]), name)
                            for name, c in preset_cases.items()
                        ],
                        columns=3,
                        height="auto",
                        object_fit="cover",
                        allow_preview=False,
                        label=None,
                        show_label=False,
                    )
                image_in = gr.Image(type="pil", label="Image", image_mode="RGB")
                mask_in = gr.File(
                    file_count="multiple",
                    file_types=["image"],
                    label="Mask images (binary, ≥1)",
                )
                prompt_in = gr.Textbox(value=DEFAULT_PROMPT, label="Prompt")
                with gr.Row():
                    gen_len_in = gr.Slider(8, 128, value=64, step=8, label="Gen length")
                    steps_in = gr.Slider(8, 128, value=32, step=8, label="Steps")
                run_btn = gr.Button("Run inference", variant="primary")

            with gr.Column(scale=1):
                overlay_out = gr.AnnotatedImage(
                    label="Regions (hover to highlight)",
                    color_map=color_map,
                    elem_classes=["region-anno"],
                )
                with gr.Row():
                    step_slider = gr.Slider(
                        0, 1, value=0, step=1, label="Decoding step", interactive=True, scale=4
                    )
                    play_btn = gr.Button("▶ Play", variant="secondary", scale=1)
                captions_out = gr.HTML()

        # ---- Preset loading via gallery click ----
        def load_preset(evt: gr.SelectData):
            name = preset_keys[evt.index]
            case = preset_cases[name]
            img = Image.open(case["image"]).convert("RGB")
            return img, case["masks"]

        if preset_cases:
            preset_gallery.select(
                load_preset, inputs=None, outputs=[image_in, mask_in]
            )

        # ---- Run inference ----
        def _on_run(image, mask_files, prompt, gen_len, steps):
            if image is None:
                raise gr.Error("Please provide an image.")
            if not mask_files:
                raise gr.Error("Please provide at least one mask image.")
            mask_paths = [f if isinstance(f, str) else f.name for f in mask_files]
            mask_images = [Image.open(p) for p in mask_paths]

            history, num_masks, masks_list = run_inference(
                image, mask_images, prompt, int(gen_len), int(steps),
                temperature=0.0, top_p=1.0,
            )
            overlay = make_overlay(image, masks_list)
            total = len(history)
            last = total - 1
            html = render_caption_html(history[last], history[last - 1] if last > 0 else [], last, total)
            slider_update = gr.update(minimum=0, maximum=last, value=last, step=1)
            return history, num_masks, overlay, slider_update, html

        run_btn.click(
            _on_run,
            inputs=[image_in, mask_in, prompt_in, gen_len_in, steps_in],
            outputs=[history_state, num_masks_state, overlay_out, step_slider, captions_out],
        )

        # ---- Step slider scrubbing (step_idx is 0-based) ----
        def _on_step(step_idx, history):
            if not history:
                return gr.update()
            total = len(history)
            i = int(step_idx)
            i = max(0, min(i, total - 1))
            prev = history[i - 1] if i > 0 else []
            return render_caption_html(history[i], prev, i, total)

        step_slider.change(_on_step, inputs=[step_slider, history_state], outputs=[captions_out])

        # ---- Play animation (generator streams each step, paced by play_delay) ----
        def _on_play(history):
            if not history:
                yield gr.update(), gr.update()
                return
            total = len(history)
            for i in range(total):
                prev = history[i - 1] if i > 0 else []
                html = render_caption_html(history[i], prev, i, total)
                yield gr.update(value=i), html
                if i < total - 1:
                    time.sleep(play_delay)

        play_btn.click(_on_play, inputs=[history_state], outputs=[step_slider, captions_out])

    return demo


def discover_presets() -> Dict[str, dict]:
    """Auto-discover preset cases from the repo's assets/ directory."""
    assets = os.path.join(os.path.dirname(_DEMO_DIR), "assets")
    presets: Dict[str, dict] = {}
    demo_img = os.path.join(assets, "demo.jpg")
    if os.path.exists(demo_img):
        masks = sorted(
            os.path.join(assets, f)
            for f in os.listdir(assets)
            if f.startswith("demo_mask_") and f.endswith(".jpg")
        )
        if masks:
            presets["demo (assets/demo.jpg)"] = {"image": demo_img, "masks": masks}
    return presets


def main():
    parser = argparse.ArgumentParser(description="Gradio visualizer for PDMLLM region captioning.")
    parser.add_argument("--model-path", required=True, help="Path to PDMLLM checkpoint.")
    parser.add_argument("--server-name", default="0.0.0.0")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link.")
    parser.add_argument("--play-delay", type=float, default=0.25,
                        help="Seconds between frames when playing the decoding animation (larger = slower).")
    args = parser.parse_args()

    load_model(args.model_path)
    presets = discover_presets()
    demo = build_demo(presets, play_delay=args.play_delay)
    demo.queue().launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
