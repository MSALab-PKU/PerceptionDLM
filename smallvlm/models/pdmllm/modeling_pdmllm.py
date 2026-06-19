from typing import Optional, List
import re
import torch
import torchvision
import transformers
from einops import rearrange
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedModel, AutoModel, AutoModelForCausalLM, GenerationConfig
from transformers import AutoConfig
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.feature_extraction_utils import BatchFeature
from .configuration_pdmllm import PDMLLMConfig
from .modeling_abstractor import PerceiverProjection
from .modeling_llada import LLaDAModelLM
from .cache import *
from .configuration_llada import LLaDAConfig

def build_vision_model(config, model=None):
    assert hasattr(config, "name_or_path")
    if model is None:
        model = AutoModel.from_pretrained(
            config.name_or_path, config=config, trust_remote_code=True)
    return model

def vit_forward_with_mask(
    self,
    pixel_values,
    interpolate_pos_encoding: bool = False,
    mask_embeddings=None,
    output_hidden_states: bool = False,
    **kwargs,
):
    attention_mask = kwargs.pop("attention_mask", None)
    kwargs.pop("output_hidden_states", None)
    kwargs.pop("output_attentions", None)

    _, _, height, width = pixel_values.shape
    target_dtype = self.embeddings.patch_embedding.weight.dtype
    patch_embeds = self.embeddings.patch_embedding(pixel_values.to(dtype=target_dtype))  # shape = [*, width, grid, grid]
    embeddings = patch_embeds.flatten(2).transpose(1, 2)

    #hidden_states = self.embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
    if mask_embeddings is not None:
        embeddings = embeddings + mask_embeddings.to(embeddings.device, dtype=embeddings.dtype)

    if interpolate_pos_encoding:
        embeddings = embeddings + self.embeddings.interpolate_pos_encoding(embeddings, height, width)
    else:
        embeddings = embeddings + self.embeddings.position_embedding(self.embeddings.position_ids)

    collected_hs = [] if output_hidden_states else None
    for layer in self.encoder.layers:
        hs = layer(embeddings, attention_mask=attention_mask)
        if isinstance(hs, tuple):
            hs = hs[0]
        embeddings = hs
        if collected_hs is not None:
            collected_hs.append(embeddings)

    last_hidden_state = self.post_layernorm(embeddings)
    pooler_output = self.head(last_hidden_state) if self.use_head else None

    return BaseModelOutputWithPooling(
        last_hidden_state=last_hidden_state,
        pooler_output=pooler_output,
        hidden_states=tuple(collected_hs) if collected_hs is not None else None,
    )


class PDMLLM(PreTrainedModel):
    config_class = PDMLLMConfig
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = "past_key_values"
    _supports_cache_class = False
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    accepts_loss_kwargs=False

    def __init__(self,
                 config: PDMLLMConfig,
                 language_model=None,
                 vision_model=None,
                 processor=None,
                ):
        super().__init__(config)
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.downsample_ratio = config.downsample_ratio
        self.num_image_token = config.num_image_token
        self.vision_select_layer = config.vision_select_layer
        self.replacement_noise_mode = config.replacement_noise_mode

        try:
            vision_hidden_states = self.config.vision_model_config.hidden_size
        except:
            vision_hidden_states = self.config.vision_model_config.vision_config.hidden_size
            self.config.vision_model_config.hidden_size = vision_hidden_states

        vision_model = build_vision_model(config.vision_model_config, vision_model)

        vision_abstractor = PerceiverProjection(**config.vision_abstractor_config,
                                                in_dim=self.config.vision_model_config.hidden_size * (int(1 / self.downsample_ratio) ** 2),
                                                out_dim=self.config.language_model_config.hidden_size)

        if language_model is None:
            kwargs_ = {}
            if config._attn_implementation_internal is not None:
                kwargs_['attn_implementation'] = config._attn_implementation_internal
            if 'llada' in config.language_model_config.name_or_path.lower():
                with transformers.modeling_utils.no_init_weights():
                    language_model = LLaDAModelLM(config.language_model_config)
            else:
                raise ValueError(f"Unsupported language model: {config.language_model_config.name_or_path}")

        self.vision_model = vision_model
        self.vision_abstractor = vision_abstractor
        self.language_model = language_model

        # self.mask_patch_embedding = nn.Conv2d(
        #     in_channels=1,
        #     out_channels=config.mask_patch_embedding_out_channels,
        #     kernel_size=config.kernel_size,
        #     stride=config.kernel_size,
        #     bias=False,
        # )
        
        self.mask_id_embedding = nn.Embedding(config.prompt_numbers, config.vision_model_config.vision_config.hidden_size)

        #self.vit = self.vision_model.vision_model
        #self.vit.forward = vit_forward_with_mask.__get__(self.vit, self.vit.__class__)
        self.vision_model.vision_model.forward = vit_forward_with_mask.__get__(self.vision_model.vision_model, self.vision_model.vision_model.__class__)

        # zero-init
        # for param in self.mask_patch_embedding.parameters():
        #     nn.init.zeros_(param)
        
        if processor is not None:
            self.processor = processor
        
        self.prompt_numbers = config.prompt_numbers
        # Optional override for how many RoI-aligned tokens replace a crop token.
        self.roi_output_size = getattr(config, "roi_output_size", None)
        
        # Only add special tokens when a processor is available (i.e. during training).
        # During inference via from_pretrained, the tokens are already in the saved tokenizer.
        if hasattr(self, "processor"):
            self._add_special_tokens()
        self.gradient_checkpointing_enable()

    def _add_special_tokens(self):
        assert hasattr(self, "processor")

        visual_prompt_nums = self.prompt_numbers
        visual_prompt_tokens = [f"<Prompt{i}>" for i in range(visual_prompt_nums)]
        visual_prompt_tokens.append("<NO_Prompt>")
        special_tokens = visual_prompt_tokens
        num_new_tokens = self.processor.tokenizer.add_tokens(
            special_tokens, special_tokens=True
        )
        self.language_model.resize_token_embeddings(len(self.processor.tokenizer))
        print(f"Added {num_new_tokens} special tokens.")

    def forward_vision(self, pixel_values, global_mask_values_list=None, prompt_tokens=None):
        # pixel_values (n, c, h, w)

        # Unwrap BatchFeature if needed
        if isinstance(pixel_values, BatchFeature):
            pixel_values = pixel_values["pixel_values"]

        # Precompute mask embeddings so they can be injected before the vision encoder.
        mask_embeds = None
        if global_mask_values_list is not None:
            if isinstance(global_mask_values_list, BatchFeature):
                mask_values_list = global_mask_values_list.get("pixel_values_list", None)
            else:
                mask_values_list = global_mask_values_list
            if mask_values_list is not None:
                K = self.config.kernel_size[0]
                h_patches = pixel_values.shape[2] // K
                w_patches = pixel_values.shape[3] // K
                mask_embeds = torch.zeros(
                    pixel_values.shape[0],
                    self.config.vision_model_config.vision_config.hidden_size,
                    h_patches, w_patches,
                    dtype=pixel_values.dtype,
                    device=pixel_values.device,
                )
                for prompt_token, mask_values in zip(prompt_tokens, mask_values_list):
                    prompt_id = int(re.search(r"<Prompt(\d+)>", prompt_token).group(1))
                    vp_id = torch.tensor(prompt_id, device=pixel_values.device)
                    vp_embed = self.mask_id_embedding(vp_id).to(pixel_values.device)  # (C,)
                    
                    if mask_values.shape[1] > 1:
                        mask_values = mask_values.mean(dim=1, keepdim=True)
                    mask_values = mask_values.to(pixel_values.device)
                    mask_values = torch.round((mask_values + 1.0) / 2.0 * 255.0).long()
                    mask_values = torch.clamp(mask_values, min=0, max=255)
                    binary_mask = (mask_values != 255).to(pixel_values.dtype)  # (B, 1, H, W)
                    
                    ## mask_patch_embeds = self.mask_patch_embedding(binary_mask)  # (B, C, h_patches, w_patches)
                    
                    active_patches = torch.nn.functional.interpolate(
                        binary_mask,
                        size=(h_patches, w_patches),
                        mode='nearest'
                    )  # (B, 1, h_patches, w_patches)

                    # Add mask id embedding (at active patches) + mask conv embedding
                    mask_embeds = mask_embeds + vp_embed.view(1, -1, 1, 1) * active_patches ## + mask_patch_embeds

                mask_embeds = mask_embeds.flatten(2).transpose(1, 2)  # (B, num_patches, C)

        vision_outputs = None
        if mask_embeds is not None:
            vision_outputs = self.vision_model.vision_model(
                pixel_values=pixel_values,
                mask_embeddings=mask_embeds,
                output_hidden_states=True,
            )
        
        assert vision_outputs is not None
        if self.vision_select_layer == -1:
            image_embeddings = vision_outputs.last_hidden_state
        else:
            image_embeddings = vision_outputs.hidden_states[self.vision_select_layer] # (B, N, C)

        # Keep all tile embeddings — do NOT filter by image_flags.
        # All tiles are real crops from a single image (produced by dynamic_preprocess).
        # Filtering by pixel-sum==0 can incorrectly drop tiles whose normalized
        # pixel values happen to sum to zero, causing shape mismatches with
        # input_ids image tokens and aspect_ratios in downstream _merge / RoI-align.
        vit_embeds = image_embeddings
        
        if self.downsample_ratio != 1:
            patch_num = self.image_size // self.patch_size
            vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], patch_num, patch_num, vit_embeds.shape[-1])
            vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
            vit_embeds = vit_embeds.flatten(1, 2)

        vit_embeds = self.vision_abstractor(vit_embeds)

        return vit_embeds

    def prepare_for_lm(self, input_ids, vision_embeds):
        inputs_embeds = self.get_input_embeddings()(input_ids)
        vision_embeds_ = vision_embeds
        if vision_embeds is not None:
            try:
                vision_mask = input_ids == self.config.image_token_id
                if torch.count_nonzero(vision_mask).item() != vision_embeds.shape[:-1].numel():
                    info = "vision embeddings mismatch input embeddings: " \
                           f"vision_mask shape={vision_mask.shape}; " \
                           f"vision_mask count={torch.count_nonzero(vision_mask)}; " \
                           f"vision_embeds shape={vision_embeds.shape}"
                    # print(info)
                    num_vision_1 = torch.count_nonzero(vision_mask).item()
                    num_vision_2 = vision_embeds.shape[:-1].numel()
                    vision_embeds = vision_embeds.contiguous()
                    if num_vision_1 <= num_vision_2:
                        vision_embeds = vision_embeds.view(-1, vision_embeds.size(-1))[:num_vision_1]
                    else:
                        vision_embeds = vision_embeds.view(-1, vision_embeds.size(-1))
                        less_nums = num_vision_1 - num_vision_2
                        vision_embeds = torch.cat([vision_embeds, vision_embeds[-less_nums:]], dim=0)
                    vision_embeds = vision_embeds.contiguous()

                # assert torch.count_nonzero(vision_mask).item() == vision_embeds.shape[:-1].numel(), \
                #     "vision embeddings mismatch input embeddings: " \
                #     f"vision_mask shape={vision_mask.shape}; " \
                #     f"vision_mask count={torch.count_nonzero(vision_mask)}; " \
                #     f"vision_embeds shape={vision_embeds.shape}"
                inputs_embeds = torch.masked_scatter(inputs_embeds, vision_mask.unsqueeze(-1),
                                                     vision_embeds.to(inputs_embeds.dtype).view(-1,
                                                                                                vision_embeds.size(-1)))
            except:
                inputs_embeds = inputs_embeds + torch.sum(vision_embeds_[0, 0, :]) * 0.0

        return inputs_embeds

    def _prepare_inputs_for_generation(
        self,
        input_ids,
        pixel_values=None,
        global_mask_values_list=None,
        aspect_ratios=None,
        bboxes=None,
        prompt_tokens=None,
        attention_mask=None,
        position_ids=None,
        tokenizer=None,
    ):
        vision_embeds = None
        if pixel_values is not None:
            vision_embeds = self.forward_vision(pixel_values, global_mask_values_list=global_mask_values_list, prompt_tokens=prompt_tokens)

        inputs_embeds = self.prepare_for_lm(input_ids, vision_embeds)
        reserved_token_spans: List[List[tuple]] = [[] for _ in range(input_ids.shape[0])]

        length_changed = False
        if vision_embeds is not None and aspect_ratios is not None and bboxes is not None:
            crop_tokens = [
                tokenizer.convert_tokens_to_ids(f"<|reserved_token_{pid}|>")
                for pid in range(self.prompt_numbers)
            ]

            patch_num = self.image_size // self.patch_size
            if self.downsample_ratio != 1:
                feat_h = int(patch_num * self.downsample_ratio)
                feat_w = int(patch_num * self.downsample_ratio)
            else:
                feat_h = patch_num
                feat_w = patch_num

            if vision_embeds.shape[0] != 1:
                image_features_tiles = rearrange(
                    vision_embeds[1:].unsqueeze(0), "b n (h w) c -> b n c h w", h=feat_h, w=feat_w
                )
            else:
                image_features_tiles = rearrange(
                    vision_embeds.unsqueeze(0), "b n (h w) c -> b n c h w", h=feat_h, w=feat_w
                )

            new_inputs_embeds = []
            new_input_ids_list = []
            assert inputs_embeds.shape[0] == 1, "Currently only support batch_size=1"

            for batch_idx in range(inputs_embeds.shape[0]):
                curr_inputs_embeds = inputs_embeds[batch_idx]
                curr_input_ids = input_ids[batch_idx]

                replacements = []
                orig_input_ids = input_ids[batch_idx]
                for cap_idx, crop_token in enumerate(crop_tokens):
                    target_mask = orig_input_ids.eq(crop_token)
                    if not target_mask.any():
                        continue
                    target_indices = target_mask.nonzero().squeeze()
                    if target_indices.ndim == 0:
                        head_idx = tail_idx = target_indices.item()
                    else:
                        head_idx = target_indices.min().item()
                        tail_idx = target_indices.max().item()
                    replacements.append((head_idx, tail_idx, cap_idx, crop_token))
                # Apply replacements in ascending order with running shift to keep spans aligned
                replacements.sort(key=lambda x: x[0])
                running_shift = 0

                for head_idx, tail_idx, cap_idx, crop_token in replacements:
                    adj_head = head_idx + running_shift
                    adj_tail = tail_idx + running_shift
                    image_features_recover = self._merge(
                        image_features_tiles,
                        aspect_ratios[batch_idx][0],
                        aspect_ratios[batch_idx][1],
                    )

                    feat_h, feat_w = image_features_recover.shape[2:]

                    x1, y1, x2, y2 = bboxes[batch_idx][str(crop_token)]

                    orig_h, orig_w = feat_h * 16 * 2, feat_w * 16 * 2

                    roi_orig_x1 = x1 * orig_w
                    roi_orig_y1 = y1 * orig_h
                    roi_orig_x2 = x2 * orig_w
                    roi_orig_y2 = y2 * orig_h

                    spatial_scale = feat_w / orig_w
                    roi_feat_x1 = roi_orig_x1 * spatial_scale
                    roi_feat_y1 = roi_orig_y1 * spatial_scale
                    roi_feat_x2 = roi_orig_x2 * spatial_scale
                    roi_feat_y2 = roi_orig_y2 * spatial_scale

                    roi = torch.tensor(
                        [0, roi_feat_x1, roi_feat_y1, roi_feat_x2, roi_feat_y2],
                        dtype=torch.float32,
                        device=image_features_recover.device,
                    )

                    if self.roi_output_size is None:
                        output_h, output_w = feat_h, feat_w
                    elif isinstance(self.roi_output_size, int):
                        output_h = output_w = self.roi_output_size
                    else:
                        output_h, output_w = self.roi_output_size

                    roi_features = torchvision.ops.roi_align(
                        input=image_features_recover.float(),
                        boxes=roi.unsqueeze(0),
                        output_size=(output_h, output_w),
                        spatial_scale=spatial_scale,
                        sampling_ratio=2,
                        aligned=True,
                    )

                    image_features_replay = (
                        roi_features.permute(0, 2, 3, 1)
                        .flatten(1, 2)
                        .to(image_features_recover.dtype)
                        .squeeze()
                    )

                    curr_inputs_embeds = torch.cat(
                        [
                            curr_inputs_embeds[:adj_head],
                            image_features_replay,
                            curr_inputs_embeds[adj_tail + 1 :],
                        ]
                    )
                    curr_input_ids = torch.cat(
                        [
                            curr_input_ids[:adj_head],
                            torch.full(
                                (image_features_replay.shape[0],),
                                crop_token,
                                dtype=torch.long,
                                device=curr_input_ids.device,
                            ),
                            curr_input_ids[adj_tail + 1 :],
                        ]
                    )
                    reserved_token_spans[batch_idx].append(
                        (cap_idx, adj_head, adj_head + image_features_replay.shape[0])
                    )

                    length_changed = True

                    delta = image_features_replay.shape[0] - (tail_idx - head_idx + 1)
                    running_shift += delta

                if reserved_token_spans[batch_idx]:
                    reserved_token_spans[batch_idx].sort(key=lambda x: x[1])

                new_inputs_embeds.append(curr_inputs_embeds.unsqueeze(0))
                new_input_ids_list.append(curr_input_ids.unsqueeze(0))

            inputs_embeds = torch.cat(new_inputs_embeds, dim=0)
            input_ids = torch.cat(new_input_ids_list, dim=0)

        if (
            length_changed
            or attention_mask is None
            or attention_mask.shape[1] != inputs_embeds.shape[1]
            or position_ids is None
            or position_ids.shape[1] != inputs_embeds.shape[1]
        ):
            attention_mask = torch.ones(
                inputs_embeds.shape[0],
                inputs_embeds.shape[1],
                dtype=torch.long,
                device=inputs_embeds.device,
            )
            position_ids = (
                torch.arange(
                    0,
                    inputs_embeds.shape[1],
                    dtype=torch.long,
                    device=inputs_embeds.device,
                )
                .unsqueeze(0)
                .repeat(inputs_embeds.shape[0], 1)
            )

        return inputs_embeds, attention_mask, position_ids, input_ids, reserved_token_spans

    def pixel_shuffle(self, x, scale_factor=0.5):
        x = x.contiguous()
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))
        x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def _merge(self, tiles: torch.Tensor, ncw: int, nch: int) -> torch.Tensor:
        """Merge image tiles back to original spatial layout."""
        batch_size, num_tiles, num_channels, tile_height, tile_width = tiles.size()
        assert num_tiles == ncw * nch, f"{ncw * nch} != {num_tiles}"

        tiles = tiles.view(batch_size, nch, ncw, num_channels, tile_height, tile_width)
        tiles = tiles.permute(0, 3, 1, 4, 2, 5).contiguous()

        original_height = nch * tile_height
        original_width = ncw * tile_width

        image = tiles.view(batch_size, num_channels, original_height, original_width)

        return image

    def _build_custom_4d_mask(
        self,
        input_ids: torch.Tensor,
        attention_mask_2d: torch.Tensor,
        tokenizer,
        dtype: torch.dtype,
        reserved_token_spans: Optional[List[List[tuple]]] = None,
    ) -> Optional[torch.Tensor]:
        """Construct a 4D attention mask so each Mask_Cap_i block only attends to itself,
        image tokens, and its corresponding reserved token embeddings.

        Args:
            input_ids: (B, L)
            attention_mask_2d: (B, L) padding mask
            tokenizer: tokenizer with convert_tokens_to_ids
            dtype: target dtype for the mask (match hidden states)
            reserved_token_spans: optional per-batch list of (idx, start, end) spans that
                replaced <|reserved_token_i|>. End is exclusive.
        Returns:
            mask_4d: (B, 1, L, L) or None if tokenizer is missing
        """
        if tokenizer is None:
            return None

        device = input_ids.device
        batch_size, seq_len = input_ids.shape
        neg_value = torch.finfo(dtype).min

        image_token_id = getattr(self.config, "image_token_id", None)
        image_positions = input_ids.eq(image_token_id) if image_token_id is not None else None

        eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")

        # Precompute Mask_Cap and reserved token ids
        mask_cap_ids = []
        reserved_token_ids = []
        for i in range(self.prompt_numbers):
            mask_cap_ids.append((i, tokenizer.convert_tokens_to_ids(f"<|Mask_Cap_{i}|>")))
            reserved_token_ids.append(tokenizer.convert_tokens_to_ids(f"<|reserved_token_{i}|>"))

        mask_4d = torch.zeros((batch_size, 1, seq_len, seq_len), device=device, dtype=dtype)

        for b in range(batch_size):
            seq = input_ids[b]
            valid_positions = attention_mask_2d[b].bool()
            valid_indices = torch.nonzero(valid_positions, as_tuple=False).flatten().tolist()
            img_idx = (
                torch.nonzero(image_positions[b], as_tuple=False).flatten().tolist()
                if image_positions is not None
                else []
            )

            for cap_idx, cap_token_id in mask_cap_ids:
                if cap_token_id is None or cap_token_id < 0:
                    continue
                cap_locs = torch.nonzero(seq == cap_token_id, as_tuple=False).flatten()
                if cap_locs.numel() == 0:
                    continue
                start = cap_locs[0].item()

                # Determine the end boundary: next mask_cap or last token in the sentence.
                # NOTE: <|eot_id|> is NOT used as boundary because it now serves as
                # padding within each caption block after the caption-padding change.
                end_candidates = []
                for later_idx, later_token_id in mask_cap_ids:
                    if later_idx <= cap_idx:
                        continue
                    later_pos = torch.nonzero(seq == later_token_id, as_tuple=False).flatten()
                    if later_pos.numel() > 0:
                        end_candidates.append(later_pos[0].item())
                end = min(end_candidates) if len(end_candidates) > 0 else seq_len

                group_tokens = [i for i in range(start, end) if valid_positions[i]]
                if len(group_tokens) == 0:
                    continue

                # Collect reserved token spans for this caption block
                allowed_reserved_positions: List[int] = []
                if reserved_token_spans is not None and len(reserved_token_spans) > b:
                    for idx, span_start, span_end in reserved_token_spans[b]:
                        if idx == cap_idx:
                            allowed_reserved_positions.extend(range(span_start, min(span_end, seq_len)))

                # Fallback to original reserved token id if no recorded span
                if len(allowed_reserved_positions) == 0:
                    reserved_id = reserved_token_ids[cap_idx]
                    if reserved_id is not None and reserved_id >= 0:
                        allowed_reserved_positions.extend(
                            torch.nonzero(seq == reserved_id, as_tuple=False).flatten().tolist()
                        )
                fix_prompt_positions = torch.nonzero(
                    seq == tokenizer.convert_tokens_to_ids('<|reserved_token_0|>'),
                    as_tuple=False,
                ).flatten()
                fix_prompt_len = fix_prompt_positions[0].item() if fix_prompt_positions.numel() > 0 else 0
                # Use the latest recorded reserved span (after sorting) when available
                last_span_end = (
                    reserved_token_spans[b][-1][2]
                    if reserved_token_spans is not None
                    and len(reserved_token_spans) > b
                    and len(reserved_token_spans[b]) > 0
                    else fix_prompt_len
                )
                mask_cap_0_position = torch.nonzero(
                    seq == tokenizer.convert_tokens_to_ids('<|Mask_Cap_0|>'),
                    as_tuple=False,
                ).flatten().tolist()
                fix_prompt_idx = torch.arange(fix_prompt_len, device=device).tolist() + list(range(last_span_end, mask_cap_0_position[0]))
                allowed_targets = set(group_tokens) | set(fix_prompt_idx) | set(allowed_reserved_positions)
                disallowed = set(valid_indices) - allowed_targets
                if len(disallowed) == 0:
                    continue
                disallowed_tensor = torch.tensor(list(disallowed), device=device)
                for q in group_tokens:
                    mask_4d[b, 0, q, disallowed_tensor] = neg_value

            # Optionally mask out padding for all queries (consistency)
            if len(valid_indices) < seq_len:
                invalid = torch.nonzero(~valid_positions, as_tuple=False).flatten()
                if invalid.numel() > 0:
                    mask_4d[b, 0, :, invalid] = neg_value

        return mask_4d

    def forward(self,
                input_ids: torch.LongTensor = None,
                attention_mask: Optional[torch.BoolTensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                pixel_values: Optional[torch.Tensor] = None,
                global_mask_values_list: Optional[List[torch.Tensor]] = None,
                aspect_ratios: Optional[List] = None,
                bboxes: Optional[List] = None,
                prompt_tokens: Optional[List] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                labels: Optional[torch.LongTensor] = None,
                return_dict: bool = True,
                **kwargs,
                ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # ========Get visual embedding========
        if pixel_values is not None:
            vision_embeds = self.forward_vision(pixel_values, global_mask_values_list=global_mask_values_list, prompt_tokens=prompt_tokens)
        else:
            vision_embeds = None

        # ========Prepare inputs for LM========
        # print(f"input_ids.shape: {input_ids.shape}", {vision_embeds.shape})
        inputs_embeds = self.prepare_for_lm(input_ids, vision_embeds)
        # print(f"inputs_embeds.shape: {inputs_embeds.shape}")
        p_mask = None
        answer_length = None
        reserved_token_spans = [[] for _ in range(input_ids.shape[0])]

        # ========Feature Replay (from grasp_any_region)========
        if vision_embeds is not None and aspect_ratios is not None and bboxes is not None:
            # Get crop tokens from reserved special tokens
            crop_tokens = [
                self.processor.tokenizer.convert_tokens_to_ids(
                    f"<|reserved_token_{pid}|>"
                )
                for pid in range(self.prompt_numbers)
            ]
            
            # Reshape vision_embeds to tiles format for feature replay
            # Assuming vision_embeds shape: (num_tiles, num_tokens, hidden_dim)
            # Need to convert to (batch, num_tiles, channels, h, w) format
            patch_num = self.image_size // self.patch_size
            if self.downsample_ratio != 1:
                feat_h = int(patch_num * self.downsample_ratio)
                feat_w = int(patch_num * self.downsample_ratio)
            else:
                feat_h = patch_num
                feat_w = patch_num
            
            # Reshape vision_embeds: (num_tiles, num_tokens, hidden_dim) -> (1, num_tiles, hidden_dim, h, w)
            if vision_embeds.shape[0] != 1:
                image_features_tiles = rearrange(
                    vision_embeds[1:].unsqueeze(0), "b n (h w) c -> b n c h w", h=feat_h, w=feat_w
                )
            else:
                image_features_tiles = rearrange(
                    vision_embeds.unsqueeze(0), "b n (h w) c -> b n c h w", h=feat_h, w=feat_w
                )

            
            new_inputs_embeds = []
            new_input_ids_list = []
            new_labels = [] if labels is not None else None
            length_changed = False
            assert inputs_embeds.shape[0] == 1, "Currently only support batch_size=1"

            for batch_idx in range(inputs_embeds.shape[0]):
                curr_inputs_embeds = inputs_embeds[batch_idx]
                curr_input_ids = input_ids[batch_idx]
                curr_labels = labels[batch_idx] if labels is not None else None
                # Collect all replacements first to avoid index shifting during insertion
                orig_input_ids = input_ids[batch_idx]
                replacements = []
                for cap_idx, crop_token in enumerate(crop_tokens):
                    target_mask = orig_input_ids.eq(crop_token)
                    if not target_mask.any():
                        continue
                    target_indices = target_mask.nonzero().squeeze()
                    if target_indices.ndim == 0:
                        head_idx = tail_idx = target_indices.item()
                    else:
                        head_idx = target_indices.min().item()
                        tail_idx = target_indices.max().item()
                    replacements.append((head_idx, tail_idx, cap_idx, crop_token))
                # Apply replacements in ascending order with running shift to keep spans aligned
                replacements.sort(key=lambda x: x[0])
                running_shift = 0

                for head_idx, tail_idx, cap_idx, crop_token in replacements:
                    adj_head = head_idx + running_shift
                    adj_tail = tail_idx + running_shift
                        
                    # Merge tiles back to original spatial layout
                    image_features_recover = self._merge(
                        image_features_tiles,
                        aspect_ratios[batch_idx][0],
                        aspect_ratios[batch_idx][1],
                    )
                    feat_h, feat_w = image_features_recover.shape[2:]
                    
                    # Get bbox coordinates
                    x1, y1, x2, y2 = bboxes[batch_idx][str(crop_token)]
                    
                    # RoI-Align
                    orig_h, orig_w = feat_h * 28, feat_w * 28  # Original image size
                    
                    # Origin box
                    roi_orig_x1 = x1 * orig_w
                    roi_orig_y1 = y1 * orig_h
                    roi_orig_x2 = x2 * orig_w
                    roi_orig_y2 = y2 * orig_h
                    
                    # Feature box
                    spatial_scale = feat_w / orig_w
                    roi_feat_x1 = roi_orig_x1 * spatial_scale
                    roi_feat_y1 = roi_orig_y1 * spatial_scale
                    roi_feat_x2 = roi_orig_x2 * spatial_scale
                    roi_feat_y2 = roi_orig_y2 * spatial_scale
                    
                    roi = torch.tensor(
                        [0, roi_feat_x1, roi_feat_y1, roi_feat_x2, roi_feat_y2],
                        dtype=torch.float32,
                        device=image_features_recover.device,
                    )
                    
                    # output_size controls how many tokens are inserted (output_h * output_w)
                    if self.roi_output_size is None:
                        output_h, output_w = feat_h, feat_w
                    elif isinstance(self.roi_output_size, int):
                        output_h = output_w = self.roi_output_size
                    else:
                        output_h, output_w = self.roi_output_size

                    roi_features = torchvision.ops.roi_align(
                        input=image_features_recover.float(),
                        boxes=roi.unsqueeze(0),
                        output_size=(output_h, output_w),
                        spatial_scale=spatial_scale,
                        sampling_ratio=2,
                        aligned=True,
                    )

                    image_features_replay = (
                        roi_features.permute(0, 2, 3, 1)
                        .flatten(1, 2)
                        .to(image_features_recover.dtype)
                        .squeeze()
                    )

                    # Replace crop token embeddings with RoI features
                    curr_inputs_embeds = torch.cat(
                        [
                            curr_inputs_embeds[:adj_head],
                            image_features_replay,
                            curr_inputs_embeds[adj_tail + 1 :],
                        ]
                    )
                    curr_input_ids = torch.cat(
                        [
                            curr_input_ids[:adj_head],
                            torch.full(
                                (image_features_replay.shape[0],),
                                crop_token,
                                dtype=torch.long,
                                device=input_ids.device,
                            ),
                            curr_input_ids[adj_tail + 1 :],
                        ]
                    )
                    reserved_token_spans[batch_idx].append(
                        (cap_idx, adj_head, adj_head + image_features_replay.shape[0])
                    )

                    if curr_labels is not None:
                        curr_labels = torch.cat(
                            [
                                curr_labels[:adj_head],
                                -100 * torch.ones(
                                    image_features_replay.shape[0],
                                    dtype=torch.long,
                                    device=labels.device,
                                ),
                                curr_labels[adj_tail + 1 :],
                            ]
                        )

                    assert (
                        curr_labels is None or curr_inputs_embeds.shape[0] == curr_labels.shape[0]
                    ), f"shape mismatch, got {curr_inputs_embeds.shape[0]} != {curr_labels.shape[0]}"
                
                    length_changed = True

                    # Track shift caused by this replacement for subsequent insertions
                    delta = image_features_replay.shape[0] - (tail_idx - head_idx + 1)
                    running_shift += delta

                # Keep spans ordered by start so downstream masking reads consistent positions
                if reserved_token_spans[batch_idx]:
                    reserved_token_spans[batch_idx].sort(key=lambda x: x[1])
                
                new_inputs_embeds.append(curr_inputs_embeds.unsqueeze(0))
                new_input_ids_list.append(curr_input_ids.unsqueeze(0))
                if new_labels is not None:
                    new_labels.append(curr_labels)
            
            inputs_embeds = torch.cat(new_inputs_embeds, dim=0)
            input_ids = torch.cat(new_input_ids_list, dim=0)
            if new_labels is not None:
                labels = torch.cat(new_labels, dim=0)

            if (
                length_changed
                or attention_mask is None
                or attention_mask.shape[1] != inputs_embeds.shape[1]
                or position_ids is None
                or position_ids.shape[1] != inputs_embeds.shape[1]
            ):
                attention_mask = torch.ones(
                    inputs_embeds.shape[0],
                    inputs_embeds.shape[1],
                    dtype=torch.long,
                    device=inputs_embeds.device,
                )
                position_ids = (
                    torch.arange(
                        0,
                        inputs_embeds.shape[1],
                        dtype=torch.long,
                        device=inputs_embeds.device,
                    )
                    .unsqueeze(0)
                    .repeat(inputs_embeds.shape[0], 1)
                )

        if attention_mask is None:
            attention_mask = torch.ones(
                inputs_embeds.shape[0],
                inputs_embeds.shape[1],
                dtype=torch.long,
                device=inputs_embeds.device,
            )
        if position_ids is None:
            position_ids = (
                torch.arange(
                    0,
                    inputs_embeds.shape[1],
                    dtype=torch.long,
                    device=inputs_embeds.device,
                )
                .unsqueeze(0)
                .repeat(inputs_embeds.shape[0], 1)
            )

        tokenizer_for_mask = kwargs.pop("tokenizer", None)
        if tokenizer_for_mask is None and hasattr(self, "processor") and hasattr(self.processor, "tokenizer"):
            tokenizer_for_mask = self.processor.tokenizer

        custom_mask = self._build_custom_4d_mask(
            input_ids=input_ids,
            attention_mask_2d=attention_mask,
            tokenizer=tokenizer_for_mask,
            dtype=inputs_embeds.dtype,
            reserved_token_spans=reserved_token_spans,
        )
        if custom_mask is not None:
            attention_mask = custom_mask

        if self.is_gradient_checkpointing and torch.is_grad_enabled():
            inputs_embeds.requires_grad_(True)

        # Normalize label shape to (batch, seq_len) to match logits masking in language model
        if labels is not None and labels.dim() == 1:
            expected_tokens = inputs_embeds.shape[0] * inputs_embeds.shape[1]
            if labels.numel() == expected_tokens:
                labels = labels.view(inputs_embeds.shape[0], inputs_embeds.shape[1])

        # ========Forward into LM========
        outputs = self.language_model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            return_dict=return_dict,
            labels=labels,
            use_cache=False,
            conversation_ids=None,
            replacement_noise_mode=self.replacement_noise_mode,
            p_mask = p_mask,
            answer_length = answer_length,
            **kwargs,
        )

        return outputs

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        super().gradient_checkpointing_enable(gradient_checkpointing_kwargs)
        self.language_model.gradient_checkpointing_enable()
        self.language_model.enable_input_require_grads()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    def get_decoder(self):
        return self.language_model.get_decoder()

    def tie_weights(self):
        return self.language_model.tie_weights()

    @torch.no_grad()
    def generate(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.FloatTensor] = None,
            global_mask_values_list: Optional[torch.FloatTensor] = None,
            aspect_ratios: Optional[List] = None,
            bboxes: Optional[List] = None,
            prompt_tokens: Optional[List] = None,
            tokenizer=None,
            **generate_kwargs,
    ) -> torch.LongTensor:
        inputs_embeds, attention_mask, position_ids, input_ids, reserved_token_spans = self._prepare_inputs_for_generation(
            input_ids=input_ids,
            pixel_values=pixel_values,
            global_mask_values_list=global_mask_values_list,
            aspect_ratios=aspect_ratios,
            bboxes=bboxes,
            prompt_tokens=prompt_tokens,
            tokenizer=tokenizer,
        )

        tokenizer_for_mask = tokenizer
        if tokenizer_for_mask is None and hasattr(self, "processor") and hasattr(self.processor, "tokenizer"):
            tokenizer_for_mask = self.processor.tokenizer

        custom_mask = self._build_custom_4d_mask(
            input_ids=input_ids,
            attention_mask_2d=attention_mask,
            tokenizer=tokenizer_for_mask,
            dtype=inputs_embeds.dtype,
            reserved_token_spans=reserved_token_spans,
        )
        if custom_mask is not None:
            attention_mask = custom_mask
        if 'llada' in self.config.language_model_config.name_or_path.lower():
            outputs = self.language_model.generate_with_embeds_nonblock(
                inputs_embeds=inputs_embeds,
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generate_kwargs,
            )
        return outputs

    @torch.no_grad()
    def generate_replace_noise(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.FloatTensor] = None,
            global_mask_values_list: Optional[torch.FloatTensor] = None,
            aspect_ratios: Optional[List] = None,
            bboxes: Optional[List] = None,
            prompt_tokens: Optional[List] = None,
            tokenizer=None,
            **generate_kwargs,
    ) -> torch.LongTensor:
        inputs_embeds, attention_mask, position_ids, input_ids, reserved_token_spans = self._prepare_inputs_for_generation(
            input_ids=input_ids,
            pixel_values=pixel_values,
            global_mask_values_list=global_mask_values_list,
            aspect_ratios=aspect_ratios,
            bboxes=bboxes,
            prompt_tokens=prompt_tokens,
            tokenizer=tokenizer,
        )

        tokenizer_for_mask = tokenizer
        if tokenizer_for_mask is None and hasattr(self, "processor") and hasattr(self.processor, "tokenizer"):
            tokenizer_for_mask = self.processor.tokenizer

        custom_mask = self._build_custom_4d_mask(
            input_ids=input_ids,
            attention_mask_2d=attention_mask,
            tokenizer=tokenizer_for_mask,
            dtype=inputs_embeds.dtype,
            reserved_token_spans=reserved_token_spans,
        )
        if custom_mask is not None:
            attention_mask = custom_mask

        outputs, all_steps_response = self.language_model.generate_with_embeds_replace_noise(
            inputs_embeds=inputs_embeds,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generate_kwargs,
        )
        return outputs, all_steps_response

    def get_template(self):
        if 'llada' in self.config.language_model_config.name_or_path.lower():
            template = dict(
                SYSTEM=("<|start_header_id|>system<|end_header_id|>\n{system}<|eot_id|>\n"),
                INSTRUCTION=("<|start_header_id|>user<|end_header_id|>\n{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"),
                SUFFIX="<|eot_id|>",
                SUFFIX_AS_EOS=True,
                SEP="\n",
                STOP_WORDS=["<|eot_id|>"],
            )
        return template

    @torch.no_grad()
    def chat(
            self,
            tokenizer,
            pixel_values,
            question,
            generation_config,
            global_mask_values=None,
            aspect_ratios=None,
            bboxes=None,
            history=None,
            return_history=False,
            num_patches_list=None,
            IMG_START_TOKEN='<img>',
            IMG_END_TOKEN='</img>',
            IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
            verbose=False
    ):

        if history is None and pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question

        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        template = self.get_template()
        eos_token_id = tokenizer.convert_tokens_to_ids(template["SUFFIX"].strip())

        history = "" if history is None else history
        prompt = history
        prompt = prompt + template["INSTRUCTION"].format(input=question)

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        prompt = prompt[::-1]
        for num_patches in num_patches_list[::-1]:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            prompt = prompt.replace('<image>'[::-1], image_tokens[::-1], 1)
        prompt = prompt[::-1]
        model_inputs = tokenizer(prompt, return_tensors='pt')
        device = torch.device(self.language_model.device if torch.cuda.is_available() else 'cpu')
        input_ids = model_inputs['input_ids'].to(device)
        attention_mask = model_inputs['attention_mask'].to(device)
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            global_mask_values=global_mask_values,
            aspect_ratios=aspect_ratios,
            bboxes=bboxes,
            input_ids=input_ids,
            **generation_config
        )
        response = [
            tokenizer.decode(g[len(p) :].tolist())
            for p, g in zip(input_ids, generation_output)
        ][0]
        # response = tokenizer.batch_decode(generation_output, skip_special_tokens=False)[0]
        history = history + prompt + response
        response = response.split(template["SUFFIX"].strip())[0].strip()
        if return_history:
            return response, history
        else:
            if verbose:
                print(response)
            return response
        return

    @torch.no_grad()
    def chat_replace_noise(
            self,
            tokenizer,
            pixel_values,
            question,
            generation_config,
            global_mask_values=None,
            aspect_ratios=None,
            bboxes=None,
            history=None,
            return_history=False,
            num_patches_list=None,
            IMG_START_TOKEN='<img>',
            IMG_END_TOKEN='</img>',
            IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
            verbose=False

    ):
        if history is None and pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question

        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        template = self.get_template()
        eos_token_id = tokenizer.convert_tokens_to_ids(template["SUFFIX"].strip())

        history = "" if history is None else history
        prompt = history
        prompt = prompt + template["INSTRUCTION"].format(input=question)

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        prompt = prompt[::-1]
        for num_patches in num_patches_list[::-1]:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            prompt = prompt.replace('<image>'[::-1], image_tokens[::-1], 1)
        prompt = prompt[::-1]
        model_inputs = tokenizer(prompt, return_tensors='pt')
        device = torch.device(self.language_model.device if torch.cuda.is_available() else 'cpu')
        input_ids = model_inputs['input_ids'].to(device)
        attention_mask = model_inputs['attention_mask'].to(device)
        generation_config['eos_token_id'] = eos_token_id
        generation_output, all_steps_response = self.generate_replace_noise(
            pixel_values=pixel_values,
            global_mask_values=global_mask_values,
            aspect_ratios=aspect_ratios,
            bboxes=bboxes,
            input_ids=input_ids,
            **generation_config
        )
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=False)[0]

        all_steps_response_ = []
        for step_response in all_steps_response:
            step_response = tokenizer.batch_decode(step_response, skip_special_tokens=False)[0]
            all_steps_response_.append(step_response)
        all_steps_response = all_steps_response_
        for i, step_response in enumerate(all_steps_response):
            print(f"Step {i}: {step_response}\n")

        history = history + prompt + response
        response = response.split(template["SUFFIX"].strip())[0].strip()
        if return_history:
            return response, history
        else:
            if verbose:
                print(response)
            return response
        return

AutoConfig.register("pdmllm", PDMLLMConfig)
AutoModel.register(PDMLLMConfig, PDMLLM)