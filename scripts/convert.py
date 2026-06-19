import os
import json
import argparse
import shutil
from safetensors.torch import load_file, save_file

def convert_config(src_path, dst_path=None):
    if dst_path is None:
        dst_path = "converted_config.json"
        
    print(f"Reading original config from {src_path}...")
    
    with open(src_path, "r", encoding="utf-8") as f:
        config1 = json.load(f)

    config2_template = {
        "architectures": ["LLaDAModelLM"],
        "auto_map": {
            "AutoConfig": "configuration_llada.LLaDAConfig",
            "AutoModelForCausalLM": "modeling_llada.LLaDAModelLM",
            "AutoModel": "modeling_llada.LLaDAModelLM"
        },
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": 128000,
        "eos_token_id": 126081,
        "hidden_act": "silu",
        "hidden_size": 4096,
        "initializer_range": 0.02,
        "intermediate_size": 12288,
        "max_position_embeddings": 16384,
        "model_type": "llada",
        "num_attention_heads": 32,
        "num_hidden_layers": 32,
        "num_key_value_heads": 32,
        "pretraining_tp": 1,
        "rms_norm_eps": 1e-05,
        "rope_scaling": None,
        "rope_theta": 500000.0,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "transformers_version": "4.39.1",
        "use_cache": False,
        "vocab_size": 126464
    }

    mapping = {
        "architectures": "architectures",
        "auto_map": "auto_map",
        "attention_bias": "include_qkv_bias",
        "attention_dropout": "attention_dropout",
        "eos_token_id": "eos_token_id",
        "hidden_act": "activation_type",
        "hidden_size": "d_model",
        "initializer_range": "init_std",
        "intermediate_size": "mlp_hidden_size",
        "max_position_embeddings": "max_sequence_length",
        "model_type": "model_type",
        "num_attention_heads": "n_heads",
        "num_hidden_layers": "n_layers",
        "num_key_value_heads": "n_kv_heads",
        "rms_norm_eps": "rms_norm_eps",
        "rope_theta": "rope_theta",
        "tie_word_embeddings": "weight_tying",
        "use_cache": "use_cache",
        "vocab_size": "vocab_size"
    }

    new_config = {}

    for target_key, default_val in config2_template.items():
        if target_key in mapping:
            src_key = mapping[target_key]
            if src_key in config1:
                val = config1[src_key]
                new_config[target_key] = val
            else:
                new_config[target_key] = default_val
        elif target_key == "torch_dtype":
            precision = config1.get("precision", "amp_bf16")
            if precision == "amp_bf16":
                new_config["torch_dtype"] = "bfloat16"
            elif precision == "amp_fp16":
                new_config["torch_dtype"] = "float16"
            else:
                new_config["torch_dtype"] = "bfloat16"
        else:
            new_config[target_key] = default_val

    with open(dst_path, "w", encoding="utf-8") as f:
        json.dump(new_config, f, indent=4)

def get_hf_key(llada_key):
    if llada_key == "model.transformer.wte.weight":
        return "model.embed_tokens.weight"
    if llada_key == "model.transformer.ln_f.weight":
        return "model.norm.weight"
    if llada_key == "model.transformer.ff_out.weight":
        return "lm_head.weight" 
    if llada_key.startswith("model.transformer.blocks."):
        parts = llada_key.split(".")
        layer_idx = parts[3]
        sub_layer = parts[4]
        if sub_layer == "attn_norm":
            return f"model.layers.{layer_idx}.input_layernorm.weight"
        if sub_layer == "ff_norm":
            return f"model.layers.{layer_idx}.post_attention_layernorm.weight"
        elif sub_layer == "q_proj":
            return f"model.layers.{layer_idx}.self_attn.q_proj.weight"
        elif sub_layer == "k_proj":
            return f"model.layers.{layer_idx}.self_attn.k_proj.weight"
        elif sub_layer == "v_proj":
            return f"model.layers.{layer_idx}.self_attn.v_proj.weight"
        elif sub_layer == "attn_out":
            return f"model.layers.{layer_idx}.self_attn.o_proj.weight"
        elif sub_layer == "ff_proj":
            return f"model.layers.{layer_idx}.mlp.gate_proj.weight"
        elif sub_layer == "up_proj":
            return f"model.layers.{layer_idx}.mlp.up_proj.weight"
        elif sub_layer == "ff_out":
            return f"model.layers.{layer_idx}.mlp.down_proj.weight"
            
    return llada_key
    
def convert_model_keys(model_path, output_path=None):
    if output_path is None:
        output_path = model_path.rstrip("/") + "_converted"
        
    os.makedirs(output_path, exist_ok=True)
    
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        print(f"Error: {index_path} not found.")
        return
        
    print(f"Converting model weights from {model_path} to {output_path}...")
    print("This may take a while. Please wait...")
    
    with open(index_path, "r") as f:
        index_data = json.load(f)
        
    weight_map = index_data.get("weight_map", {})
    files_to_process = set(weight_map.values())
    new_weight_map = {}
    
    # Process each safetensors file
    for filename in sorted(files_to_process):
        file_path = os.path.join(model_path, filename)
        if not os.path.exists(file_path):
            print(f"Warning: {filename} not found, skipping.")
            continue
            
        # Load original tensors and rename keys
        tensors = load_file(file_path)
        new_tensors = {}
        for old_key, tensor in tensors.items():
            new_key = get_hf_key(old_key)
            new_tensors[new_key] = tensor
            # Update the new weight_map
            new_weight_map[new_key] = filename
            
        # Save new safetensors
        out_filepath = os.path.join(output_path, filename)
        save_file(new_tensors, out_filepath)
        
    # Rebuild and write index json
    total_size = 0
    for filename in sorted(files_to_process):
        out_filepath = os.path.join(output_path, filename)
        if os.path.exists(out_filepath):
            total_size += os.path.getsize(out_filepath)
            
    new_index_data = {
        "metadata": {"total_size": total_size},
        "weight_map": new_weight_map
    }
    
    new_index_path = os.path.join(output_path, "model.safetensors.index.json")
    with open(new_index_path, "w", encoding="utf-8") as f:
        json.dump(new_index_data, f, indent=2)

    # Convert config.json instead of direct copy
    orig_config_path = os.path.join(model_path, "config.json")
    new_config_path = os.path.join(output_path, "config.json")
    if os.path.exists(orig_config_path):
        print("Converting config.json...")
        convert_config(orig_config_path, new_config_path)
    else:
        print("Warning: config.json not found in the original directory.")
        
    # Copy other configuration files
    for item in os.listdir(model_path):
        if not item.endswith(".safetensors") and item not in ["model.safetensors.index.json", "config.json"]:
            src = os.path.join(model_path, item)
            dst = os.path.join(output_path, item)
            if os.path.isfile(src):
                shutil.copy2(src, dst)
                
    print(f"Conversion complete. New model saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert safetensors keys and generate a new index.json")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the original model directory")
    parser.add_argument("--output_path", type=str, default=None, help="Path to save the converted model (defaults to model_path + '_converted')")
    
    args = parser.parse_args()
    convert_model_keys(args.model_path, args.output_path)

'''
python convert.py \
--model_path /mnt/bn/strategy-mllm-train/user/wangyuhao/models/LLaDA-8B-Instruct
'''