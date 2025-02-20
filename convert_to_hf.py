import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from pathlib import Path
import json

def convert_pruned_to_hf(
    output_dir: str,
    target_save_path: str,
    base_model_name_or_path: str = None,
    checkpoint_dir: str = "checkpoint-12"
):
    """
    Convert a pruned LoRA model to standard HuggingFace format.
    This handles both the LoRA weights and pruning masks.
    """
    checkpoint_path = os.path.join(output_dir, checkpoint_dir)
    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Checkpoint directory {checkpoint_path} does not exist")

    # 获取base_model路径
    if base_model_name_or_path is None:
        config_path = os.path.join(checkpoint_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                base_model_name_or_path = config.get("_name_or_path", None)
        
        if base_model_name_or_path is None:
            raise ValueError("base_model_name_or_path must be provided")

    print(f"Loading base model from {base_model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name_or_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # 加载优化器状态以获取pruning信息
    optimizer_path = os.path.join(checkpoint_path, "optimizer.pt")
    if os.path.exists(optimizer_path):
        print("Loading optimizer state...")
        optimizer_state = torch.load(optimizer_path, map_location="cpu")
        
        # 获取masks和稀疏度信息
        masks = {}
        sparsity_stats = {}
        
        # 遍历模型的所有Linear层
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                # 检查是否是目标层
                if any(target in name for target in ["gate_proj", "up_proj", "down_proj", "o_proj"]):
                    # 创建结构化剪枝mask
                    mask = torch.ones(module.out_features, dtype=torch.float32)
                    masks[name] = mask
                    
                    # 获取该层的参数ID
                    param_id = id(module.weight)
                    
                    # 从优化器状态获取mask信息
                    if "state" in optimizer_state:
                        for param_group in optimizer_state["param_groups"]:
                            for state_param_id in param_group["params"]:
                                state = optimizer_state["state"].get(state_param_id, {})
                                if "pruning_mask" in state:
                                    # 确保mask形状匹配
                                    state_mask = state["pruning_mask"]
                                    if state_mask.shape == mask.shape:
                                        masks[name] = state_mask
                                        sparsity = (state_mask == 0).float().mean().item()
                                        sparsity_stats[name] = sparsity
                                        print(f"Found pruning mask for {name}, sparsity: {sparsity:.4f}")
                                        break
        
        # 应用masks到模型
        print("\nApplying masks to model...")
        for name, module in model.named_modules():
            if name in masks:
                mask = masks[name]
                if isinstance(module, torch.nn.Linear):
                    # 应用结构化剪枝
                    with torch.no_grad():
                        # 对输出维度进行剪枝
                        module.weight.data[mask == 0] = 0
                        if module.bias is not None:
                            module.bias.data[mask == 0] = 0
                    print(f"Applied mask to {name}")
                
        # 打印稀疏度统计
        print("\nSparsity statistics:")
        for name, sparsity in sparsity_stats.items():
            print(f"{name}: {sparsity:.2f}% sparse")
    
    # 创建目标保存目录
    target_save_path = Path(target_save_path)
    target_save_path.mkdir(parents=True, exist_ok=True)
    
    # 保存转换后的模型
    print(f"Saving converted model to {target_save_path}")
    model.save_pretrained(target_save_path)
    
    # 保存tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
    tokenizer.save_pretrained(target_save_path)
    
    # 保存剪枝信息
    pruning_info = {
        "original_model": base_model_name_or_path,
        "pruning_checkpoint": checkpoint_path,
        "sparsity_stats": sparsity_stats
    }
    
    with open(os.path.join(target_save_path, "pruning_info.json"), "w") as f:
        json.dump(pruning_info, f, indent=2)
    
    # 保存masks供后续使用
    if masks:
        torch.save(masks, os.path.join(target_save_path, "pruning_mask.pt"))
        print("Saved pruning masks")
    
    print("Conversion completed successfully!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert pruned LoRA model to HF format")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory containing the pruned model")
    parser.add_argument("--target_save_path", type=str, required=True, help="Where to save the converted model")
    parser.add_argument("--base_model_name_or_path", type=str, help="Original base model path")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoint-20", help="Name of the checkpoint directory")
    
    args = parser.parse_args()
    
    convert_pruned_to_hf(
        output_dir=args.output_dir,
        target_save_path=args.target_save_path,
        base_model_name_or_path=args.base_model_name_or_path,
        checkpoint_dir=args.checkpoint_dir
    )
