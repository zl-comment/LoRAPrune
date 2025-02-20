import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from pathlib import Path

def convert_pruned_to_hf(
    output_dir: str,
    target_save_path: str,
    base_model_name_or_path: str = None,
    checkpoint_dir: str = "checkpoint-12"
):
    """
    Convert a pruned model (with masks applied) to standard HuggingFace format.
    
    Args:
        output_dir (str): Directory containing the pruned model
        target_save_path (str): Where to save the converted model
        base_model_name_or_path (str): Original base model path, if not provided will use the one from config
        checkpoint_dir (str): Name of the checkpoint directory containing the model files
    """
    # 构建checkpoint路径
    checkpoint_path = os.path.join(output_dir, checkpoint_dir)
    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Checkpoint directory {checkpoint_path} does not exist")
    
    # 加载优化器状态，其中包含了模型状态
    optimizer_path = os.path.join(checkpoint_path, "optimizer.pt")
    if not os.path.exists(optimizer_path):
        raise ValueError(f"Optimizer state file {optimizer_path} does not exist")
    
    optimizer_state = torch.load(optimizer_path, map_location="cpu", weights_only=True)
    
    # 如果base_model_name_or_path未提供，尝试从config获取
    if base_model_name_or_path is None:
        import json
        config_path = os.path.join(checkpoint_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                base_model_name_or_path = config.get("_name_or_path", None)
        
        if base_model_name_or_path is None:
            raise ValueError("base_model_name_or_path must be provided either as argument or in config.json")

    # 加载基础模型
    print(f"Loading base model from {base_model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name_or_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # 创建目标保存目录
    target_save_path = Path(target_save_path)
    target_save_path.mkdir(parents=True, exist_ok=True)
    
    # 从优化器状态中提取模型状态
    if "state" in optimizer_state and isinstance(optimizer_state["state"], dict):
        pruned_state = {}
        for param_group in optimizer_state.get("param_groups", []):
            for param_idx in param_group.get("params", []):
                if param_idx in optimizer_state["state"]:
                    param_state = optimizer_state["state"][param_idx]
                    if "exp_avg" in param_state:  # 这里可能需要根据实际的键名调整
                        pruned_state[f"param_{param_idx}"] = param_state["exp_avg"]
    else:
        raise ValueError("Unexpected optimizer state format")
    
    # 更新模型权重
    model_state = model.state_dict()
    
    # 更新匹配的键
    for key in model_state.keys():
        if key in pruned_state:
            print(f"Updating weights for {key}")
            model_state[key] = pruned_state[key]
    
    # 保存转换后的模型
    print(f"Saving converted model to {target_save_path}")
    model.save_pretrained(target_save_path)
    
    # 复制tokenizer文件
    tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
    tokenizer.save_pretrained(target_save_path)
    
    print("Conversion completed successfully!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert pruned model to HF format")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory containing the pruned model")
    parser.add_argument("--target_save_path", type=str, required=True, help="Where to save the converted model")
    parser.add_argument("--base_model_name_or_path", type=str, help="Original base model path")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoint-12", help="Name of the checkpoint directory containing the model files")
    
    args = parser.parse_args()
    
    convert_pruned_to_hf(
        output_dir=args.output_dir,
        target_save_path=args.target_save_path,
        base_model_name_or_path=args.base_model_name_or_path,
        checkpoint_dir=args.checkpoint_dir
    )
