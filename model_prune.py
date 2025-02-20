import os
import sys
from typing import List

import fire
import torch
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

class ModelPruner:
    def __init__(self, model):
        self.model = model
        self.masks = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def calculate_importance_score(self, name, param, activations=None):
        """计算参数重要性分数"""
        if activations is not None:
            # 使用参数和激活值计算重要性
            input_acts, output_acts = activations
            importance = torch.abs(param.data) * torch.abs(input_acts.mean(0))
        else:
            # 仅使用参数大小
            importance = torch.abs(param.data)
        return importance
        
    def prune_layer(self, layer, ratio, blocksize=1):
        """对单层进行剪枝"""
        # 收集该层的所有参数
        params_to_prune = []
        for name, param in layer.named_parameters():
            if param.requires_grad:
                params_to_prune.append((name, param))
        
        if not params_to_prune:
            return
            
        # 计算阈值
        all_scores = []
        for name, param in params_to_prune:
            scores = self.calculate_importance_score(name, param)
            all_scores.append(scores.view(-1))
            
        all_scores = torch.cat(all_scores)
        threshold = torch.quantile(all_scores, ratio)
        
        # 应用剪枝
        for name, param in params_to_prune:
            scores = self.calculate_importance_score(name, param)
            mask = (scores > threshold).float()
            
            # 如果使用块稀疏化
            if blocksize > 1:
                # 重塑mask以适应块大小
                orig_shape = mask.shape
                mask = mask.view(-1, blocksize)
                mask = (mask.mean(1, keepdim=True) > 0.5).expand(-1, blocksize)
                mask = mask.view(orig_shape)
            
            param.data *= mask
            self.masks[name] = mask
            
    def sequential_pruning(self, ratio, dataloader=None, blocksize=1):
        """逐层进行剪枝"""
        print("Starting sequential pruning...")
        
        # 暂时禁用模型缓存
        use_cache = self.model.config.use_cache
        self.model.config.use_cache = False
        
        # 获取所有层
        if hasattr(self.model, 'model'):
            layers = self.model.model.layers
        else:
            layers = self.model.layers
            
        # 逐层处理
        for i, layer in enumerate(layers):
            print(f"Processing layer {i+1}/{len(layers)}")
            
            # 将层移到GPU
            layer = layer.to(self.device)
            
            # 进行剪枝
            self.prune_layer(layer, ratio, blocksize)
            
            # 将层移回CPU
            layer = layer.cpu()
            torch.cuda.empty_cache()
            
        # 恢复模型设置
        self.model.config.use_cache = use_cache
        print("Sequential pruning completed.")
        
    def get_sparsity(self):
        """获取模型稀疏度"""
        total_params = 0
        zero_params = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                total_params += param.numel()
                zero_params += (param.data == 0).sum().item()
        return zero_params / total_params if total_params > 0 else 0

class PruneTrainer(transformers.Trainer):
    def __init__(self, pruner, ratio, prune_freq, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pruner = pruner
        self.ratio = ratio
        self.prune_freq = prune_freq
        
    def training_step(self, *args, **kwargs):
        loss = super().training_step(*args, **kwargs)
        
        # 定期进行剪枝
        if self.state.global_step > 0 and self.state.global_step % self.prune_freq == 0:
            self.pruner.sequential_pruning(self.ratio)
            sparsity = self.pruner.get_sparsity()
            self.log({"sparsity": sparsity})
            
        return loss

def train(
    # model/data params
    base_model: str = "",
    data_path: str = "",
    output_dir: str = "pruned_model",
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 256,
    val_set_size: int = 2000,
    train_on_inputs: bool = False,  # if False, masks out inputs in loss
    # pruning hyperparams
    ratio: float = 0.5,        # 最终要剪掉的参数比例
    prune_freq: int = 100,     # 每多少步进行一次剪枝
    prune_metric: str = 'magnitude',  # magnitude|grad
    blocksize: int = 1,        # 块稀疏化的块大小
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    resume_from_checkpoint: str = None,
):
    print(
        f"Pruning original model with params:\n"
        f"base_model: {base_model}\n"
        f"data_path: {data_path}\n"
        f"output_dir: {output_dir}\n"
        f"ratio: {ratio}\n"
        f"prune_freq: {prune_freq}\n"
        f"prune_metric: {prune_metric}\n"
    )
    
    gradient_accumulation_steps = batch_size // micro_batch_size
    
    # 设置设备
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # wandb设置
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project

    # 加载模型和分词器
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map=device_map,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    # 创建剪枝器
    pruner = ModelPruner(model)

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()
        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = generate_prompt(data_point)
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = generate_prompt({**data_point, "target": ""})
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]
        return tokenized_full_prompt

    # 加载数据集
    if data_path.endswith(".json"):
        data = load_dataset("json", data_files=data_path)
    elif data_path.endswith(".csv"):
        data = load_dataset("csv", data_files=data_path)
    else:
        data = load_dataset(data_path)

    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = train_val["test"].shuffle().map(generate_and_tokenize_prompt)
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None

    # 创建训练器
    trainer = PruneTrainer(
        pruner=pruner,
        ratio=ratio,
        prune_freq=prune_freq,
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=200 if val_set_size > 0 else None,
            save_steps=200,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=True,
            ddp_find_unused_parameters=False if ddp else None,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )

    # 训练模型
    if resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    else:
        trainer.train()

    # 保存最终的剪枝后模型
    final_sparsity = pruner.get_sparsity()
    print(f"Final model sparsity: {final_sparsity:.2%}")
    
    # 保存模型和配置
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # 保存剪枝信息
    pruning_info = {
        "sparsity": final_sparsity,
        "ratio": ratio,
        "metric": prune_metric
    }
    import json
    with open(os.path.join(output_dir, "pruning_info.json"), "w") as f:
        json.dump(pruning_info, f, indent=2)

def generate_prompt(data_point):
    return f"""A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed and polite answers to the user's questions.
User: {data_point['goal']}
Assistant: {data_point['target']}
"""

if __name__ == "__main__":
    fire.Fire(train)
