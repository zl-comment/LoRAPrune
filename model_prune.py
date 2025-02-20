import os
import sys
import json
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum

import fire
import torch
import torch.nn as nn
import transformers
import bitsandbytes as bnb
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PruningMetric(Enum):
    MAGNITUDE = "magnitude"
    GRADIENT = "gradient"
    ACTIVATION = "activation"

@dataclass
class PruningConfig:
    """剪枝配置"""
    ratio: float = 0.5  # 剪枝比例
    metric: PruningMetric = PruningMetric.MAGNITUDE  # 剪枝度量标准
    blocksize: int = 1  # 块大小
    progressive: bool = True  # 是否使用渐进式剪枝
    initial_warmup_steps: int = 1000  # 预热步数
    frequency: int = 100  # 剪枝频率
    excluded_layers: List[str] = None  # 排除的层
    min_ratio: float = 0.1  # 最小剪枝比例
    max_ratio: float = 0.5  # 最大剪枝比例

class ModelPruner:
    def __init__(self, model, config: PruningConfig):
        self.model = model
        self.config = config
        self.masks = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.step_count = 0
        self.gradient_dict = {}
        self.activation_dict = {}
        
        # 初始化日志文件
        self.log_file = os.path.join("pruning_logs", "pruning_stats.json")
        os.makedirs("pruning_logs", exist_ok=True)
        self.pruning_stats = []
        
        # 注册钩子来收集梯度和激活值
        self._register_hooks()
        
    def _register_hooks(self):
        """注册钩子以收集梯度和激活值信息"""
        def gradient_hook(name):
            def hook(grad):
                self.gradient_dict[name] = grad.detach().cpu()
            return hook
            
        def activation_hook(name):
            def hook(module, input, output):
                self.activation_dict[name] = output.detach().cpu()
            return hook
            
        if self.config.metric in [PruningMetric.GRADIENT, PruningMetric.ACTIVATION]:
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    param.register_hook(gradient_hook(name))
                    
    def calculate_importance_score(self, name: str, param: torch.Tensor) -> torch.Tensor:
        """计算参数重要性分数
        
        支持多种重要性计算方法：
        - magnitude: 使用参数的绝对值
        - gradient: 使用参数梯度的绝对值
        - activation: 结合参数值和激活值
        """
        if self.config.metric == PruningMetric.MAGNITUDE:
            return torch.abs(param.data)
        elif self.config.metric == PruningMetric.GRADIENT:
            if name in self.gradient_dict:
                return torch.abs(param.data * self.gradient_dict[name])
            return torch.abs(param.data)
        elif self.config.metric == PruningMetric.ACTIVATION:
            if name in self.activation_dict:
                return torch.abs(param.data) * torch.abs(self.activation_dict[name].mean())
            return torch.abs(param.data)
            
    def get_current_ratio(self) -> float:
        """获取当前应该使用的剪枝比例（用于渐进式剪枝）"""
        if not self.config.progressive:
            return self.config.ratio
            
        # 在预热期使用最小剪枝比例
        if self.step_count < self.config.initial_warmup_steps:
            return self.config.min_ratio
            
        # 线性增加剪枝比例
        progress = min(1.0, (self.step_count - self.config.initial_warmup_steps) / 
                      (10000 - self.config.initial_warmup_steps))
        current_ratio = self.config.min_ratio + (self.config.max_ratio - self.config.min_ratio) * progress
        return current_ratio
        
    def prune_layer(self, layer: nn.Module, name: str):
        """对单层进行剪枝"""
        logger.info(f"Pruning layer: {name}")
        
        # 收集该层的可训练参数
        params_to_prune = []
        for param_name, param in layer.named_parameters():
            if param.requires_grad and not any(skip in param_name for skip in 
                ["bias", "norm", "embedding"] + (self.config.excluded_layers or [])):
                logger.info(f"Found prunable parameter: {param_name}")
                params_to_prune.append((param_name, param))
                
        if not params_to_prune:
            logger.info("No prunable parameters found in this layer")
            return
            
        # 计算当前应该使用的剪枝比例
        current_ratio = self.get_current_ratio()
        
        # 计算阈值
        all_scores = []
        for param_name, param in params_to_prune:
            scores = self.calculate_importance_score(f"{name}.{param_name}", param)
            all_scores.append(scores.view(-1))
            
        all_scores = torch.cat(all_scores)
        threshold = torch.quantile(all_scores, current_ratio)
        
        # 应用剪枝
        total_params = 0
        zero_params = 0
        for param_name, param in params_to_prune:
            scores = self.calculate_importance_score(f"{name}.{param_name}", param)
            mask = (scores > threshold).float()
            
            # 如果使用块稀疏化
            if self.config.blocksize > 1:
                orig_shape = mask.shape
                mask = mask.view(-1, self.config.blocksize)
                mask = (mask.mean(1, keepdim=True) > 0.5).expand(-1, self.config.blocksize)
                mask = mask.view(orig_shape)
                
            # 应用mask
            param.data *= mask
            self.masks[f"{name}.{param_name}"] = mask
            
            # 统计
            total_params += param.numel()
            zero_params += (param.data == 0).sum().item()
            
        if total_params > 0:
            sparsity = zero_params / total_params
            logger.info(f"Layer sparsity: {sparsity:.2%}")
            return {"total_params": total_params, "zero_params": zero_params}
        return None
        
    def sequential_pruning(self):
        """逐层进行剪枝"""
        logger.info("Starting sequential pruning...")
        self.step_count += 1
        
        # 如果还在预热期，跳过剪枝
        if self.step_count < self.config.initial_warmup_steps:
            logger.info(f"Still in warmup period ({self.step_count}/{self.config.initial_warmup_steps})")
            return
            
        # 获取所有层
        if hasattr(self.model, 'model'):
            if hasattr(self.model.model, 'model'):
                layers = self.model.model.model.layers
                prefix = "model.model.model.layers"
            else:
                layers = self.model.model.layers
                prefix = "model.model.layers"
        else:
            layers = self.model.layers
            prefix = "layers"
            
        logger.info(f"Found {len(layers)} layers to prune")
        
        # 记录剪枝统计信息
        pruning_stats = {
            "step": self.step_count,
            "current_ratio": self.get_current_ratio(),
            "layers": {}
        }
        
        # 逐层处理
        for i, layer in enumerate(layers):
            logger.info(f"Processing layer {i+1}/{len(layers)}")
            layer_name = f"{prefix}.{i}"
            
            # 进行剪枝并记录统计信息
            stats = self.prune_layer(layer, layer_name)
            if stats:
                pruning_stats["layers"][layer_name] = stats
                
        # 计算并记录总体稀疏度
        total_stats = self.get_sparsity()
        pruning_stats.update(total_stats)
        
        # 保存统计信息
        self.pruning_stats.append(pruning_stats)
        with open(self.log_file, 'w') as f:
            json.dump(self.pruning_stats, f, indent=2)
            
        logger.info(f"Overall model sparsity: {total_stats['overall_sparsity']:.2%}")
        
    def get_sparsity(self) -> Dict:
        """获取模型稀疏度统计信息"""
        total_params = 0
        zero_params = 0
        layer_stats = {}
        
        for name, param in self.model.named_parameters():
            if param.requires_grad and not any(skip in name for skip in 
                ["bias", "norm", "embedding"] + (self.config.excluded_layers or [])):
                total_params += param.numel()
                zero_params += (param.data == 0).sum().item()
                
                # 记录每层的稀疏度
                layer_name = name.split('.')[0]
                if layer_name not in layer_stats:
                    layer_stats[layer_name] = {"total": 0, "zero": 0}
                layer_stats[layer_name]["total"] += param.numel()
                layer_stats[layer_name]["zero"] += (param.data == 0).sum().item()
                
        overall_sparsity = zero_params / total_params if total_params > 0 else 0
        
        return {
            "overall_sparsity": overall_sparsity,
            "total_params": total_params,
            "zero_params": zero_params,
            "layer_stats": {
                name: {"sparsity": stats["zero"] / stats["total"]}
                for name, stats in layer_stats.items()
            }
        }

class PruneTrainer(transformers.Trainer):
    def __init__(self, pruner, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pruner = pruner
        self.current_step = 0
        self.best_eval_loss = float('inf')
        self.no_improve_count = 0
        self.early_stopping_patience = 3
        self.eval_history = []
        
    def get_current_lr(self):
        """获取当前学习率"""
        return self.optimizer.param_groups[0]["lr"] if self.optimizer else self.args.learning_rate
        
    def training_step(self, model, inputs, num_items_in_batch=None):
        """重写训练步骤以添加剪枝和监控"""
        # 计算损失
        loss = super().training_step(model, inputs, num_items_in_batch)
        
        # 定期进行剪枝
        self.current_step += 1
        if self.current_step % self.pruner.config.frequency == 0:
            logger.info(f"\nApplying pruning at step {self.current_step}")
            
            # 在剪枝前评估模型
            pre_prune_metrics = self.evaluate()
            logger.info(f"Metrics before pruning: {pre_prune_metrics}")
            
            # 执行剪枝
            self.pruner.sequential_pruning()
            sparsity = self.pruner.get_sparsity()["overall_sparsity"]
            current_lr = self.get_current_lr()
            self.log({
                "sparsity": sparsity,
                "learning_rate": current_lr
            })
            
            # 在剪枝后评估模型
            post_prune_metrics = self.evaluate()
            logger.info(f"Metrics after pruning: {post_prune_metrics}")
            
            # 记录评估结果
            eval_result = {
                "step": self.current_step,
                "sparsity": sparsity,
                "pre_prune_loss": pre_prune_metrics["eval_loss"],
                "post_prune_loss": post_prune_metrics["eval_loss"],
                "learning_rate": current_lr
            }
            self.eval_history.append(eval_result)
            
            # 检查是否需要调整学习率或早停
            if post_prune_metrics["eval_loss"] < self.best_eval_loss:
                self.best_eval_loss = post_prune_metrics["eval_loss"]
                self.no_improve_count = 0
            else:
                self.no_improve_count += 1
                
            # 如果连续多次没有改善，降低学习率
            if self.no_improve_count >= 2:
                current_lr = self.get_current_lr()
                new_lr = current_lr * 0.5
                logger.info(f"Reducing learning rate from {current_lr} to {new_lr}")
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = new_lr
                self.no_improve_count = 0
            
            # 保存评估历史
            with open("eval_history.json", "w") as f:
                json.dump(self.eval_history, f, indent=2)
            
        return loss
        
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """重写评估方法以添加更多指标"""
        # 获取基础评估结果
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        # 添加稀疏度指标
        sparsity_stats = self.pruner.get_sparsity()
        metrics.update({
            f"{metric_key_prefix}_sparsity": sparsity_stats["overall_sparsity"],
            f"{metric_key_prefix}_zero_params": sparsity_stats["zero_params"],
            f"{metric_key_prefix}_total_params": sparsity_stats["total_params"],
        })
        
        # 添加每层的稀疏度
        for layer_name, layer_stats in sparsity_stats["layer_stats"].items():
            metrics[f"{metric_key_prefix}_layer_{layer_name}_sparsity"] = layer_stats["sparsity"]
        
        return metrics
        
    def _save_checkpoint(self, model, trial, metrics=None):
        """重写检查点保存方法以保存额外信息"""
        # 调用父类的保存方法
        checkpoint_folder = super()._save_checkpoint(model, trial, metrics)
        
        # 保存剪枝配置和统计信息
        if checkpoint_folder is not None:
            pruning_info = {
                "config": self.pruner.config.__dict__,
                "sparsity_stats": self.pruner.get_sparsity(),
                "eval_history": self.eval_history
            }
            
            output_dir = os.path.join(checkpoint_folder, "pruning_info.json")
            with open(output_dir, "w") as f:
                json.dump(pruning_info, f, indent=2)
            
        return checkpoint_folder
        
    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval):
        """重写日志记录方法以添加更多信息"""
        # 调用父类的方法
        metrics = super()._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval)
        
        # 添加额外的训练信息
        if metrics is not None and self.current_step > 0:
            metrics["train_step"] = self.current_step
            metrics["learning_rate"] = self.get_current_lr()
            metrics["sparsity"] = self.pruner.get_sparsity()["overall_sparsity"]
            metrics["grad_norm"] = grad_norm
            
        return metrics

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
    # lora hyperparams for quantized model
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device = int(os.environ.get("LOCAL_RANK") or 0)
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # wandb设置
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project

    # 设置量化配置
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
    )

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float32,
    )

    # 准备模型进行训练
    model = prepare_model_for_kbit_training(model)

    # 添加LoRA适配器
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
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

    # 创建剪枝器
    pruning_config = PruningConfig(
        ratio=ratio,
        metric=PruningMetric(prune_metric),
        blocksize=blocksize,
        frequency=prune_freq,
    )
    pruner = ModelPruner(model, pruning_config)

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
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=False,
            bf16=False,
            logging_steps=10,
            optim="paged_adamw_8bit",  # 使用8bit优化器
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
            gradient_checkpointing=True,
            max_grad_norm=1.0,
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
    final_sparsity = pruner.get_sparsity()["overall_sparsity"]
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
