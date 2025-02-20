python prune-v1.py --base_model "D:\ZLCODE\model\Llama-2-7b-chat-hf" --data_path "./data/harmful_behaviors.csv" --output_dir "./output" --batch_size 128 --micro_batch_size 4 --num_epochs 3 --learning_rate 3e-4  #0.01%的剪枝率 性能和原本基本一样对于我的数据集

python prune-v1.py --base_model "D:\ZLCODE\model\Llama-2-7b-chat-hf" --data_path "./data/harmful_behaviors.csv" --output_dir "./output-v2" --batch_size 128 --micro_batch_size 4 --num_epochs 5 --learning_rate 5e-4 --ratio 0.7 --prune_freq 5 --warmup_iters 0.05 --cooldown_iters 0.05 --prune_metric magnitude --lora_r 16 --lora_alpha 32 --lora_dropout 0.1

python prune-v1.py --base_model "D:\ZLCODE\model\Llama-2-7b-chat-hf" --data_path "./data/harmful_behaviors.csv" --output_dir "./output-v3" --batch_size 256 --micro_batch_size 4 --num_epochs 5 --learning_rate 5e-4 --ratio 0.7 --prune_freq 5 --warmup_iters 0.15 --cooldown_iters 0.05 --prune_metric magnitude --lora_r 20 --lora_alpha 32 --lora_dropout 0.1
