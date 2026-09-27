
device=0,1,2,3 # GPUs to be trained on, minimum requirement: 1 gpu

deepspeed --include localhost:$device --module openrlhf.cli.train_sft \
   --save_path /weka/harpa-cs/harpa-rlvr-branch/hypothesis_generation/ASD-specific/harpa_distilled_Sep12_v3 \
   --save_steps 500 \
   --logging_steps 1 \
   --eval_steps -1 \
   --train_batch_size 4 \
   --micro_train_batch_size 1 \
   --pretrain Qwen/Qwen2.5-7B-Instruct \
   --bf16 \
   --max_epochs 1 \
   --max_len 12288 \
   --zero_stage 3 \
   --learning_rate 5e-6 \
   --dataset /weka/harpa-cs/harpa-rlvr-branch/hypothesis_generation/ASD-specific/RM-R1-data/rmr1_sft_split_dataset \
   --apply_chat_template \
   --input_key context_messages \
   --output_key winner \
   --flash_attn \
   --gradient_checkpointing \
   --packing_samples \
   --adam_offload \
   --save_hf_ckpt \
   --use_wandb "true" \
   --wandb_project "RM-R1-harpa"
   #--wandb_run_name Qwen2.5-7b-instruct-distilled \