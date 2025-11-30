set -x

export RAY_DEDUP_LOGS=0

export VLLM_ATTENTION_BACKEND=XFORMERS

export http_proxy=127.0.0.1:7890 https_proxy=127.0.0.1:7890

math_train_path=./mini_data/math/train.parquet
math_test_path=./mini_data/math/test.parquet
aime2025_test_path=./mini_data/aime2025/test.parquet
amc23_test_path=./mini_data/amc23/test.parquet

train_files="['$math_train_path']"
test_files="['$math_test_path', '$aime2025_test_path', '$amc23_test_path']"
# advantage="positive"   # PSR
# advantage="negative"   # NSR
advantage="weighted"   # W-REINFORCE
positive_advantage_weight=0.1   # For W-REINFORCE only
kl_coef=0.0
lr=1e-6
model_name=/root/autodl-tmp/Models/Qwen2.5-0.5B-Instruct

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=psr_nsr \
    algorithm.advantage=$advantage \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=8 \
    data.max_prompt_length=1024 \
    data.max_response_length=3072 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4800 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=6400 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=6400 \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    trainer.experiment_name="small_test_1130" \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb'] \
    trainer.project_name='rlvr_reasoning_exps' \
    trainer.n_gpus_per_node=2 \
    +trainer.val_before_train=True \
    trainer.nnodes=1 \
    trainer.save_freq=7 \
    trainer.test_freq=7 \
    trainer.total_epochs=20 $@
    # algorithm.positive_advantage_weight=$positive_advantage_weight \
