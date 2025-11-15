
set -x
# export HYDRA_FULL_ERROR=1
ulimit -n 65535




unset ROCR_VISIBLE_DEVICES



# PROJECT_DIR="$(pwd)"
PROJECT_DIR=./verl
CONFIG_PATH="$PROJECT_DIR/recipe/zoomin"

export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
export CUDA_LAUNCH_BLOCKING=1

export RAY_SPILL_DISK_UTILIZATION_THRESHOLD=0.999



ray stop
pkill -9 raylet




CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='zoomin_multiturn_grpo' \
    ray_kwargs.ray_init.num_cpus=128 \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=64 \
    data.max_prompt_length=25600 \
    data.max_response_length=32756 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    data.return_multi_modal_inputs=False \
    actor_rollout_ref.model.path=sft_model_path \
    data.train_files=train_data_path \
    data.val_files=val_data_path \
    data.image_key=images \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=6 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','swanlab'] \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=2 \
    trainer.total_epochs=15 \
    trainer.val_before_train=False \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$PROJECT_DIR/recipe/zoomin/zoomin_tool_config.yaml" \
    trainer.project_name='project_name' \
    trainer.experiment_name='experiment_name' \
    trainer.default_local_dir=save_model_path \
    actor_rollout_ref.rollout.do_sample=True \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode=disable \
    custom_reward_function.path=$PROJECT_DIR/recipe/zoomin/zoomin_reward3.py \
    custom_reward_function.name=zoomin_reward_function \
    +custom_reward_function.reward_kwargs.reward_type=acc_format \
    critic.model.fsdp_config.param_offload=True \
    critic.model.fsdp_config.optimizer_offload=True \
    critic.model.fsdp_config.offload_policy=True \
    reward_model.model.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    +reward_model.model.fsdp_config.mixed_precision=bf16 \
    +critic.model.fsdp_config.mixed_precision=bf16 \
    +actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    +critic.model.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    $@
