python -m verl.model_merger merge \
 --backend fsdp \
 --local_dir checkpoints/global_step_10/actor \
 --target_dir checkpoints/global_step_10/hf