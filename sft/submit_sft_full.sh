#!/bin/bash
export IMAGE=592892253131.dkr.ecr.us-east-1.amazonaws.com/chronos-o1:latest
export NNODES=8
export BASE_MODEL="/fsx/hf/Qwen/Qwen3-32B"
# export BASE_MODEL="/fsx/hf/Qwen/Qwen3-Coder-30B-A3B-Instruct"
export EPOCHS=1
export ALIAS="xiyuanz"
export JOB_NAME=${ALIAS}-mmts-${NNODES}n-full

# sft gsm8k
# export COMMAND="set -e && export RUN_NAME=\${BASE_MODEL#*/}-sft-$(date +%Y%m%d-%H%M%S) && export LOGDIR=/fsx/mmts/\$RUN_NAME && mkdir -p \$LOGDIR && export NCCL_DEBUG=INFO && export NCCL_SOCKET_IFNAME=eth && export FI_EFA_USE_DEVICE_RDMA=1 && export TENSORBOARD_DIR=/fsx/mmts && echo Starting training && python3 /opt/verl/examples/data_preprocess/gsm8k.py --local_dir /opt/data/gsm8k && /usr/local/bin/torchrun --nproc_per_node=8 --nnodes=\${NNODES} --rdzv_conf=timeout=9000 -m verl.trainer.fsdp_sft_trainer data.train_files=/opt/data/gsm8k/train.parquet data.val_files=/opt/data/gsm8k/test.parquet data.prompt_key=extra_info data.response_key=extra_info data.prompt_dict_keys=['question'] data.response_dict_keys=['answer'] data.multiturn.enable=False data.train_batch_size=32 data.max_length=8192 data.micro_batch_size_per_gpu=1 model.partial_pretrain=\${BASE_MODEL} model.enable_gradient_checkpointing=True model.lora_rank=32 model.lora_alpha=16 model.target_modules=all-linear trainer.default_local_dir=\$LOGDIR trainer.project_name=\${ALIAS}-sft trainer.experiment_name=\${RUN_NAME} trainer.total_epochs=\${EPOCHS} trainer.logger=['console','tensorboard'] trainer.default_hdfs_dir=null \${EXTRA_TRAINER_ARGS}"

# sft mmts
export COMMAND="set -e && export RUN_NAME=Qwen3-32B-sft-8nodes && export LOGDIR=/fsx/mmts/\$RUN_NAME && mkdir -p \$LOGDIR && export NCCL_DEBUG=INFO && export NCCL_SOCKET_IFNAME=eth && export FI_EFA_USE_DEVICE_RDMA=1 && export TENSORBOARD_DIR=/fsx/mmts && echo Starting training && /usr/local/bin/torchrun --nproc_per_node=8 --nnodes=\${NNODES} --rdzv_conf=timeout=9000 -m verl.trainer.fsdp_sft_trainer data.train_files=/fsx/s3/chronos-o1/dataset/parquet/train_path.parquet data.val_files=/fsx/s3/chronos-o1/dataset/parquet/test_path.parquet data.prompt_key=extra_info data.response_key=extra_info data.prompt_dict_keys=['question'] data.response_dict_keys=['answer'] data.multiturn.enable=False data.train_batch_size=64 data.max_length=2000 data.micro_batch_size_per_gpu=1 model.partial_pretrain=\${BASE_MODEL} model.enable_gradient_checkpointing=True trainer.default_local_dir=\$LOGDIR trainer.project_name=\${ALIAS}-sft trainer.experiment_name=\${RUN_NAME} trainer.total_epochs=\${EPOCHS} trainer.logger=['console','tensorboard'] trainer.default_hdfs_dir=null \${EXTRA_TRAINER_ARGS}"

usage() {
    echo "Usage: \$0 [-a|-d]"
    echo "  -d: Delete the configuration"
    exit 1
}

action="apply"

while getopts "d" opt; do
    case $opt in
        d)
        echo "DELETE"
        action="delete"
        ;;
        \?)
        usage
        ;;
    esac
done

cd "$(dirname "${BASH_SOURCE[0]}")"

envsubst < sft.yaml | kubectl $action --validate=false -f -
