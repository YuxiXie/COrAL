#!/usr/bin/env bash

if [ -z "${BASH_VERSION}" ]; then
	echo "Please use bash to run this script." >&2
	exit 1
fi

set -x

SCRIPT_DIR="$(cd "$(dirname "$0")" &>/dev/null && pwd)"
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export LOGLEVEL="${LOGLEVEL:-WARNING}"

MODEL_NAME_OR_PATH="mistralai/Mistral-7B-v0.3"
OUTPUT_DIR="/share/edc/home/yuxi_xie/oa_dag/checkpoints/dev"
unset HOSTFILE
ZERO_STAGE=3
OFFLOAD="optimizer"
WANDB_API_KEY="1396a7d2a29a8e8241dff6e0e6371f2ad61e11e2"
WANDB_MODE=online

mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(cd "${OUTPUT_DIR}" &>/dev/null && pwd)"
if [[ ! -f "${OUTPUT_DIR}/.gitignore" ]]; then
	echo '*' >"${OUTPUT_DIR}/.gitignore"
fi

cp -f "$0" "${OUTPUT_DIR}/script.sh"

if [[ -z "${WANDB_API_KEY}" ]]; then
	export WANDB_MODE="offline"
fi

MASTER_PORT_START=10000
MASTER_PORT_END=65535
MASTER_PORT="$(
	comm -23 \
		<(seq "${MASTER_PORT_START}" "${MASTER_PORT_END}" | sort) \
		<(ss -Htan | awk '{ print $4 }' | awk -F ':' '{ print $NF }' | sort -u) |
		shuf | head -n 1
)"

DEEPSPEED_ARGS=()
if [[ -n "${HOSTFILE+x}" ]]; then
	DEEPSPEED_ARGS+=("--hostfile" "${HOSTFILE}")
fi
DEEPSPEED_ARGS+=("--master_port" "${MASTER_PORT}")

exec 1> >(tee "${OUTPUT_DIR}/stdout.log" >&1) 2> >(tee "${OUTPUT_DIR}/stderr.log" >&2)

gpu_vis=7

deepspeed --include localhost:$gpu_vis --master_port $MASTER_PORT \
	--module oa_dag.values.reward \
	--train_datasets PKU-SafeRLHF/train \
	--eval_datasets PKU-SafeRLHF/test \
	--model_name_or_path "${MODEL_NAME_OR_PATH}" \
	--max_length 512 \
	--trust_remote_code True \
	--loss_type sequence-wise \
	--epochs 2 \
	--per_device_train_batch_size 16 \
	--per_device_eval_batch_size 16 \
	--gradient_accumulation_steps 1 \
	--gradient_checkpointing \
	--regularization 0.001 \
	--normalize_score_during_training False \
	--normalizer_type ExponentialMovingAverage \
	--normalizer_momentum 0.9 \
	--learning_rate 2e-5 \
	--lr_scheduler_type cosine \
	--lr_warmup_ratio 0.03 \
	--weight_decay 0.1 \
	--seed 42 \
	--need_eval \
	--eval_strategy epoch \
	--output_dir "${OUTPUT_DIR}" \
	--log_type wandb \
	--log_project Safe-RLHF-RM \
	--zero_stage "${ZERO_STAGE}" \
	--offload "${OFFLOAD}" \
	--bf16 True \
	--tf32 True
