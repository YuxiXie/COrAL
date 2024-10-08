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

# MODEL_NAME_OR_PATH="meta-math/MetaMath-Mistral-7B"
MODEL_NAME_OR_PATH="path_to_sft"
OUTPUT_DIR=""
unset HOSTFILE
ZERO_STAGE=2
OFFLOAD="none"

mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(cd "${OUTPUT_DIR}" &>/dev/null && pwd)"
if [[ ! -f "${OUTPUT_DIR}/.gitignore" ]]; then
	echo '*' >"${OUTPUT_DIR}/.gitignore"
fi

cp -f "$0" "${OUTPUT_DIR}/script.sh"

export WANDB_MODE=online
export WANDB_API_KEY=""
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

exec 1> >(tee "${OUTPUT_DIR}/stdout.log" >&1) 2> >(tee "${OUTPUT_DIR}/stderr.log" >&2)

gpu_vis=$1

deepspeed --include localhost:$gpu_vis --master_port $MASTER_PORT \
	--module coral.algorithms.oa \
	--train_datasets MetaMath \
	--model_type metamath \
	--model_name_or_path "${MODEL_NAME_OR_PATH}" \
	--no_noise \
	--context_corrupt \
	--multi_context_granularity \
	--max_corrupt_context_size 4 \
	--max_length 512 \
	--context_window 4 \
	--n_back_pred 2 \
	--trust_remote_code True \
	--epochs 3 \
	--replace_ratio_mu 0.0 \
	--replace_ratio_std 0.25 \
	--replace_ratio_max 0.5 \
	--replace_ratio_min 0.0 \
	--per_device_train_batch_size 4 \
	--per_device_eval_batch_size 4 \
	--gradient_accumulation_steps 4 \
	--gradient_checkpointing \
	--learning_rate 5e-6 \
	--tune_final_layer_only \
	--tune_lm_head \
	--lr_scheduler_type cosine \
	--lr_warmup_ratio 0.03 \
	--weight_decay 0.0 \
	--seed 42 \
	--output_dir "${OUTPUT_DIR}" \
	--log_type wandb \
	--log_project COrAL-S1-ARITHMETIC \
	--zero_stage "${ZERO_STAGE}" \
	--offload "${OFFLOAD}" \
	--bf16 True \
	--tf32 True

# --not_lazy_tokenization \