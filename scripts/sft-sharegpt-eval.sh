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

# MODEL_NAME_OR_PATH="huggyllama/llama-7b"
MODEL_NAME_OR_PATH="mistralai/Mistral-7B-v0.3"
# MODEL_NAME_OR_PATH="mistralai/Mistral-7B-Instruct-v0.2"
# MODEL_NAME_OR_PATH="meta-math/MetaMath-Mistral-7B"
# MODEL_NAME_OR_PATH="/share/edc/home/yuxi_xie/oa_dag/checkpoints/v0705/oa-denoise-mu0.25to0.75-dymin-rmu1.0-r0.15/checkpoint-3219"
OUTPUT_DIR="/share/edc/home/yuxi_xie/oa_dag/checkpoints/v0712/metamath-mistral-mu0.2to2.0-dymin-rmu1.0-r0.55"
unset HOSTFILE
ZERO_STAGE=3
OFFLOAD="optimizer"

mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(cd "${OUTPUT_DIR}" &>/dev/null && pwd)"
if [[ ! -f "${OUTPUT_DIR}/.gitignore" ]]; then
	echo '*' >"${OUTPUT_DIR}/.gitignore"
fi

cp -f "$0" "${OUTPUT_DIR}/script.sh"

export WANDB_MODE=dryrun
export WANDB_API_KEY="1396a7d2a29a8e8241dff6e0e6371f2ad61e11e2"
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

gpu_vis=2

deepspeed --include localhost:$gpu_vis --master_port $MASTER_PORT \
	--module oa_dag.algorithms.oa \
	--train_datasets MetaMath \
	--model_type metamath \
	--model_name_or_path "${MODEL_NAME_OR_PATH}" \
	--max_length 1024 \
	--trust_remote_code True \
	--epochs 2 \
	--save_interval 10240 \
	--tune_final_layer_only \
	--dynamic_mask_ratio_mu \
	--mask_ratio_mu 0.95 \
	--max_mask_ratio_mu 2.0 \
	--min_mask_ratio_mu 0.2 \
	--reconstruct \
	--replace_ratio_mu 0.55 \
	--replace_ratio_std 0.5 \
	--replace_ratio_max 1.0 \
	--replace_ratio_min 0.0 \
	--replace_with_prob 0.5 \
	--per_device_train_batch_size 2 \
	--per_device_eval_batch_size 4 \
	--gradient_accumulation_steps 8 \
	--gradient_checkpointing \
	--learning_rate 2e-5 \
	--lr_scheduler_type cosine \
	--lr_warmup_ratio 0.03 \
	--weight_decay 0.0 \
	--seed 42 \
	--output_dir "${OUTPUT_DIR}" \
	--log_type wandb \
	--log_project OA-TEST \
	--zero_stage "${ZERO_STAGE}" \
	--offload "${OFFLOAD}" \
	--bf16 True \
	--tf32 True

# --exclude_l2r_order \
# --tune_final_layer_only \
# --replace_ratio_min 0.05 \
# --replace_ratio_max 0.95 \
# --replace_ratio_std 0.45 \
# --not_lazy_tokenization \

# bash scripts/sft-sharegpt-math.sh
