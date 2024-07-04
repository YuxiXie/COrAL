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
OUTPUT_DIR="/share/edc/home/yuxi_xie/oa_dag/checkpoints/dev"
unset HOSTFILE
ZERO_STAGE=3
OFFLOAD="optimizer"
while [[ "$#" -gt 0 ]]; do
	arg="$1"
	shift
	case "${arg}" in
		--model_name_or_path)
			MODEL_NAME_OR_PATH="$1"
			shift
			;;
		--model_name_or_path=*)
			MODEL_NAME_OR_PATH="${arg#*=}"
			;;
		--output_dir)
			OUTPUT_DIR="$1"
			shift
			;;
		--output_dir=*)
			OUTPUT_DIR="${arg#*=}"
			;;
		--hostfile)
			HOSTFILE="$1"
			shift
			;;
		--hostfile=*)
			HOSTFILE="${arg#*=}"
			;;
		--zero_stage)
			ZERO_STAGE="$1"
			shift
			;;
		--zero_stage=*)
			ZERO_STAGE="${arg#*=}"
			;;
		--offload)
			OFFLOAD="$1"
			shift
			;;
		--offload=*)
			OFFLOAD="${arg#*=}"
			;;
		*)
			echo "Unknown parameter passed: '${arg}'" >&2
			exit 1
			;;
	esac
done

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

# deepspeed "${DEEPSPEED_ARGS[@]}" \
gpu_vis=7

deepspeed --include localhost:$gpu_vis --master_port $MASTER_PORT \
	--module oa_dag.algorithms.oa \
	--train_datasets alpaca \
	--model_name_or_path "${MODEL_NAME_OR_PATH}" \
	--max_length 512 \
	--trust_remote_code True \
	--epochs 3 \
	--per_device_train_batch_size 8 \
	--per_device_eval_batch_size 4 \
	--learning_rate 2e-5 \
	--lr_scheduler_type cosine \
	--lr_warmup_ratio 0.03 \
	--weight_decay 0.0 \
	--seed 42 \
	--output_dir "${OUTPUT_DIR}" \
	--log_type wandb \
	--log_project Safe-RLHF-SFT \
	--zero_stage "${ZERO_STAGE}" \
	--offload "${OFFLOAD}" \
	--bf16 True \
	--tf32 True

# --gradient_accumulation_steps 4 \
# --gradient_checkpointing \