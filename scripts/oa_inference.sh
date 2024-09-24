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
# REF_MODEL_NAME_OR_PATH="mistralai/Mistral-7B-v0.3"
# MODEL_NAME_OR_PATH="/share/edc/home/yuxi_xie/oa_dag/model-checkpoints/coa/finallayer/metamath-05-multictxt4/checkpoint-53613"
MODEL_NAME_OR_PATH="/share/edc/home/yuxi_xie/oa_dag/model-checkpoints/coa/stage3/metamath-05-multictxt4/checkpoint-53613"

OUTPUT_DIR="/home/yuxi/Projects/OA-DAG/outputs/dev"
unset HOSTFILE
ZERO_STAGE=0
OFFLOAD="none"

mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(cd "${OUTPUT_DIR}" &>/dev/null && pwd)"
if [[ ! -f "${OUTPUT_DIR}/.gitignore" ]]; then
	echo '*' >"${OUTPUT_DIR}/.gitignore"
fi

cp -f "$0" "${OUTPUT_DIR}/script.sh"

export WANDB_MODE="offline"
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
	--module oa_dag.algorithms.oa \
	--train_datasets MetaMath/valid \
	--result_fname oa_e3-gsm8k-skip \
	--eval_datasets GSM8K \
	--model_type metamath \
	--not_lazy_tokenization \
	--need_eval \
	--do_decoding \
	--model_name_or_path "${MODEL_NAME_OR_PATH}" \
	--max_length 512 \
	--temperature 0.0 \
	--seq_temperature 0.0 \
	--decoding_occurance_threshold 4 \
	--decoding_block_size 128 \
	--context_window 4 \
	--n_back_pred 2 \
	--trust_remote_code True \
	--per_device_eval_batch_size 1 \
	--seed 42 \
	--output_dir "${OUTPUT_DIR}" \
	--log_type wandb \
	--log_project OA-AR \
	--zero_stage "${ZERO_STAGE}" \
	--offload "${OFFLOAD}" \
	--bf16 True \
	--tf32 True

# --verbal_decoding \
# --left2right \
# --result_fname left2right_e3-gsm8k \
# --add_denoising \
# --eval_datasets MATH \
# --model_name_or_path "${REF_MODEL_NAME_OR_PATH}" \
# --resume_from_ckpt "${MODEL_NAME_OR_PATH}" \

# bash scripts/oa.sh $gpu_vis $OUTPUT_DIR