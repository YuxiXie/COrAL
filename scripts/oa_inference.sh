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
REF_MODEL_NAME_OR_PATH="mistralai/Mistral-7B-v0.3"
# MODEL_NAME_OR_PATH="/share/edc/home/yuxi_xie/oa_dag/checkpoints/oa/full_dependency/cw8_nb2_r0.25_len512/checkpoint-70290"
# MODEL_NAME_OR_PATH="/share/edc/home/yuxi_xie/oa_dag/checkpoints/oa/full_dependency/cw8_r0.25_len512/checkpoint-70290"
# MODEL_NAME_OR_PATH="/share/edc/home/yuxi_xie/oa_dag/checkpoints/oa/full_dependency/cw8_r0.25_len512/checkpoint-23430"
# MODEL_NAME_OR_PATH="/share/edc/home/yuxi_xie/oa_dag/checkpoints/oa/full_dependency/forcrpt4_cw8_r0.0_len512/checkpoint-23430"
# MODEL_NAME_OR_PATH="/share/edc/home/yuxi_xie/oa_dag/checkpoints/sft/norm_full_dependency/metamath-mistral/checkpoint-6172"
MODEL_NAME_OR_PATH="/share/edc/home/yuxi_xie/oa_dag/checkpoints/sft/norm_full_dependency/metamath-mistral/checkpoint-12344"
# MODEL_NAME_OR_PATH="/share/edc/home/yuxi_xie/oa_dag/checkpoints/sft/norm_full_dependency/metamath-mistral/checkpoint-18516"
# MODEL_NAME_OR_PATH="/share/edc/home/yuxi_xie/oa_dag/checkpoints/sft/full_dependency/metamath-mistral/checkpoint-148125"
# MODEL_NAME_OR_PATH="/share/edc/home/yuxi_xie/oa_dag/checkpoints/oa/full_dependency/sft_interleaved_forcrpt0_cw4_r0.0_len512/checkpoint-11715"
# MODEL_NAME_OR_PATH="/share/edc/home/yuxi_xie/oa_dag/checkpoints/oa/norm_full_dependency/sft_interleaved_simple-sym_forcrpt2_cw4_r0.0_len512/checkpoint-20480"
# MODEL_NAME_OR_PATH="/share/edc/home/yuxi_xie/oa_dag/checkpoints/oa/norm_full_dependency/sft_interleaved_simple_forcrpt2_cw4_r0.0_len512/checkpoint-10240"
# MODEL_NAME_OR_PATH="/share/edc/home/yuxi_xie/oa_dag/checkpoints/oa/norm_full_dependency/sft_interleaved_simple-sym_forcrpt2_cw4_r0.0_len512/checkpoint-5858"
# MODEL_NAME_OR_PATH="/share/edc/home/yuxi_xie/oa_dag/checkpoints/oa/norm_full_dependency/sft_interleaved_forcrpt1_cw2_r0.0_len512/checkpoint-17574"
# MODEL_NAME_OR_PATH="/share/edc/home/yuxi_xie/oa_dag/checkpoints/oa/norm_full_dependency/sft_interleaved_forcrpt2_cw4_r0.0_len512/checkpoint-11715"
MODEL_NAME_OR_PATH="/share/edc/home/yuxi_xie/oa_dag/checkpoints/oa/norm_full_dependency/sft_interleaved_forcrpt2_cw4_r0.0_len512/checkpoint-23430"
# MODEL_NAME_OR_PATH="/share/edc/home/yuxi_xie/oa_dag/checkpoints/oa/norm_full_dependency/sft_interleaved_forcrpt2_cw4_r0.0_len512/checkpoint-35145"
# MODEL_NAME_OR_PATH="/share/edc/home/yuxi_xie/oa_dag/checkpoints/oa/norm_full_dependency/sft_interleaved_dym2_r0.0_len512/checkpoint-5858"
# MODEL_NAME_OR_PATH="/share/edc/home/yuxi_xie/oa_dag/checkpoints/oa/norm_full_dependency/sft_interleaved_dym4_r0.0_len512/checkpoint-11715"
# MODEL_NAME_OR_PATH="/share/edc/home/yuxi_xie/oa_dag/checkpoints/oa/norm_full_dependency/sft_interleaved_dym4_r0.0_len512/checkpoint-23430"
# MODEL_NAME_OR_PATH="/share/edc/home/yuxi_xie/oa_dag/checkpoints/oa/norm_full_dependency/sft_interleaved_dym4_r0.0_len512/checkpoint-35145"
# MODEL_NAME_OR_PATH="/share/edc/home/yuxi_xie/oa_dag/checkpoints/oa/norm_full_dependency/sft_interleaved_simple_forcrpt2_cw4_r0.0_len512/checkpoint-35145"
OUTPUT_DIR="/share/edc/home/yuxi_xie/oa_dag/checkpoints/oa/full_dependency/dev"
unset HOSTFILE
ZERO_STAGE=3
OFFLOAD="none"

mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(cd "${OUTPUT_DIR}" &>/dev/null && pwd)"
if [[ ! -f "${OUTPUT_DIR}/.gitignore" ]]; then
	echo '*' >"${OUTPUT_DIR}/.gitignore"
fi

cp -f "$0" "${OUTPUT_DIR}/script.sh"

export WANDB_MODE=dryrun
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
	--left2right \
	--result_fname left2right_dn-e2 \
	--verbal_decoding \
	--eval_datasets GSM8K \
	--model_type metamath \
	--not_lazy_tokenization \
	--need_eval \
	--do_decoding \
	--model_name_or_path "${MODEL_NAME_OR_PATH}" \
	--max_length 512 \
	--temperature 0.0 \
	--seq_temperature 0.0 \
	--decoding_occurance_threshold 8 \
	--decoding_block_size 64 \
	--context_window 4 \
	--n_back_pred 1 \
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

# --add_denoising \
# --model_name_or_path "${REF_MODEL_NAME_OR_PATH}" \
# --resume_from_ckpt "${MODEL_NAME_OR_PATH}" \

# bash scripts/oa.sh $gpu_vis $OUTPUT_DIR