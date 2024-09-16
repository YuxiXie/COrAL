"""Constant variables."""

from __future__ import annotations


__all__ = [
    'IGNORE_INDEX',
    'DEFAULT_BOS_TOKEN',
    'DEFAULT_EOS_TOKEN',
    'DEFAULT_PAD_TOKEN',
    'DEFAULT_UNK_TOKEN',
    'PROMPT_BEGIN',
    'PROMPT_USER',
    'PROMPT_ASSISTANT',
    'PROMPT_INPUT',
    'PROMPT_DICT',
    'PROMPT_DICTS',
    'ADAM_BETAS',
]


IGNORE_INDEX: int = -100
DEFAULT_BOS_TOKEN: str = '<s>'
DEFAULT_EOS_TOKEN: str = '</s>'
DEFAULT_PAD_TOKEN: str = '<pad>'
DEFAULT_UNK_TOKEN: str = '<unk>'

# PROMPT_BEGIN: str = 'BEGINNING OF CONVERSATION: '
# PROMPT_USER: str = 'USER: {input} '
# PROMPT_ASSISTANT: str = 'ASSISTANT:'  # should not have a space at the end

PROMPT_BEGIN: str = ''
PROMPT_USER: str = '[INST] {input} [/INST]'
PROMPT_ASSISTANT: str = ''

PROMPT_INPUT: str = PROMPT_BEGIN + PROMPT_USER + PROMPT_ASSISTANT

PROMPT_DICT: dict[str, str] = {
    'prompt_begin': PROMPT_BEGIN,
    'prompt_user': PROMPT_USER,
    'prompt_assistant': PROMPT_ASSISTANT,
    'prompt_input': PROMPT_INPUT,
}

PROMPT_DICTS: dict[str, dict[str, str]] = {
    'mistral-instruct': PROMPT_DICT,
    'metamath': {
        'prompt_begin': 'Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n',
        'prompt_user': '### Instruction:\n{input}\n\n',
        'prompt_assistant': '### Response:'
    },
    'llama3-instruct': {
        'prompt_begin': '<|start_header_id|>system<|end_header_id|>\n\nBelow is an instruction that describes a task. Write a response that appropriately completes the request.<|eot_id|>',
        'prompt_user': '<|start_header_id|>user<|end_header_id|>\n\n{input}<|eot_id|>',
        'prompt_assistant': '<|start_header_id|>assistant<|end_header_id|>\n\n'
    },
    'tulu': {
        'prompt_begin': 'Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n',
        'prompt_user': '<|user|>\n{input}\n',
        'prompt_assistant': '<|assistant|>\n'
    }
}

ADAM_BETAS: tuple[float, float] = (0.9, 0.95)

metamath_accu = [
    [88.69, 7.38, 2.57, 1.85, 0.5, 0.33, 0.22, 0.17, 0.15, 0.04],
    [88.92, 7.79, 2.60, 1.81, 0.48, 0.37, 0.22, 0.17, 0.13, 0.04],
    [89.05, 7.5, 2.58, 1.81, 0.46, 0.33, 0.22, 0.18, 0.13, 0.05],
    [89.19, 7.49, 2.55, 1.78, 0.49, 0.32, 0.22, 0.17, 0.15, 0.04],
    [89.18, 7.61, 2.55, 1.78, 0.48, 0.31, 0.24, 0.18, 0.14, 0.04],
    [89.09, 7.62, 2.60, 1.81, 0.47, 0.33, 0.24, 0.17, 0.15, 0.05],
    [87.95, 7.79, 2.58, 1.82, 0.49, 0.34, 0.22, 0.18, 0.13, 0.05],
    [87.95, 8.35, 2.76, 1.88, 0.55, 0.38, 0.23, 0.18, 0.15, 0.04],
    [81.89, 9.91, 3.86, 2.58, 1.03, 0.74, 0.57, 0.39, 0.32, 0.09],
    [84.13, 6.85, 2.46, 1.32, 0.81, 0.55, 0.4, 0.29, 0.28, 0.09],
    [69.77, 9.57, 4.24, 2.65, 1.77, 1.32, 0.97, 0.79, 0.65, 0.24],
    [55.53, 11.83, 5.96, 3.72, 2.64, 2.03, 1.62, 1.29, 1.05, 0.4],
    [41.34, 13.02, 7.23, 4.98, 3.52, 2.86, 2.22, 1.93, 1.57, 0.61],
]
