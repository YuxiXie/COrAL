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

ADAM_BETAS: tuple[float, float] = (0.9, 0.95)
