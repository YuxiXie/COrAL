# from transformers import (
#     AutoModelForCausalLM,
#     AutoTokenizer,
# )

# model = AutoModelForCausalLM.from_pretrained(
#     "meta-llama/Meta-Llama-3.1-8B",
#     token="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",
# )
# tokenizer = AutoTokenizer.from_pretrained(
#     "meta-llama/Meta-Llama-3.1-8B",
#     token="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",
# )

from tqdm import tqdm
from datasets import load_dataset

metamath = load_dataset('meta-math/MetaMathQA', split='train')
gsm8k = load_dataset('openai/gsm8k', 'main', split='train')

questions = [x['question'] for x in gsm8k]
data = [x for x in metamath if x['original_question'] in questions]

import ipdb; ipdb.set_trace()
