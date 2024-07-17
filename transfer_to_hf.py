import os
from tqdm import tqdm
from huggingface_hub import HfApi
api = HfApi()

for dirname in tqdm([
    "mu0.05to0.55-dymin-rmu0.55/checkpoint-147741", "mu0.05to0.95-dymin-rmu0.55/checkpoint-147741", "mu0.05to0.95-dymin-rmu0.55-e3/checkpoint-112640"
]):
    for fname in [
        "config.json",  "latest",  "special_tokens_map.json",  "tokenizer_config.json",  "tokenizer.json",  "zero_to_fp32.py",
        "added_tokens.json", "tokenizer.model",
        "pytorch_model.bin", 
    ]:
        if not os.path.exists(f"/share/edc/home/yuxi_xie/oa_dag/checkpoints/oa/metamath-mistral/{dirname}/{fname}"): continue
        api.upload_file(
            path_or_fileobj=f"/share/edc/home/yuxi_xie/oa_dag/checkpoints/oa/metamath-mistral/{dirname}/{fname}",
            path_in_repo="{}/{}".format(dirname.replace('/', '@'), fname),
            repo_id="yuxixie/OA-DAG"
        )

from huggingface_hub import hf_hub_download

dirname = "mu0.05to0.55-dymin-rmu0.55@checkpoint-147741"
target_dir = "/scratch/e0672129/hf"
os.makedirs(f'{target_dir}/{dirname}', exist_ok=True)
for fname in [
    "config.json",  "latest",  "special_tokens_map.json",  "tokenizer_config.json",  "tokenizer.json",  "zero_to_fp32.py",
    "added_tokens.json", "tokenizer.model",
    "pytorch_model.bin", 
]:
    hf_hub_download(
        repo_id="yuxixie/LLaVA-DPO", 
        filename=f'{dirname}/{fname}',
        local_dir=f'{target_dir}/{dirname}',
    )
