"""Write data/trace0_tokens.json: the response tokens of TextSeal training trace #1 with their Qwen3.5-4B entropies
and whether the highest-entropy-50% run dropped them (entropy above the global median over all training tokens).
Usage (repo root on PYTHONPATH, HF cache available): python extract_trace_tokens.py TRACES.jsonl ENTROPY.npz"""
import hashlib
import json
import sys

import numpy as np

from config import ModelSpec
from stages import stage3_finetune as stage
from utils.tokenization import load_tokenizer

traces, entropy_file = sys.argv[1:3]
d = np.load(entropy_file)
ent = d['entropy'].astype(np.float64)
threshold = np.quantile(ent, 0.5)
keys = d['keys'].tolist()
tok = load_tokenizer(ModelSpec(name='Qwen/Qwen3.5-4B', dtype='bfloat16', pad_token=''), padding_side='left')
row = json.loads(open(traces).readline())
f = stage._mask_prompt(tok, row['prompt'], row['response'], 4096)
i = keys.index(hashlib.sha256(json.dumps(f['input_ids']).encode()).hexdigest())
e = ent[d['offsets'][i]:d['offsets'][i + 1]]
ids = [t for t, label in zip(f['input_ids'], f['labels']) if label != -100]
out = [(tok.decode([t]), round(float(h), 2), bool(h > threshold)) for t, h in zip(ids, e)]
json.dump(out, open('data/trace0_tokens.json', 'w'))
print('threshold', round(threshold, 4), 'tokens', len(out))
