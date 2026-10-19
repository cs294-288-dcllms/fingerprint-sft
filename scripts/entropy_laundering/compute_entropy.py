"""Per-token entropy and surprisal of teacher traces under a reference model (no key needed).

Tokenizes each trace exactly as stage-3 SFT does (stages.stage3_finetune._mask_prompt, 4,096
tokens), runs the base student, and stores for every supervised position t the entropy of the
reference model's next-token distribution H(p(. | x_<t)) and the surprisal -log p(x_t | x_<t).
Output: .npz with concatenated float16 arrays, per-example offsets and sha256 keys of input_ids
(the key the SFT wrapper uses to look rows up).
Usage: python compute_entropy.py TRACES OUT.npz [--model Qwen/Qwen3.5-4B]
"""
import argparse
import hashlib
import json
import time

import numpy as np
import torch

from config import ModelSpec
from models.loader import load_causal_lm
from stages import stage3_finetune as stage
from utils.io import read_jsonl_rows
from utils.tokenization import load_tokenizer

CHUNK = 512
BUCKET = 512
TOKENS_PER_BATCH = 8192


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('traces')
    parser.add_argument('out')
    parser.add_argument('--model', default='Qwen/Qwen3.5-4B')
    parser.add_argument('--max-rows', type=int, default=0)
    args = parser.parse_args()

    spec = ModelSpec(name=args.model, dtype='bfloat16', pad_token='')
    tokenizer = load_tokenizer(spec, padding_side='left')
    model = load_causal_lm(spec)
    model.resize_token_embeddings(len(tokenizer))
    model.to('cuda').eval()

    rows = read_jsonl_rows(args.traces)
    if args.max_rows:
        rows = rows[:args.max_rows]
    feats = [stage._mask_prompt(tokenizer, r['prompt'], r['response'], 4096) for r in rows]
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    # Right-pad to a few fixed lengths so kernels compile once per bucket; padding sits after
    # every real token, so causal outputs at real positions are unchanged.
    def bucket(n):
        return min(4096, -(-n // BUCKET) * BUCKET)
    order = sorted(range(len(feats)), key=lambda i: len(feats[i]['input_ids']))
    batches, cur = [], []
    for i in order:
        if cur and (bucket(len(feats[i]['input_ids'])) != bucket(len(feats[cur[0]]['input_ids']))
                    or (len(cur) + 1) * bucket(len(feats[i]['input_ids'])) > TOKENS_PER_BATCH):
            batches.append(cur); cur = []
        cur.append(i)
    if cur:
        batches.append(cur)
    results = {}
    start = time.time()
    for b, idx in enumerate(batches):
        L = bucket(len(feats[idx[-1]]['input_ids']))
        ids = torch.full((len(idx), L), pad_id, dtype=torch.long)
        mask = torch.zeros((len(idx), L), dtype=torch.long)
        for j, i in enumerate(idx):
            n = len(feats[i]['input_ids'])
            ids[j, :n] = torch.tensor(feats[i]['input_ids'])
            mask[j, :n] = 1
        with torch.inference_mode():
            logits = model(ids.cuda(), attention_mask=mask.cuda(), use_cache=False).logits
            for j, i in enumerate(idx):
                labels = feats[i]['labels']
                if labels and labels[0] != -100:
                    raise ValueError('first token is supervised; no context to score it')
                positions = [t for t, label in enumerate(labels) if label != -100]
                pos = torch.tensor(positions, device='cuda')
                tgt = torch.tensor([feats[i]['input_ids'][t] for t in positions], device='cuda')
                ent, sur = [], []
                for s0 in range(0, len(positions), CHUNK):
                    logp = torch.log_softmax(logits[j, pos[s0:s0 + CHUNK] - 1].float(), dim=-1)
                    ent.append(-(logp.exp() * logp).sum(-1))
                    sur.append(-logp.gather(1, tgt[s0:s0 + CHUNK, None])[:, 0])
                results[i] = (torch.cat(ent).half().cpu().numpy(), torch.cat(sur).half().cpu().numpy())
        del logits
        if b % 50 == 0:
            print(f'batch {b}/{len(batches)} rows {len(results)}/{len(feats)} {time.time() - start:.0f}s', flush=True)
    keys, offsets, entropies, surprisals = [], [0], [], []
    for i, f in enumerate(feats):
        keys.append(hashlib.sha256(json.dumps(f['input_ids']).encode()).hexdigest())
        entropies.append(results[i][0]); surprisals.append(results[i][1])
        offsets.append(offsets[-1] + len(results[i][0]))
    np.savez(args.out, keys=np.array(keys), offsets=np.array(offsets, dtype=np.int64),
             entropy=np.concatenate(entropies), surprisal=np.concatenate(surprisals))
    print(json.dumps({'examples': len(keys), 'tokens': int(offsets[-1]),
                      'seconds': round(time.time() - start)}), flush=True)


if __name__ == '__main__':
    main()
