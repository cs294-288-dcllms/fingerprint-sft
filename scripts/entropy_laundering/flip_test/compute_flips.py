"""Per-position 'flip' labels for teacher traces: was the sampled (watermarked) token different from the teacher's
own top choice? Also stores the teacher's sampling-distribution entropy (temperature + top-p, as at generation time).
Positions and keys match compute_entropy.py (same stage-3 tokenization), so results join with the reference entropy.
Usage: python compute_flips.py TRACES OUT.npz --adapter TEACHER_ADAPTER [--max-rows 3000]"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from config import ModelSpec
from models.loader import load_causal_lm
from stages import stage3_finetune as stage
from utils.io import read_jsonl_rows
from utils.tokenization import load_tokenizer

CHUNK, BUCKET, TOKENS_PER_BATCH = 512, 512, 8192


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('traces'); p.add_argument('out')
    p.add_argument('--teacher', default='Qwen/Qwen3.5-9B'); p.add_argument('--adapter', type=Path, required=True)
    p.add_argument('--tokenizer-model', default='Qwen/Qwen3.5-4B', help='tokenize as the student / entropy pass did')
    p.add_argument('--temperature', type=float, default=0.7); p.add_argument('--top-p', type=float, default=0.95)
    p.add_argument('--max-rows', type=int, default=0)
    args = p.parse_args()

    tok = load_tokenizer(ModelSpec(name=args.tokenizer_model, dtype='bfloat16', pad_token=''), padding_side='left')
    ttok = load_tokenizer(ModelSpec(name=args.teacher, dtype='bfloat16', pad_token=''), padding_side='left')
    if tok.get_vocab() != ttok.get_vocab():
        raise ValueError('teacher and student tokenizers differ; positions would not align')
    model = load_causal_lm(ModelSpec(name=args.teacher, dtype='bfloat16', pad_token=''), args.adapter)
    model.resize_token_embeddings(len(ttok))
    model.to('cuda').eval()

    rows = read_jsonl_rows(args.traces)[:args.max_rows or None]
    feats = [stage._mask_prompt(tok, r['prompt'], r['response'], 4096) for r in rows]
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    bucket = lambda n: min(4096, -(-n // BUCKET) * BUCKET)
    order = sorted(range(len(feats)), key=lambda i: len(feats[i]['input_ids']))
    batches, cur = [], []
    for i in order:
        L = bucket(len(feats[i]['input_ids']))
        if cur and (L != bucket(len(feats[cur[0]]['input_ids'])) or (len(cur) + 1) * L > TOKENS_PER_BATCH):
            batches.append(cur); cur = []
        cur.append(i)
    batches.append(cur)
    out, start = {}, time.time()
    for b, idx in enumerate(batches):
        L = bucket(len(feats[idx[-1]]['input_ids']))
        ids = torch.full((len(idx), L), pad_id, dtype=torch.long); mask = torch.zeros_like(ids)
        for j, i in enumerate(idx):
            n = len(feats[i]['input_ids']); ids[j, :n] = torch.tensor(feats[i]['input_ids']); mask[j, :n] = 1
        with torch.inference_mode():
            logits = model(ids.cuda(), attention_mask=mask.cuda(), use_cache=False).logits
            for j, i in enumerate(idx):
                pos = [t for t, lab in enumerate(feats[i]['labels']) if lab != -100]
                pos_t = torch.tensor(pos, device='cuda')
                tgt = torch.tensor([feats[i]['input_ids'][t] for t in pos], device='cuda')
                cols = {k: [] for k in ('flip', 'h_samp', 'h_raw', 'pmax', 'in_top_p')}
                for s0 in range(0, len(pos), CHUNK):
                    z = logits[j, pos_t[s0:s0 + CHUNK] - 1].float(); y = tgt[s0:s0 + CHUNK]
                    lp = torch.log_softmax(z, -1)
                    cols['h_raw'].append(-(lp.exp() * lp).sum(-1))
                    cols['flip'].append(z.argmax(-1) != y)
                    ps = torch.softmax(z / args.temperature, -1)                     # generation-time distribution
                    sp, si = ps.sort(-1, descending=True)
                    keep = (sp.cumsum(-1) - sp) < args.top_p                         # HF top-p rule
                    sp = sp * keep; sp = sp / sp.sum(-1, keepdim=True)
                    cols['h_samp'].append(-(sp * torch.log(sp.clamp_min(1e-30))).sum(-1))
                    cols['pmax'].append(sp[:, 0])
                    rank = (si == y[:, None]).float().argmax(-1)
                    cols['in_top_p'].append(keep.gather(1, rank[:, None])[:, 0])
                out[i] = {k: torch.cat(v).cpu().numpy() for k, v in cols.items()}
        del logits
        if b % 50 == 0:
            print(f'batch {b}/{len(batches)} rows {len(out)}/{len(feats)} {time.time() - start:.0f}s', flush=True)
    keys = [hashlib.sha256(json.dumps(f['input_ids']).encode()).hexdigest() for f in feats]
    offsets = np.cumsum([0] + [len(out[i]['flip']) for i in range(len(feats))])
    cat = lambda k, dt: np.concatenate([out[i][k] for i in range(len(feats))]).astype(dt)
    np.savez(args.out, keys=np.array(keys), offsets=offsets, flip=cat('flip', bool), h_samp=cat('h_samp', np.float16),
             h_raw=cat('h_raw', np.float16), pmax=cat('pmax', np.float16), in_top_p=cat('in_top_p', bool))
    print(json.dumps({'examples': len(keys), 'tokens': int(offsets[-1]), 'seconds': round(time.time() - start)}), flush=True)


if __name__ == '__main__':
    main()
