"""Per-token entropy of teacher traces under a reference model from another family.

The reference model scores the response in its own tokenization (question placed in its own chat
template). Each Qwen response token, as built by stage-3 SFT, gets the maximum reference entropy
over the reference tokens whose character spans overlap it (and the summed surprisal). Output has
the same layout and keys as compute_entropy.py, so sft_tokendrop.py can use it unchanged.
Usage: python compute_entropy_xref.py TRACES OUT.npz --ref-model PATH [--mapping max|start] [--max-rows N]
"""
import argparse
import bisect
import hashlib
import json
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import ModelSpec
from stages import stage3_finetune as stage
from utils.io import read_jsonl_rows
from utils.tokenization import load_tokenizer

CHUNK, BUCKET, TOKENS_PER_BATCH = 512, 512, 8192
USER_TAG, END_TAG = '<|im_start|>user\n', '<|im_end|>'


def question_from_prompt(prompt: str) -> str:
    start = prompt.index(USER_TAG) + len(USER_TAG)
    return prompt[start:prompt.index(END_TAG, start)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('traces')
    parser.add_argument('out')
    parser.add_argument('--ref-model', required=True)
    parser.add_argument('--max-rows', type=int, default=0)
    parser.add_argument('--mapping', choices=['max', 'start'], default='max',
                        help='max: max entropy over overlapping reference tokens; '
                             'start: entropy of the reference token containing the first character')
    args = parser.parse_args()

    qtok = load_tokenizer(ModelSpec(name='Qwen/Qwen3.5-4B', dtype='bfloat16', pad_token=''), padding_side='left')
    rtok = AutoTokenizer.from_pretrained(args.ref_model)
    model = AutoModelForCausalLM.from_pretrained(args.ref_model, dtype=torch.bfloat16).to('cuda').eval()

    rows = read_jsonl_rows(args.traces)
    if args.max_rows:
        rows = rows[:args.max_rows]
    items = []
    for row in rows:
        feats = stage._mask_prompt(qtok, row['prompt'], row['response'], 4096)
        n_sup = sum(label != -100 for label in feats['labels'])
        q = qtok(row['response'], add_special_tokens=False, return_offsets_mapping=True)
        prompt_len = len(feats['input_ids']) - n_sup
        expected = feats['input_ids'][prompt_len:]
        has_eos = len(expected) == len(q['input_ids']) + 1
        if expected[:len(q['input_ids'])] != q['input_ids'][:len(expected)]:
            raise ValueError('Qwen response tokenization differs from SFT tokenization')
        prefix = rtok.apply_chat_template([{'role': 'user', 'content': question_from_prompt(row['prompt'])}],
                                          tokenize=False, add_generation_prompt=True)
        if not prefix.rstrip().endswith('<think>'):   # reasoning templates already open the think block
            prefix += '<think>\n'
        p_ids = rtok(prefix, add_special_tokens=False)['input_ids']
        r = rtok(row['response'], add_special_tokens=False, return_offsets_mapping=True)
        ids = (p_ids + r['input_ids'] + [rtok.eos_token_id])[:4096]
        items.append({'key': hashlib.sha256(json.dumps(feats['input_ids']).encode()).hexdigest(),
                      'n_sup': n_sup, 'q_offsets': q['offset_mapping'][:n_sup], 'has_eos': has_eos,
                      'ref_ids': ids, 'ref_start': len(p_ids), 'r_offsets': r['offset_mapping']})

    bucket = lambda n: min(4096, -(-n // BUCKET) * BUCKET)
    order = sorted(range(len(items)), key=lambda i: len(items[i]['ref_ids']))
    batches, cur = [], []
    for i in order:
        L = bucket(len(items[i]['ref_ids']))
        if cur and (L != bucket(len(items[cur[0]]['ref_ids'])) or (len(cur) + 1) * L > TOKENS_PER_BATCH):
            batches.append(cur); cur = []
        cur.append(i)
    if cur:
        batches.append(cur)

    pad = rtok.pad_token_id if rtok.pad_token_id is not None else rtok.eos_token_id
    ent_out, sur_out = {}, {}
    start = time.time()
    for b, idx in enumerate(batches):
        L = bucket(len(items[idx[-1]]['ref_ids']))
        ids = torch.full((len(idx), L), pad, dtype=torch.long)
        mask = torch.zeros((len(idx), L), dtype=torch.long)
        for j, i in enumerate(idx):
            n = len(items[i]['ref_ids'])
            ids[j, :n] = torch.tensor(items[i]['ref_ids']); mask[j, :n] = 1
        with torch.inference_mode():
            logits = model(ids.cuda(), attention_mask=mask.cuda(), use_cache=False).logits
            for j, i in enumerate(idx):
                it = items[i]
                pos = list(range(it['ref_start'], len(it['ref_ids'])))   # scored reference tokens
                p = torch.tensor(pos, device='cuda')
                tgt = torch.tensor([it['ref_ids'][t] for t in pos], device='cuda')
                ent, sur = [], []
                for s0 in range(0, len(pos), CHUNK):
                    logp = torch.log_softmax(logits[j, p[s0:s0 + CHUNK] - 1].float(), dim=-1)
                    ent.append(-(logp.exp() * logp).sum(-1)); sur.append(-logp.gather(1, tgt[s0:s0 + CHUNK, None])[:, 0])
                ref_ent = torch.cat(ent).float().cpu().numpy(); ref_sur = torch.cat(sur).float().cpu().numpy()
                n_resp = min(len(it['r_offsets']), len(pos))
                starts = [o[0] for o in it['r_offsets'][:n_resp]]
                ends = [o[1] for o in it['r_offsets'][:n_resp]]
                qe = np.zeros(it['n_sup'], dtype=np.float32); qs = np.zeros(it['n_sup'], dtype=np.float32)
                for k, (s, e) in enumerate(it['q_offsets']):
                    lo = max(bisect.bisect_right(ends, s) , 0)
                    hi = bisect.bisect_left(starts, max(e, s + 1))
                    if hi <= lo:
                        lo, hi = max(min(lo, n_resp - 1), 0), max(min(lo, n_resp - 1), 0) + 1
                    if args.mapping == 'start':
                        c = min(max(bisect.bisect_right(starts, s) - 1, 0), n_resp - 1)
                        qe[k] = ref_ent[c]; qs[k] = ref_sur[c]
                    else:
                        qe[k] = ref_ent[lo:hi].max(); qs[k] = ref_sur[lo:hi].sum()
                if it['has_eos'] and len(it['q_offsets']) < it['n_sup']:
                    qe[-1] = ref_ent[-1]; qs[-1] = ref_sur[-1]   # Qwen EOS <- reference EOS
                ent_out[i], sur_out[i] = qe.astype(np.float16), qs.astype(np.float16)
        del logits
        if b % 50 == 0:
            print(f'batch {b}/{len(batches)} rows {len(ent_out)}/{len(items)} {time.time() - start:.0f}s', flush=True)

    offsets = [0]
    for i in range(len(items)):
        offsets.append(offsets[-1] + items[i]['n_sup'])
    np.savez(args.out, keys=np.array([it['key'] for it in items]), offsets=np.array(offsets, dtype=np.int64),
             entropy=np.concatenate([ent_out[i] for i in range(len(items))]),
             surprisal=np.concatenate([sur_out[i] for i in range(len(items))]))
    print(json.dumps({'examples': len(items), 'tokens': offsets[-1], 'seconds': round(time.time() - start)}), flush=True)


if __name__ == '__main__':
    main()
