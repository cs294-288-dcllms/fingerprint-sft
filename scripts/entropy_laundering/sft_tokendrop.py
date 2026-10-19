"""Stage-3 SFT where a random fraction of response tokens is dropped from the loss.

Dropped tokens stay in the input (the student still reads them) but get label -100,
so the student is never trained to predict them. The drop pattern is a deterministic
function of (drop seed, token ids), so every DDP rank builds the same dataset.
Also passes max_length=4096 to SFTConfig (TRL 0.24 otherwise truncates at 1024).

Mask modes: random (each response token independently with probability --drop-rate), or a
corpus-wide selection of exactly --drop-rate of all response tokens ranked by a per-token score
from compute_entropy.py: top-entropy, bottom-entropy, top-surprisal (ties broken at random).

Usage: python sft_tokendrop.py --drop-rate 0.5 [--mask-mode top-entropy --entropy-file F.npz]
       [--drop-seed 0] <stage3_finetune args>
"""
import argparse
import hashlib
import json
import os
import random

MAX_LENGTH = 4096


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--drop-rate', type=float, required=True)
    parser.add_argument('--drop-seed', type=int, default=0)
    parser.add_argument('--mask-mode', default='random',
                        choices=['random', 'top-entropy', 'bottom-entropy', 'top-surprisal'])
    parser.add_argument('--entropy-file')
    parser.add_argument('--protect-special', action='store_true',
                        help='never drop special tokens (EOS, <think>, </think>, other added tokens)')
    options, stage_args = parser.parse_known_args()
    if not 0.0 <= options.drop_rate < 1.0:
        raise ValueError('--drop-rate must be in [0, 1)')

    from stages import stage3_finetune as stage
    args = stage.build_parser().parse_args(stage_args)
    if args.max_seq_length != MAX_LENGTH:
        raise ValueError(f'--max-seq-length must be {MAX_LENGTH}')

    stats = {'examples': 0, 'response_tokens': 0, 'kept_tokens': 0}
    ranked = None
    if options.mask_mode != 'random':
        import numpy as np
        data = np.load(options.entropy_file)
        score = (data['surprisal'] if options.mask_mode == 'top-surprisal' else data['entropy']).astype(np.float64)
        if options.mask_mode == 'bottom-entropy':
            score = -score
        jitter = np.random.default_rng(options.drop_seed).random(score.shape) * 1e-6
        count = int(round(options.drop_rate * score.size))
        drop = np.zeros(score.size, dtype=bool)
        drop[np.argsort(-(score + jitter), kind='stable')[:count]] = True
        offsets = data['offsets']
        ranked = {key: drop[offsets[i]:offsets[i + 1]] for i, key in enumerate(data['keys'].tolist())}
    original_mask_prompt = stage._mask_prompt

    protected = None

    def mask_prompt(tokenizer, prompt, response, max_length):
        nonlocal protected
        if protected is None:
            protected = set(tokenizer.all_special_ids) if options.protect_special else set()
            if options.protect_special:
                protected |= {i for i in tokenizer.convert_tokens_to_ids(['<think>', '</think>'])
                              if i is not None and i != tokenizer.unk_token_id}
                protected |= set(tokenizer.get_added_vocab().values())
        features = original_mask_prompt(tokenizer, prompt, response, max_length)
        digest = hashlib.sha256(json.dumps(features['input_ids']).encode())
        rng = random.Random(options.drop_seed * 1_000_003 + int.from_bytes(digest.digest()[:8], 'big'))
        selected = None
        if ranked is not None:
            selected = ranked[digest.hexdigest()]
            if len(selected) != sum(label != -100 for label in features['labels']):
                raise ValueError('entropy file does not match this tokenization')
        labels, j = [], 0
        for label in features['labels']:
            if label != -100:
                stats['response_tokens'] += 1
                dropped = selected[j] if selected is not None else rng.random() < options.drop_rate
                j += 1
                if dropped and label in protected:
                    dropped = False
                    stats['protected_kept'] = stats.get('protected_kept', 0) + 1
                if dropped:
                    label = -100
                else:
                    stats['kept_tokens'] += 1
            labels.append(label)
        stats['examples'] += 1
        return {**features, 'labels': labels}

    original_config, original_trainer = stage.SFTConfig, stage.SFTTrainer

    def config(*positional, **kwargs):
        return original_config(*positional, **(kwargs | {'max_length': MAX_LENGTH}))

    class AuditedTrainer(original_trainer):
        def __init__(self, *positional, **kwargs):
            super().__init__(*positional, **kwargs)
            if self.args.max_length != MAX_LENGTH:
                raise ValueError('Trainer did not receive max_length=4096')
            longest = max(len(row['input_ids']) for row in self.train_dataset)
            audit = {**stats, 'drop_rate': options.drop_rate, 'drop_seed': options.drop_seed,
                     'mask_mode': options.mask_mode, 'entropy_file': options.entropy_file,
                     'protect_special': options.protect_special,
                     'kept_fraction': stats['kept_tokens'] / max(stats['response_tokens'], 1),
                     'longest_example': longest, 'rank': int(os.environ.get('RANK', '0'))}
            if longest > MAX_LENGTH:
                raise ValueError('Example longer than max_length would be truncated')
            os.makedirs(self.args.output_dir, exist_ok=True)
            with open(os.path.join(self.args.output_dir, f"tokendrop-audit-rank{audit['rank']}.json"), 'w') as handle:
                json.dump(audit, handle, indent=2)
            print('TOKENDROP_AUDIT ' + json.dumps(audit), flush=True)

    stage._mask_prompt = mask_prompt
    stage.SFTConfig, stage.SFTTrainer = config, AuditedTrainer
    stage.main(stage_args)


if __name__ == '__main__':
    main()
