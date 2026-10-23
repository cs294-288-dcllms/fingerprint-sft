"""Join teacher flip labels with reference-model entropies and report how well entropy predicts flips.
Usage: python analyze_flips.py out/NAME.npz ../tokendrop/entropy/NAME.npz"""
import json
import sys

import numpy as np


def auroc(score, label):
    order = np.argsort(score, kind='stable'); rank = np.empty(len(score)); rank[order] = np.arange(1, len(score) + 1)
    n1 = label.sum(); n0 = len(label) - n1
    return float((rank[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


f, r = np.load(sys.argv[1]), np.load(sys.argv[2])
ref_all = r['entropy'].astype(np.float64)
tau = float(np.median(ref_all))                                     # the threshold the laundering run used
index = {k: i for i, k in enumerate(r['keys'].tolist())}
h_ref = []
for j, k in enumerate(f['keys'].tolist()):
    i = index[k]
    seg = ref_all[r['offsets'][i]:r['offsets'][i + 1]]
    if len(seg) != f['offsets'][j + 1] - f['offsets'][j]:
        raise ValueError('position mismatch')
    h_ref.append(seg)
h_ref = np.concatenate(h_ref)
flip, h_t, pmax, intp = f['flip'], f['h_samp'].astype(np.float64), f['pmax'].astype(np.float64), f['in_top_p']
masked = h_ref > tau
dec = np.digitize(h_ref, np.quantile(h_ref, np.linspace(0.1, 0.9, 9)))
res = {
    'positions': int(len(flip)), 'flip_rate': float(flip.mean()),
    'expected_flip_rate_1_minus_pmax': float((1 - pmax).mean()),
    'trace_token_inside_top_p': float(intp.mean()),
    'auroc_reference_entropy': auroc(h_ref, flip), 'auroc_teacher_entropy': auroc(h_t, flip),
    'tau_reference': tau, 'masked_share': float(masked.mean()),
    'flips_captured_by_mask': float(flip[masked].sum() / flip.sum()),
    'flip_rate_kept': float(flip[~masked].mean()), 'flip_rate_masked': float(flip[masked].mean()),
    'flip_rate_by_reference_entropy_decile': [round(float(flip[dec == k].mean()), 4) for k in range(10)],
    'corr_reference_vs_teacher_entropy': float(np.corrcoef(h_ref, h_t)[0, 1]),
}
print(json.dumps(res, indent=1))
