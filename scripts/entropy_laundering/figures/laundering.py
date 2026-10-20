"""Slide figure for entropy laundering: watermark lives in uncertain tokens -> drop them from the loss -> fingerprint gone."""
import math
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

INK, MUTED, RULE, GREYF = '#1d2128', '#6b7280', '#e5e7eb', '#f3f4f6'
HOT, LHOT = '#d9480f', '#ffe3d3'               # high-entropy tokens
GRN, LGRN = '#15803d', '#dcfce7'               # tokens kept in the loss
TS, SY = '#2a78d6', '#c98a00'                   # TextSeal, SynthID (validated pair)
plt.rcParams.update({'mathtext.fontset': 'cm', 'font.family': 'DejaVu Sans', 'font.size': 11, 'text.color': INK,
                     'axes.edgecolor': RULE, 'axes.labelcolor': MUTED, 'xtick.color': MUTED, 'ytick.color': MUTED})
fig = plt.figure(figsize=(13, 6.2), dpi=200); fig.patch.set_facecolor('white')

# Real tokens: opening of TextSeal training trace #1, with Qwen3.5-4B entropies (nats). In the
# highest-entropy-50% run a token is dropped if its entropy is above the global median (0.39 nats).
import json
toks = json.load(open('data/trace0_tokens.json'))[:18]

def token_rows(ax, y0, style, numbers=False):
    """Lay tokens out left to right, wrapping at the panel edge. style(tok, h, dropped) -> (fill, edge, colour, mark)."""
    x, y = 0.005, y0
    for t, h, dropped in toks:
        label = t.strip() or t
        w = 0.022 * len(label) + 0.046
        if x + w > 1.0:
            x, y = 0.005, y - 0.135
        fill, edge, col, mark = style(t, h, dropped)
        ax.add_patch(FancyBboxPatch((x, y - 0.035), w - 0.012, 0.07, boxstyle='round,pad=0.003,rounding_size=0.012',
                                    facecolor=fill, edgecolor=edge, linewidth=1.1, clip_on=False))
        ax.text(x + (w - 0.012) / 2, y, label, ha='center', va='center', fontsize=10, color=col)
        under = f'{h:.2f}' if numbers else mark
        if under:
            ax.text(x + (w - 0.012) / 2, y - 0.058, under, ha='center', va='center', fontsize=7.5 if numbers else 8.5,
                    color=HOT if (numbers and dropped) else (GRN if mark == '✓' else MUTED))
        x += w

# ---------- Panel 1 ----------
ax1 = fig.add_axes([0.02, 0.05, 0.28, 0.86]); ax1.set_xlim(0, 1); ax1.set_ylim(0, 1); ax1.axis('off')
ax1.text(0.0, 1.0, '1  Where the watermark lives', fontsize=13, weight='bold')
ax1.text(0.0, 0.90, 'Start of a real TextSeal training trace, with each\ntoken\'s entropy under a reference model (Qwen):', fontsize=10,
         color=MUTED, va='center', linespacing=1.4)
token_rows(ax1, 0.78, lambda t, h, d: (LHOT, HOT, INK, None) if d else (GREYF, RULE, INK, None), numbers=True)
ax1.add_patch(FancyBboxPatch((0.0, 0.37), 0.05, 0.035, boxstyle='round,pad=0,rounding_size=0.008', facecolor=LHOT, edgecolor=HOT))
ax1.text(0.07, 0.387, 'top half by entropy (> 0.39 nats): "Okay", "let", "So"', fontsize=9, color=MUTED, va='center')
ax1.add_patch(FancyBboxPatch((0.0, 0.32), 0.05, 0.035, boxstyle='round,pad=0,rounding_size=0.008', facecolor=GREYF, edgecolor=RULE))
ax1.text(0.07, 0.337, 'bottom half: forced by context ("question", "term")', fontsize=9, color=MUTED, va='center')
ax1.text(0.0, 0.24, 'The key can only tip a choice when there is a choice,\nso the watermark signal concentrates in the\norange tokens.',
         fontsize=10, va='top', linespacing=1.4)
ax1.text(0.0, 0.07, 'TextSeal training trace #1; threshold = median entropy over all traces.', fontsize=8.5, color=MUTED, style='italic')

# ---------- Panel 2 ----------
ax2 = fig.add_axes([0.335, 0.05, 0.28, 0.86]); ax2.set_xlim(0, 1); ax2.set_ylim(0, 1); ax2.axis('off')
ax2.text(0.0, 1.0, '2  Drop those tokens from the loss', fontsize=13, weight='bold')
ax2.text(0.0, 0.90, 'Fine-tune the student on the same trace, but only\nlearn to predict the bottom-half tokens:', fontsize=10,
         color=MUTED, va='center', linespacing=1.4)
token_rows(ax2, 0.78, lambda t, h, d: ('white', '#c7ccd3', '#b8bec7', '✗') if d else (LGRN, GRN, INK, '✓'))
ax2.add_patch(FancyBboxPatch((0.0, 0.392), 0.05, 0.035, boxstyle='round,pad=0,rounding_size=0.008', facecolor=LGRN, edgecolor=GRN))
ax2.text(0.07, 0.41, 'trained on', fontsize=9, color=MUTED, va='center')
ax2.add_patch(FancyBboxPatch((0.30, 0.392), 0.05, 0.035, boxstyle='round,pad=0,rounding_size=0.008', facecolor='white', edgecolor='#c7ccd3'))
ax2.text(0.37, 0.41, 'dropped (still read as context)', fontsize=9, color=MUTED, va='center')
ax2.text(0.0, 0.30, r'$\mathcal{L}(\theta) = -\frac{1}{|K|}\sum_{(i,t)\,\in\, K} \log p_\theta(x_{i,t} \mid x_{i,<t})$', fontsize=14, va='center')
ax2.text(0.0, 0.205, r'$K = \{(i,t):\ t \in \mathrm{response}_i,\ \ H_{\mathrm{ref}}(i,t) \leq \tau\}$', fontsize=12, va='center')
ax2.text(0.0, 0.125, 'τ = median entropy over all training tokens (0.39 nats) → 50% dropped.\n|K| = kept tokens per 16-trace batch. Prompt tokens never trained.',
         fontsize=8.5, color=MUTED, va='center', linespacing=1.4)
ax2.text(0.0, 0.035, 'Needs only the traces: no teacher access, no key.', fontsize=10, weight='bold', va='center')

# ---------- Panel 3: results ----------
fig.text(0.655, 0.91, '3  Fingerprint gone, accuracy kept', fontsize=13, weight='bold', va='baseline')
fig.text(0.655, 0.865, 'Detection on the trained student (entropy-weighted p) and accuracy', fontsize=9.5, color=MUTED)
rows = [('No masking', 3.3e-10, 6.8e-15, '83.2 / 82.9%'),
        ('Random 50%', 6.4e-6, 5.5e-10, '82.0 / 82.9%'),
        ('Lowest-entropy 50%', 3.7e-11, 7.7e-16, '82.9 / 83.0%'),
        ('Highest-entropy 50%', 0.83, 0.30, '81.4 / 82.4%'),
        ('Never watermarked', 0.85, 0.40, '—')]
ax3 = fig.add_axes([0.81, 0.16, 0.13, 0.66])
n = len(rows); bh = 0.34
for i, (label, pts, psy, acc) in enumerate(rows):
    y = n - 1 - i
    for off, pv, col in ((bh / 2 + 0.02, pts, TS), (-bh / 2 - 0.02, psy, SY)):
        v = -math.log10(pv)
        ax3.barh(y + off, max(v, 0.15), height=bh, color=col, edgecolor='white', linewidth=0)
        ax3.text(max(v, 0.15) + 0.3 if v > 2 else 2.3, y + off, f'p={pv:.2g}', va='center', fontsize=8, color=INK)
    fig.text(0.655, 0.16 + 0.66 * (y + 0.6) / (n + 0.2), label, fontsize=10, va='center',
             weight='bold' if 'Highest' in label else 'normal', color=HOT if 'Highest' in label else INK)
    fig.text(0.655, 0.16 + 0.66 * (y + 0.6) / (n + 0.2) - 0.035, f'accuracy {acc}' if acc != '—' else 'control (null)',
             fontsize=8.5, va='center', color=MUTED)
    if 'Highest' in label:
        fig.text(0.655, 0.16 + 0.66 * (y + 0.6) / (n + 0.2) - 0.068, '✓ accuracy holds',
                 fontsize=8.5, va='center', color=GRN, weight='bold')
ax3.axvline(2, color=MUTED, lw=1, ls=(0, (3, 3)))
ax3.text(2.2, n - 0.45, 'p = 0.01', fontsize=8, color=MUTED, va='bottom')
ax3.set_xlim(0, 16); ax3.set_ylim(-0.6, n - 0.4); ax3.set_yticks([])
ax3.set_xticks([0, 5, 10, 15]); ax3.set_xlabel('evidence: −log₁₀ p', fontsize=9)
ax3.spines[['top', 'right', 'left']].set_visible(False); ax3.tick_params(labelsize=8.5)
for k, (name, col) in enumerate((('TextSeal', TS), ('SynthID', SY))):
    fig.patches.append(FancyBboxPatch((0.80 + k * 0.08, 0.055), 0.012, 0.022, boxstyle='round,pad=0,rounding_size=0.003',
                                      facecolor=col, edgecolor='none', transform=fig.transFigure, figure=fig))
    fig.text(0.816 + k * 0.08, 0.066, name, fontsize=9, va='center')
fig.text(0.655, 0.066, 'accuracy: TextSeal / SynthID\n(95% CI ±2.3 points)', fontsize=8, color=MUTED, va='center', linespacing=1.3)

for ext in ('png', 'svg'):
    fig.savefig(f'laundering.{ext}', bbox_inches='tight', facecolor='white')
print('saved')
