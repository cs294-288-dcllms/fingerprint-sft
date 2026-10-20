"""Slide figure for SynthID-Text: candidates, key-scored knockout tournament, mean-g detection."""
import math
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

INK, MUTED, RULE = '#1d2128', '#6b7280', '#e5e7eb'
GOLD, LGOLD = '#c98a00', '#fff3d1'        # g = 1
GREYF = '#f3f4f6'                         # g = 0 / neutral
BLUE, LBLUE, GREY = '#2a78d6', '#dbe8fa', '#9aa1ac'
plt.rcParams.update({'mathtext.fontset': 'cm', 'font.family': 'DejaVu Sans', 'font.size': 11, 'text.color': INK,
                     'axes.edgecolor': RULE, 'axes.labelcolor': MUTED, 'xtick.color': MUTED, 'ytick.color': MUTED})
fig = plt.figure(figsize=(13, 6.2), dpi=200); fig.patch.set_facecolor('white')

def chip(ax, x, y, text, fill, edge, w=0.17, h=0.07, size=11, color=INK, weight='normal', lw=1.2):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle='round,pad=0.008,rounding_size=0.02',
                                facecolor=fill, edgecolor=edge, linewidth=lw))
    ax.text(x, y, text, ha='center', va='center', fontsize=size, color=color, weight=weight)

def gchip(ax, x, y, g):
    chip(ax, x, y, f'g={g}', LGOLD if g else GREYF, '#e0b84a' if g else RULE, w=0.10, h=0.055, size=9,
         color=GOLD if g else MUTED, weight='bold' if g else 'normal')

def binom_sf(k, n):
    """P(Binomial(n, 0.5) >= k), via log-space summation."""
    logs = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * math.log(2) for i in range(k, n + 1)]
    m = max(logs)
    return math.exp(m) * sum(math.exp(l - m) for l in logs)

# ---------- Panel 1: draw candidates ----------
ax1 = fig.add_axes([0.02, 0.05, 0.27, 0.86]); ax1.set_xlim(0, 1); ax1.set_ylim(0, 1); ax1.axis('off')
ax1.text(0.0, 1.0, '1  Draw candidates from the model', fontsize=13, weight='bold')
chip(ax1, 0.25, 0.86, '"…sat on the"', GREYF, RULE, w=0.46)
toks = ['mat', 'rug', 'bed', 'chair']; p = [0.40, 0.30, 0.25, 0.05]
ax1.text(0.0, 0.75, 'model probabilities', fontsize=9.5, color=MUTED)
for i, (w, q) in enumerate(zip(toks, p)):
    y = 0.68 - i * 0.065
    ax1.text(0.0, y, w, fontsize=10.5, va='center')
    ax1.add_patch(FancyBboxPatch((0.22, y - 0.018), 0.55 * q / 0.4, 0.036, boxstyle='round,pad=0,rounding_size=0.008',
                                 facecolor=GREY, edgecolor='none'))
    ax1.text(0.24 + 0.55 * q / 0.4, y, f'{q:.0%}', fontsize=9.5, va='center', color=MUTED)
ax1.text(0.0, 0.35, 'sample 4 times (normal sampling):', fontsize=9.5, color=MUTED)
for i, w in enumerate(['mat', 'rug', 'mat', 'bed']):
    chip(ax1, 0.11 + i * 0.22, 0.28, w, GREYF, RULE, w=0.19)
ax1.text(0.5, 0.17, 'Likely words show up more often.\nThe key plays no part yet.', ha='center', va='top',
         fontsize=10, color=MUTED, linespacing=1.4)

# ---------- Panel 2: tournament ----------
ax2 = fig.add_axes([0.33, 0.05, 0.34, 0.86]); ax2.set_xlim(0, 1); ax2.set_ylim(0, 1); ax2.axis('off')
ax2.text(0.0, 1.0, '2  Knockout tournament scored by the key', fontsize=13, weight='bold')
ax2.text(0.0, 0.93, 'Each round the key gives every word g = 0 or 1.\nHigher g wins; ties are broken by a coin.',
         fontsize=9.5, color=MUTED, va='center', linespacing=1.4)
ax2.text(0.13, 0.86, 'candidates', ha='center', fontsize=9.5, color=MUTED)
ax2.text(0.50, 0.86, 'round 1', ha='center', fontsize=9.5, color=MUTED)
ax2.text(0.84, 0.86, 'round 2', ha='center', fontsize=9.5, color=MUTED)
leaves = [('mat', 1), ('rug', 0), ('mat', 1), ('bed', 1)]      # round-1 g-values (same word -> same g)
ys = [0.77, 0.66, 0.50, 0.39]
for (w, g), y in zip(leaves, ys):
    chip(ax2, 0.09, y, w, GREYF, RULE, w=0.14)
    gchip(ax2, 0.24, y, g)
mid = [('mat', 0.715, 'mat beats rug (1 > 0)'), ('bed', 0.445, 'tie (1 = 1) → coin: bed')]
r2g = {'mat': 0, 'bed': 1}
for (w, y, note), (ya, yb) in zip(mid, [(ys[0], ys[1]), (ys[2], ys[3])]):
    for yy in (ya, yb):
        ax2.plot([0.30, 0.36, 0.36], [yy, yy, y], color=RULE, lw=1.4)
    ax2.plot([0.36, 0.42], [y, y], color=RULE, lw=1.4)
    chip(ax2, 0.49, y, w, GREYF, RULE, w=0.14)
    gchip(ax2, 0.64, y, r2g[w])
    ax2.text(0.42, y - 0.075, note, fontsize=8.5, color=MUTED)
for yy in (0.715, 0.445):
    ax2.plot([0.70, 0.75, 0.75], [yy, yy, 0.58], color=RULE, lw=1.4)
ax2.plot([0.75, 0.79], [0.58, 0.58], color=RULE, lw=1.4)
chip(ax2, 0.87, 0.58, 'bed  ✓', LBLUE, BLUE, w=0.16, color=BLUE, weight='bold')
ax2.text(0.87, 0.505, 'bed beats mat\n(1 > 0): output', ha='center', va='top', fontsize=8.5, color=MUTED)
ax2.text(0.0, 0.24, 'Over many keys, each word still wins at its normal rate:', fontsize=10, color=MUTED)
ax2.text(0.0, 0.18, 'distortion-free on average, and still random (no loops).', fontsize=10, color=MUTED)
ax2.text(0.0, 0.08, 'But winners tend to have g = 1: that is the watermark.', fontsize=10.5, weight='bold')
ax2.text(0.0, 0.01, 'We use 30 rounds and a 5-word context\n(computed exactly, not by drawing 2³⁰ samples).',
         fontsize=8.5, color=MUTED, style='italic')

# ---------- Panel 3: detection ----------
ax3 = fig.add_axes([0.71, 0.05, 0.28, 0.86]); ax3.set_xlim(0, 1); ax3.set_ylim(0, 1); ax3.axis('off')
ax3.text(0.0, 1.0, '3  Detect: average the g-values', fontsize=13, weight='bold')
ax3.text(0.0, 0.91, 'For every token in the text, recompute its g in', fontsize=10, color=MUTED)
ax3.text(0.0, 0.86, 'every round from the key and previous words:', fontsize=10, color=MUTED)
ax3.text(0.0, 0.76, r'$\bar g = \mathrm{average}\,\{\, g_\ell(x_t) \,\}_{\,t \leq T,\ \ell \leq m}$', fontsize=15)
ax3.text(0.0, 0.66, 'T tokens, m rounds.  Clean text: each g is a fair coin.', fontsize=9.5, color=MUTED)

# mean-g gauges for a 200-token text with 30 rounds
T, m = 200, 30; n = T * m
cases = [('Normal text', 0.500, GREY), ('Watermarked text', 0.520, BLUE)]
ax3.text(0.0, 0.57, f'Example: {T} tokens × {m} rounds = {n:,} g-values', fontsize=10, color=MUTED)
x0, x1, lo, hi = 0.05, 0.95, 0.47, 0.53
def gx(v): return x0 + (v - lo) / (hi - lo) * (x1 - x0)
for i, (label, gbar, col) in enumerate(cases):
    y = 0.43 - i * 0.21
    ax3.add_patch(FancyBboxPatch((x0, y - 0.012), x1 - x0, 0.024, boxstyle='round,pad=0,rounding_size=0.01',
                                 facecolor=GREYF, edgecolor='none'))
    ax3.plot([gx(0.5), gx(0.5)], [y - 0.012, y + 0.03], color=MUTED, lw=1, ls=(0, (2, 2)))
    ax3.plot([gx(gbar)], [y], 'o', ms=11, color=col, mec='white', mew=2)
    k = round(gbar * n)
    pv = binom_sf(k, n)
    ax3.text(0.0, y + 0.06, f'{label}:  ḡ = {gbar:.3f}', fontsize=10.5, va='center')
    ax3.text(0.0, y - 0.085, f'p = {pv:.2g}' + ('  → watermarked' if pv < 0.01 else '  → no evidence'),
             fontsize=10, va='center', weight='bold')
for i in range(2):
    y = 0.43 - i * 0.21
    for v in (0.47, 0.48, 0.49, 0.50, 0.51, 0.52, 0.53):
        ax3.plot([gx(v), gx(v)], [y - 0.012, y - 0.022], color=GREY, lw=0.8)
        ax3.text(gx(v), y - 0.035, f'{v:.2f}', ha='center', va='top', fontsize=7, color=MUTED)
ax3.text(gx(0.5) + 0.02, 0.43 + 0.03, 'clean 0.5', ha='left', fontsize=8.5, color=MUTED)
ax3.text(0.0, 0.07, 'p = chance a clean text averages this high\n(binomial tail; ours also weights by entropy)',
         fontsize=8.5, color=MUTED, style='italic', va='center')

for ext in ('png', 'svg'):
    fig.savefig(f'synthid.{ext}', bbox_inches='tight', facecolor='white')
print('p wm', binom_sf(round(0.52 * n), n), 'p clean', binom_sf(round(0.5 * n), n))
