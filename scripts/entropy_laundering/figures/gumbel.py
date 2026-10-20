"""Slide figure for the Gumbel-max watermark: key numbers, pick, and the scoring/detection rule."""
import math
import random
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

INK, MUTED, RULE, GOLD, LGOLD = '#1d2128', '#6b7280', '#e5e7eb', '#c98a00', '#fff3d1'
BLUE, LBLUE, GREY = '#2a78d6', '#dbe8fa', '#9aa1ac'
plt.rcParams.update({'mathtext.fontset': 'cm', 'font.family': 'DejaVu Sans', 'font.size': 11, 'text.color': INK,
                     'axes.edgecolor': RULE, 'axes.labelcolor': MUTED, 'xtick.color': MUTED, 'ytick.color': MUTED})
fig = plt.figure(figsize=(13, 6.2), dpi=200); fig.patch.set_facecolor('white')

def chip(ax, x, y, text, fill, edge, w=0.17, h=0.075, size=11, color=INK, weight='normal'):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle='round,pad=0.008,rounding_size=0.02',
                                facecolor=fill, edgecolor=edge, linewidth=1.2))
    ax.text(x, y, text, ha='center', va='center', fontsize=size, color=color, weight=weight)

def gamma_sf(s, n):
    """P(Gamma(n, 1) >= s) for integer n."""
    term, total = 1.0, 1.0
    for k in range(1, n):
        term *= s / k; total += term
    return math.exp(-s) * total

toks = ['mat', 'rug', 'bed', 'chair']; p = [0.40, 0.30, 0.25, 0.05]; R = [0.31, 0.92, 0.55, 0.70]
score = [r ** (1 / q) for r, q in zip(R, p)]; win = max(range(4), key=lambda i: score[i])
rng = random.Random(0); wins = [0] * 4; N = 20000
for _ in range(N):
    rs = [rng.random() for _ in toks]
    wins[max(range(4), key=lambda i: rs[i] ** (1 / p[i]))] += 1
rates = [w / N for w in wins]

# ---------- Panel 1: key numbers ----------
ax1 = fig.add_axes([0.02, 0.05, 0.29, 0.86]); ax1.set_xlim(0, 1); ax1.set_ylim(0, 1); ax1.axis('off')
ax1.text(0.0, 1.0, '1  The key gives each word a number', fontsize=13, weight='bold')
chip(ax1, 0.22, 0.80, '"…sat on the"', '#f3f4f6', RULE, w=0.40)
chip(ax1, 0.22, 0.66, 'secret key', LGOLD, '#e0b84a', w=0.40)
ax1.text(0.22, 0.55, 'previous tokens + key → hash', ha='center', va='center', fontsize=9.5, color=MUTED)
ax1.annotate('', xy=(0.52, 0.73), xytext=(0.43, 0.73), arrowprops=dict(arrowstyle='-|>', color=MUTED, lw=1.4))
ax1.text(0.86, 0.93, r'secret number $R$', ha='center', fontsize=9.5, color=GOLD)
for i, (w, r) in enumerate(zip(toks, R)):
    y = 0.86 - i * 0.10
    chip(ax1, 0.64, y, w, '#f3f4f6', RULE, w=0.19)
    chip(ax1, 0.86, y, f'{r:.2f}', LGOLD, '#e0b84a', w=0.14, size=10.5)
ax1.text(0.5, 0.42, 'One random-looking number in [0, 1] per word,\nre-drawn at every position.\n\n'
         'Without the key the numbers look like noise;\nwith the key anyone can recompute them.',
         ha='center', va='top', fontsize=10, color=MUTED, linespacing=1.4)

# ---------- Panel 2: pick ----------
ax2 = fig.add_axes([0.36, 0.05, 0.28, 0.86]); ax2.set_xlim(0, 1); ax2.set_ylim(0, 1); ax2.axis('off')
ax2.text(0.0, 1.0, '2  Pick with probability + luck', fontsize=13, weight='bold')
cols = [0.02, 0.30, 0.52, 0.76]
for x, h in zip(cols, [r'word $w$', r'model $p_w$', r'key $R_w$', 'wins over\nmany keys']):
    ax2.text(x, 0.88, h, fontsize=9.5, color=MUTED, va='center')
for i, w in enumerate(toks):
    y = 0.79 - i * 0.085
    if i == win:
        ax2.add_patch(FancyBboxPatch((0.0, y - 0.035), 0.97, 0.07, boxstyle='round,pad=0,rounding_size=0.02', facecolor=LBLUE, edgecolor=BLUE, lw=1.2))
    ax2.text(cols[0], y, w + ('  ✓' if i == win else ''), fontsize=11, va='center', weight='bold' if i == win else 'normal', color=BLUE if i == win else INK)
    ax2.text(cols[1] + 0.06, y, f'{p[i]:.0%}', fontsize=11, va='center', ha='center')
    ax2.text(cols[2] + 0.06, y, f'{R[i]:.2f}', fontsize=11, va='center', ha='center', color=GOLD)
    ax2.text(cols[3] + 0.08, y, f'{rates[i]:.0%}', fontsize=11, va='center', ha='center', color=MUTED)
ax2.text(0.0, 0.38, r'pick  $w^\star = \arg\max_w \; R_w^{\,1/p_w}$', fontsize=15)
ax2.text(0.0, 0.26, '"rug" wins this time thanks to its high number.', fontsize=10, color=MUTED, va='center')
ax2.text(0.0, 0.18, 'Over many keys each word still wins at exactly its\nnormal rate, so output quality is unchanged.', fontsize=10, color=MUTED, va='center')
ax2.text(0.0, 0.07, 'But the winners tend to have high R:\nthat is the watermark.', fontsize=10.5, weight='bold', va='center')

# ---------- Panel 3: scoring ----------
ax3 = fig.add_axes([0.69, 0.05, 0.30, 0.86]); ax3.set_xlim(0, 1); ax3.set_ylim(0, 1); ax3.axis('off')
ax3.text(0.0, 1.0, '3  Detect: score every token', fontsize=13, weight='bold')
ax3.text(0.0, 0.91, r'per token:  $s_t = -\ln(1 - R_t)$', fontsize=14)
ax3.text(0.0, 0.84, r'whole text:  $S = \sum_t s_t$', fontsize=14)

# Inset: score curve
axc = fig.add_axes([0.715, 0.43, 0.25, 0.25])
xs = [i / 200 for i in range(0, 199)]
axc.plot(xs, [-math.log(1 - x) for x in xs], color=GOLD, lw=2)
for rv in (0.5, 0.9, 0.99):
    sv = -math.log(1 - rv); axc.plot([rv], [sv], 'o', color=GOLD, ms=6, mec='white', mew=1.5)
    if rv == 0.5:
        axc.text(rv + 0.03, sv - 0.55, f'R={rv} → {sv:.1f}', ha='left', fontsize=8.5, color=INK)
    else:
        axc.text(rv - 0.02, sv + 0.25, f'R={rv} → {sv:.1f}', ha='right', fontsize=8.5, color=INK)
axc.axhline(1, color=MUTED, lw=0.8, ls=(0, (3, 3)))
axc.text(0.02, 1.15, 'clean average = 1', fontsize=8.5, color=MUTED)
axc.set_xlim(0, 1); axc.set_ylim(0, 5.2); axc.set_xlabel('key number R of the token', fontsize=9); axc.set_ylabel('score s', fontsize=9)
axc.spines[['top', 'right']].set_visible(False); axc.tick_params(labelsize=8.5)

# Worked comparison for a 200-token text
T = 200
clean_S, wm_S = 200, 260
ax3.text(0.0, 0.305, f'Example: a {T}-token text', fontsize=10.5, color=MUTED)
for y, label, S, col in [(0.22, 'Normal text', clean_S, GREY), (0.10, 'Watermarked text', wm_S, BLUE)]:
    pv = gamma_sf(S, T)
    ax3.add_patch(FancyBboxPatch((0.0, y - 0.022), 0.05, 0.045, boxstyle='round,pad=0,rounding_size=0.015', facecolor=col, edgecolor='none'))
    ax3.text(0.08, y, f'{label}:  S = {S}  (expected ≈ {T})', fontsize=10.5, va='center')
    ax3.text(0.08, y - 0.05, f'p = {pv:.2g}' + ('  → watermarked' if pv < 0.01 else '  → no evidence'), fontsize=10, va='center',
             weight='bold', color=INK)
ax3.text(0.0, 0.0, 'p = chance a clean text scores this high (Gamma tail)', fontsize=8.5, color=MUTED, style='italic')

for ext in ('png', 'svg'):
    fig.savefig(f'gumbel.{ext}', bbox_inches='tight', facecolor='white')
print('rates', [round(r, 3) for r in rates], 'p clean', gamma_sf(clean_S, T), 'p wm', gamma_sf(wm_S, T))
