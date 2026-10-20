"""Slide figure for TextSeal: two keys, loop-breaking, blended + entropy-weighted detection."""
import math
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

INK, MUTED, RULE = '#1d2128', '#6b7280', '#e5e7eb'
GOLD, LGOLD = '#c98a00', '#fff3d1'          # key A
PURP, LPURP = '#6b4bd6', '#e9e3fb'          # key B
BLUE, LBLUE, GREY = '#2a78d6', '#dbe8fa', '#9aa1ac'
plt.rcParams.update({'mathtext.fontset': 'cm', 'font.family': 'DejaVu Sans', 'font.size': 11, 'text.color': INK,
                     'axes.edgecolor': RULE, 'axes.labelcolor': MUTED, 'xtick.color': MUTED, 'ytick.color': MUTED})
fig = plt.figure(figsize=(13, 6.2), dpi=200); fig.patch.set_facecolor('white')

def chip(ax, x, y, text, fill, edge, w=0.17, h=0.075, size=11, color=INK, weight='normal'):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle='round,pad=0.008,rounding_size=0.02',
                                facecolor=fill, edgecolor=edge, linewidth=1.2))
    ax.text(x, y, text, ha='center', va='center', fontsize=size, color=color, weight=weight)

# ---------- Panel 1: two keys ----------
ax1 = fig.add_axes([0.02, 0.05, 0.29, 0.86]); ax1.set_xlim(0, 1); ax1.set_ylim(0, 1); ax1.axis('off')
ax1.text(0.0, 1.0, '1  Two keys, chosen at random', fontsize=13, weight='bold')
chip(ax1, 0.22, 0.82, '"…let me check"', '#f3f4f6', RULE, w=0.42)
ax1.text(0.22, 0.71, 'at every token, flip a coin', ha='center', fontsize=9.5, color=MUTED)
chip(ax1, 0.12, 0.60, 'key A', LGOLD, '#e0b84a', w=0.20)
chip(ax1, 0.34, 0.60, 'key B', LPURP, PURP, w=0.20)
ax1.text(0.12, 0.52, '90%', ha='center', fontsize=11, weight='bold', color=GOLD)
ax1.text(0.34, 0.52, '10%  (α)', ha='center', fontsize=11, weight='bold', color=PURP)
ax1.annotate('', xy=(0.50, 0.72), xytext=(0.45, 0.66), arrowprops=dict(arrowstyle='-|>', color=MUTED, lw=1.3))
ax1.text(0.76, 0.92, 'key A', ha='center', fontsize=9, color=GOLD); ax1.text(0.89, 0.92, 'key B', ha='center', fontsize=9, color=PURP)
toks = ['the', 'whether', 'if', 'again']
RA = [0.95, 0.20, 0.41, 0.33]; RB = [0.12, 0.88, 0.47, 0.30]
for i, w in enumerate(toks):
    y = 0.84 - i * 0.09
    chip(ax1, 0.60, y, w, '#f3f4f6', RULE, w=0.16, size=10.5, h=0.065)
    chip(ax1, 0.76, y, f'{RA[i]:.2f}', LGOLD, '#e0b84a', w=0.11, size=10, h=0.065)
    chip(ax1, 0.89, y, f'{RB[i]:.2f}', LPURP, PURP, w=0.11, size=10, h=0.065)
ax1.text(0.5, 0.36, 'Each key on its own is a Gumbel-max watermark,\nso a random mix of the two is still\ndistortion-free.',
         ha='center', va='top', fontsize=10, color=MUTED, linespacing=1.4)

# ---------- Panel 2: loop breaking ----------
ax2 = fig.add_axes([0.36, 0.05, 0.28, 0.86]); ax2.set_xlim(0, 1); ax2.set_ylim(0, 1); ax2.axis('off')
ax2.text(0.0, 1.0, '2  Repeats no longer lock in', fontsize=13, weight='bold')
def line(y, label, parts, verdict, vcol):
    ax2.text(0.0, y + 0.07, label, fontsize=10.5, weight='bold')
    x = 0.0
    for txt, fill in parts:
        w = 0.022 * len(txt) + 0.03
        ax2.add_patch(FancyBboxPatch((x, y - 0.03), w - 0.012, 0.06, boxstyle='round,pad=0.003,rounding_size=0.012',
                                     facecolor=fill, edgecolor='none'))
        ax2.text(x + (w - 0.012) / 2, y, txt, ha='center', va='center', fontsize=9.5)
        x += w
    ax2.text(0.0, y - 0.075, verdict, fontsize=9.5, color=vcol, va='center')
plain = [('let me check', '#f3f4f6'), ('the', LGOLD), ('…', 'white'), ('let me check', '#f3f4f6'), ('the', LGOLD)]
seal = [('let me check', '#f3f4f6'), ('the', LGOLD), ('…', 'white'), ('let me check', '#f3f4f6'), ('whether', LPURP)]
line(0.78, 'One key (Gumbel-max)', plain, 'same window → same numbers → same word → loop', '#c0392b')
line(0.52, 'Two keys (TextSeal)', seal, '10% of the time key B is used → a different word can win', BLUE)
ax2.text(0.0, 0.24, 'Trade-off set by α:', fontsize=10.5, weight='bold')
ax2.text(0.0, 0.16, 'more key B  →  more variety, fewer loops,\nbut a weaker signal per token', fontsize=10, color=MUTED, va='center')
ax2.text(0.0, 0.05, 'In our traces: looping 16% (one key) vs 7% (α = 0.1)', fontsize=10, va='center')

# ---------- Panel 3: detection ----------
ax3 = fig.add_axes([0.69, 0.05, 0.30, 0.86]); ax3.set_xlim(0, 1); ax3.set_ylim(0, 1); ax3.axis('off')
ax3.text(0.0, 1.0, '3  Detect: blend, then weight', fontsize=13, weight='bold')
ax3.text(0.0, 0.90, 'The detector does not know which key was used,', fontsize=10, color=MUTED)
ax3.text(0.0, 0.85, 'so it scores each token under both:', fontsize=10, color=MUTED)
ax3.text(0.0, 0.76, r'$s_t = 0.9\,[-\ln(1-R^A_t)] + 0.1\,[-\ln(1-R^B_t)]$', fontsize=13.5)
ax3.text(0.0, 0.66, 'Confident tokens carry no signal, so weight', fontsize=10, color=MUTED)
ax3.text(0.0, 0.61, "each token by the model's uncertainty $H_t$:", fontsize=10, color=MUTED)
ax3.text(0.0, 0.52, r'$D = \sum_t w_t\, s_t, \qquad w_t \propto \sqrt{H_t}$', fontsize=14)
# Example tokens with weights
ex = [('Volcanoes', 0.05), ('form', 1.1), ('where', 0.9), ('molten', 1.4), ('rock', 0.2), ('rises', 1.0)]
x = 0.0; y = 0.36
ax3.text(0.0, 0.43, 'weight per token (darker = counts more)', fontsize=9, color=MUTED)
for wd, h in ex:
    wdt = 0.019 * len(wd) + 0.03
    a = 0.12 + 0.88 * min(h / 1.4, 1)
    ax3.add_patch(FancyBboxPatch((x, y - 0.03), wdt - 0.01, 0.06, boxstyle='round,pad=0.003,rounding_size=0.012',
                                 facecolor=BLUE, alpha=a, edgecolor='none'))
    ax3.text(x + (wdt - 0.01) / 2, y, wd, ha='center', va='center', fontsize=9, color='white' if a > 0.55 else INK)
    x += wdt
ax3.text(0.0, 0.25, '"Volcanoes" and "rock" were near-certain: weight ≈ 0.\n"molten", "form": real choices, so they count most.',
         fontsize=9.5, color=MUTED, va='center')
ax3.text(0.0, 0.10, 'p-value from a Gamma distribution matched to\nthe weights (same idea as Gumbel-max).', fontsize=10, weight='bold', va='center')

for ext in ('png', 'svg'):
    fig.savefig(f'textseal.{ext}', bbox_inches='tight', facecolor='white')
print('saved')
