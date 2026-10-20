"""Three-panel slide figure for ADFP: split, proxy-aimed boost, detection in the student."""
import math
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

GREEN, RED, INK, MUTED, RULE = '#1f9d6b', '#d64545', '#1d2128', '#6b7280', '#e5e7eb'
LGREEN, LRED, BLUE, LBLUE = '#d6f0e5', '#f8dcdc', '#2a78d6', '#dbe8fa'
plt.rcParams.update({'mathtext.fontset': 'cm', 'font.family': 'DejaVu Sans', 'font.size': 11, 'text.color': INK,
                     'axes.edgecolor': RULE, 'axes.labelcolor': MUTED, 'xtick.color': MUTED, 'ytick.color': MUTED})
fig = plt.figure(figsize=(13, 4.2), dpi=200); fig.patch.set_facecolor('white')

def chip(ax, x, y, text, fill, edge, w=0.17, h=0.11, size=11, color=INK, weight='normal'):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle='round,pad=0.008,rounding_size=0.025',
                                facecolor=fill, edgecolor=edge, linewidth=1.2))
    ax.text(x, y, text, ha='center', va='center', fontsize=size, color=color, weight=weight)

# Toy next-token distribution: mat/chair green, rug/bed red; proxy agrees with the teacher.
toks = ['mat', 'chair', 'rug', 'bed']; green = [1, 1, 0, 0]
p = [0.40, 0.05, 0.30, 0.25]
L = sum(q for q, g in zip(p, green) if g)                     # proxy's total green probability
delta = [q * (g - L) for q, g in zip(p, green)]               # ADFP nudge: q_t * (1[t in S] - L)
lam = 3.0
logits = [math.log(q) + lam * d for q, d in zip(p, delta)]
Z = sum(math.exp(z) for z in logits); after = [math.exp(z) / Z for z in logits]

# Panel 1: split + proxy
ax1 = fig.add_axes([0.02, 0.05, 0.30, 0.80]); ax1.set_xlim(0, 1); ax1.set_ylim(0, 1); ax1.axis('off')
ax1.text(0.0, 1.02, '1  Split, and ask a proxy student', fontsize=13, weight='bold')
chip(ax1, 0.22, 0.84, '"…sat on the"', '#f3f4f6', RULE, w=0.40, h=0.11)
chip(ax1, 0.22, 0.67, 'secret key', '#fff7e0', '#e0b84a', w=0.40, h=0.11)
chip(ax1, 0.22, 0.42, 'proxy model', LBLUE, BLUE, w=0.40, h=0.11)
ax1.text(0.22, 0.27, 'stand-in for the\nunknown student', ha='center', va='center', fontsize=9.5, color=MUTED)
ax1.annotate('', xy=(0.52, 0.76), xytext=(0.43, 0.76), arrowprops=dict(arrowstyle='-|>', color=MUTED, lw=1.4))
for i, (w, g, q) in enumerate(zip(toks, green, p)):
    y = 0.88 - i * 0.15
    chip(ax1, 0.66, y, w, LGREEN if g else LRED, GREEN if g else RED, w=0.19)
    ax1.text(0.84, y, f'{q:.0%}', ha='left', va='center', fontsize=10.5, color=BLUE)
ax1.text(0.84, 0.97, 'proxy says', ha='left', va='center', fontsize=9, color=BLUE)
ax1.text(0.75, 0.16, 'colour: key + previous tokens\nblue %: how likely a student\nfinds each token', ha='center', fontsize=9.5, color=MUTED)

# Panel 2: aimed boost
ax2 = fig.add_axes([0.40, 0.17, 0.25, 0.60])
fig.text(0.375, 0.05 + 0.80 * 1.02, '2  Boost where the student learns', fontsize=13, weight='bold', va='baseline')
xs = range(len(toks)); bw = 0.36
for x, b, a, g in zip(xs, p, after, green):
    c = GREEN if g else RED
    ax2.bar(x - bw / 2 - 0.01, b, bw, color=c, alpha=0.28, edgecolor='none')
    ax2.bar(x + bw / 2 + 0.01, a, bw, color=c, edgecolor='none')
    ax2.text(x + bw / 2 + 0.01, a + 0.015, f'{a:.0%}', ha='center', fontsize=9, color=INK)
ax2.set_xticks(list(xs)); ax2.set_xticklabels(toks, fontsize=11, color=INK)
ax2.set_yticks([0, 0.2, 0.4, 0.6, 0.8]); ax2.set_yticklabels(['0', '20%', '40%', '60%', '80%'], fontsize=9)
ax2.set_ylim(0, 0.80); ax2.spines[['top', 'right']].set_visible(False); ax2.grid(axis='y', color=RULE, lw=0.8); ax2.set_axisbelow(True)
ax2.text(0.98, 1.04, 'light = original   solid = ADFP', transform=ax2.transAxes, ha='right', fontsize=9, color=MUTED)
ax2.set_ylabel('next-token probability', fontsize=9.5)
ax2.annotate('likely green:\nbig boost', xy=(0.38, after[0] - 0.04), xytext=(0.62, 0.62), fontsize=9, color=GREEN,
             arrowprops=dict(arrowstyle='-', color=GREEN, lw=0.8))
ax2.annotate('unlikely green:\nbarely moves', xy=(1.19, after[1] + 0.01), xytext=(1.05, 0.36), fontsize=9, color=GREEN,
             arrowprops=dict(arrowstyle='-', color=GREEN, lw=0.8))

fig.text(0.525, -0.06, r'$\tilde z_t = z_t + \lambda\, q_t \left(\mathbb{1}[t \in \mathrm{green}] \;-\; \sum_{s \in \mathrm{green}} q_s\right)$',
         ha='center', fontsize=17, color=INK)
fig.text(0.525, -0.165, 'z = teacher logits,  q = proxy probabilities,  λ = strength', ha='center', fontsize=9.5, color=MUTED)

# Panel 3: detect in the student
ax3 = fig.add_axes([0.70, 0.05, 0.29, 0.80]); ax3.set_xlim(0, 1); ax3.set_ylim(0, 1); ax3.axis('off')
ax3.text(0.0, 1.02, '3  Detect it in the student', fontsize=13, weight='bold')
ax3.text(0.0, 0.86, 'Average probability the student puts on green tokens', fontsize=10, color=MUTED)
def gauge(y, label, val, color):
    lo, hi = 0.495, 0.505
    ax3.text(0.0, y + 0.075, label, fontsize=10.5)
    ax3.add_patch(FancyBboxPatch((0.0, y - 0.03), 1.0, 0.06, boxstyle='round,pad=0,rounding_size=0.03', facecolor='#f3f4f6', edgecolor='none'))
    xm = (0.5 - lo) / (hi - lo); xv = (val - lo) / (hi - lo)
    ax3.plot([xm, xm], [y - 0.05, y + 0.05], color=MUTED, lw=1)
    ax3.plot([xv], [y], marker='o', ms=11, color=color, mec='white', mew=2)
    ax3.text(xv, y - 0.09, f'{val:.1%}', ha='center', fontsize=10, weight='bold', color=INK)
gauge(0.62, 'Student trained on normal outputs', 0.500, MUTED)
gauge(0.30, 'Student trained on ADFP outputs', 0.502, GREEN)
ax3.text(0.5, 0.80, 'tick = 50% (chance)', ha='center', fontsize=8.5, color=MUTED)
ax3.text(0.0, 0.08, 'A 0.2-point shift is invisible per token, but over\n~600,000 positions it gives p ≈ 0.004.',
         fontsize=10, weight='bold')

for ext in ('png', 'svg'):
    fig.savefig(f'adfp.{ext}', bbox_inches='tight', facecolor='white')
print('after:', [round(a, 3) for a in after], 'sum', round(sum(after), 3))
