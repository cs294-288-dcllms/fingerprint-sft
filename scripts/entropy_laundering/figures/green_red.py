"""Three-panel slide figure: green/red list watermarking (split, boost, detect)."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

GREEN, RED, INK, MUTED, RULE = '#1f9d6b', '#d64545', '#1d2128', '#6b7280', '#e5e7eb'
LGREEN, LRED = '#d6f0e5', '#f8dcdc'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'text.color': INK,
                     'axes.edgecolor': RULE, 'axes.labelcolor': MUTED, 'xtick.color': MUTED, 'ytick.color': MUTED})
fig = plt.figure(figsize=(13, 4.2), dpi=200)
fig.patch.set_facecolor('white')

def chip(ax, x, y, text, fill, edge, w=0.17, h=0.11, size=11, color=INK, weight='normal'):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle='round,pad=0.008,rounding_size=0.025',
                                facecolor=fill, edgecolor=edge, linewidth=1.2))
    ax.text(x, y, text, ha='center', va='center', fontsize=size, color=color, weight=weight)

# Panel 1: split
ax1 = fig.add_axes([0.02, 0.05, 0.30, 0.80]); ax1.set_xlim(0, 1); ax1.set_ylim(0, 1); ax1.axis('off')
ax1.text(0.0, 1.02, '1  Split the vocabulary', fontsize=13, weight='bold')
chip(ax1, 0.22, 0.80, '"…sat on the"', '#f3f4f6', RULE, w=0.40, h=0.12)
chip(ax1, 0.22, 0.60, 'secret key', '#fff7e0', '#e0b84a', w=0.40, h=0.12)
ax1.text(0.22, 0.45, 'previous tokens + key\n→ hash', ha='center', va='center', fontsize=9.5, color=MUTED)
ax1.annotate('', xy=(0.52, 0.70), xytext=(0.43, 0.70), arrowprops=dict(arrowstyle='-|>', color=MUTED, lw=1.4))
vocab = [('mat', 1), ('rug', 0), ('floor', 1), ('bed', 0), ('chair', 1), ('sofa', 0), ('grass', 0), ('step', 1)]
for i, (w, g) in enumerate(vocab):
    x = 0.64 + (i % 2) * 0.22; y = 0.88 - (i // 2) * 0.15
    chip(ax1, x, y, w, LGREEN if g else LRED, GREEN if g else RED, w=0.19)
ax1.text(0.75, 0.22, 'half green, half red\n(re-drawn at every position)', ha='center', fontsize=9.5, color=MUTED)

# Panel 2: boost
ax2 = fig.add_axes([0.40, 0.17, 0.25, 0.60])
fig.text(0.375, 0.05 + 0.80 * 1.02, '2  Boost the green tokens', fontsize=13, weight='bold', va='baseline')
toks = ['mat', 'rug', 'floor', 'bed']; green = [1, 0, 1, 0]
before = [0.40, 0.30, 0.18, 0.12]
raw = [b * (1.6 if g else 1.0) for b, g in zip(before, green)]
after = [r / sum(raw) for r in raw]
xs = range(len(toks)); bw = 0.36
for x, b, a, g in zip(xs, before, after, green):
    c = GREEN if g else RED
    ax2.bar(x - bw / 2 - 0.01, b, bw, color=c, alpha=0.28, edgecolor='none')
    ax2.bar(x + bw / 2 + 0.01, a, bw, color=c, edgecolor='none')
    ax2.text(x + bw / 2 + 0.01, a + 0.015, f'{a:.0%}', ha='center', fontsize=9, color=INK)
ax2.set_xticks(list(xs)); ax2.set_xticklabels(toks, fontsize=11, color=INK)
ax2.set_yticks([0, 0.2, 0.4, 0.6]); ax2.set_yticklabels(['0', '20%', '40%', '60%'], fontsize=9)
ax2.set_ylim(0, 0.62); ax2.spines[['top', 'right']].set_visible(False); ax2.grid(axis='y', color=RULE, lw=0.8); ax2.set_axisbelow(True)
ax2.text(0.98, 1.04, 'light = original   solid = watermarked', transform=ax2.transAxes, ha='right', fontsize=9, color=MUTED)
ax2.set_ylabel('next-token probability', fontsize=9.5)

# Panel 3: detect
ax3 = fig.add_axes([0.70, 0.05, 0.29, 0.80]); ax3.set_xlim(0, 1); ax3.set_ylim(0, 1); ax3.axis('off')
ax3.text(0.0, 1.02, '3  Detect with the key', fontsize=13, weight='bold')
def sentence(y, words, label, verdict):
    ax3.text(0.0, y + 0.115, label, fontsize=10, color=MUTED)
    x = 0.0
    for w, g in words:
        width = 0.0185 * len(w) + 0.03
        if x + width > 1.0:
            break
        ax3.add_patch(FancyBboxPatch((x, y - 0.045), width - 0.012, 0.09, boxstyle='round,pad=0.004,rounding_size=0.015',
                                     facecolor=LGREEN if g else LRED, edgecolor='none'))
        ax3.text(x + (width - 0.012) / 2, y, w, ha='center', va='center', fontsize=8.8)
        x += width
    ax3.text(0.0, y - 0.12, verdict, fontsize=10.5, weight='bold')
wm = [('The', 1), ('cat', 1), ('sat', 0), ('on', 1), ('the', 1), ('mat', 1), ('and', 0), ('slept', 1), ('all', 1), ('day', 1)]
hu = [('The', 1), ('cat', 0), ('sat', 0), ('on', 1), ('the', 0), ('rug', 1), ('and', 1), ('slept', 0), ('all', 0), ('day', 1)]
sentence(0.72, wm, 'Watermarked text', '8 / 10 green  →  well above 50%  →  watermarked')
sentence(0.30, hu, 'Normal text', '5 / 10 green  →  about 50%  →  not watermarked')
ax3.text(0.0, 0.02, 'Real tests count thousands of tokens and report a p-value.', fontsize=9, color=MUTED, style='italic')

for ext in ('png', 'svg'):
    fig.savefig(f'green_red_watermark.{ext}', bbox_inches='tight', facecolor='white')
print('saved')
