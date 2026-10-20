"""Single-slide figures for entropy laundering (text goes on the slide, not in the figure):
laundering_1  tokens of a real TextSeal trace with their entropies, top half highlighted
laundering_2  the same tokens marked trained (green) / dropped from the loss
laundering_loss  the masked SFT loss
Tokens: opening of TextSeal training trace #1 with Qwen3.5-4B entropies (data/trace0_tokens.json); a token is dropped in the
highest-entropy-50% run if its entropy is above the global median (0.39 nats)."""
import json
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

INK, MUTED, RULE, GREYF = '#1d2128', '#6b7280', '#e5e7eb', '#f3f4f6'
HOT, LHOT = '#d9480f', '#ffe3d3'      # high-entropy tokens
GRN, LGRN = '#15803d', '#dcfce7'      # tokens kept in the loss
plt.rcParams.update({'mathtext.fontset': 'cm', 'font.family': 'DejaVu Sans', 'text.color': INK})
toks = json.load(open('data/trace0_tokens.json'))[:18]
W = 9.0                    # canvas width in inches; axes units are inches
ROW, MAXROW = 0.78, 8.2    # row pitch and maximum row width


def canvas(height):
    fig = plt.figure(figsize=(W, height), dpi=220); fig.patch.set_facecolor('white')
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, height); ax.axis('off')
    return fig, ax


def save(fig, name):
    for ext in ('png', 'svg'):
        fig.savefig(f'{name}.{ext}', bbox_inches='tight', pad_inches=0.08, facecolor='white')


def width(t):
    return 0.115 * len(t.strip() or t) + 0.30


rows, cur = [], []
for tok in toks:                       # wrap into rows, then centre each row
    if cur and sum(width(t[0]) for t in cur) + width(tok[0]) > MAXROW:
        rows.append(cur); cur = []
    cur.append(tok)
rows.append(cur)


def draw_tokens(style, under):
    height = ROW * len(rows) + 0.2
    fig, ax = canvas(height)
    for r, row in enumerate(rows):
        y = height - 0.35 - r * ROW
        x = (W - sum(width(t[0]) for t in row)) / 2
        for t, h, dropped in row:
            w = width(t)
            fill, edge, col = style(dropped)
            ax.add_patch(FancyBboxPatch((x, y - 0.21), w - 0.07, 0.42, boxstyle='round,pad=0.01,rounding_size=0.07',
                                        facecolor=fill, edgecolor=edge, linewidth=1.3, clip_on=False))
            ax.text(x + (w - 0.07) / 2, y, t.strip() or t, ha='center', va='center', fontsize=13, color=col)
            text, colour = under(h, dropped)
            ax.text(x + (w - 0.07) / 2, y - 0.34, text, ha='center', va='center', fontsize=10, color=colour)
            x += w
    return fig


save(draw_tokens(lambda d: (LHOT, HOT, INK) if d else (GREYF, RULE, INK),
                 lambda h, d: (f'{h:.2f}', HOT if d else MUTED)), 'laundering_1')
save(draw_tokens(lambda d: ('white', '#c7ccd3', '#b8bec7') if d else (LGRN, GRN, INK),
                 lambda h, d: ('✗', MUTED) if d else ('✓', GRN)), 'laundering_2')

fig, ax = canvas(1.7)
ax.text(W / 2, 1.2, r'$\mathcal{L}(\theta) = -\frac{1}{|K|}\sum_{(i,t)\,\in\, K} \log p_\theta(x_{i,t} \mid x_{i,<t})$',
        fontsize=22, ha='center', va='center')
ax.text(W / 2, 0.3, r'$K = \{(i,t):\ t \in \mathrm{response}_i,\ \ H_{\mathrm{ref}}(i,t) \leq \tau\}$',
        fontsize=18, ha='center', va='center')
save(fig, 'laundering_loss')
print('saved')
