"""Slide figure: the same prompt answered with and without a green/red watermark, words highlighted."""
import hashlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

GREEN, RED, INK, MUTED = '#1f9d6b', '#d64545', '#1d2128', '#6b7280'
LGREEN, LRED = '#d6f0e5', '#f8dcdc'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'text.color': INK})

def is_green(prev, word, key='key-285'):
    """Green/red decided by hashing the secret key, the previous word and the candidate word."""
    return hashlib.sha256(f'{key}|{prev}|{word}'.encode()).digest()[0] < 128

normal = ('Volcanoes form where molten rock from deep inside the Earth rises to the surface. '
          'Over many eruptions, layers of lava and ash pile up into a mountain.').split()
# Watermarked answer: written by greedily preferring green words among natural alternatives.
options = [['Volcanoes'], ['form', 'appear', 'arise'], ['where', 'when'], ['molten', 'hot', 'melted'],
           ['rock', 'stone'], ['from'], ['deep', 'far'], ['inside', 'within', 'beneath'], ['the'], ['Earth', 'planet'],
           ['rises', 'pushes', 'moves', 'travels'], ['to', 'toward', 'up'], ['the'], ['surface.', 'crust.', 'top.'],
           ['Over', 'After', 'Through'], ['many', 'repeated', 'countless'], ['eruptions,', 'outbursts,'],
           ['layers', 'sheets'], ['of'], ['lava', 'magma'], ['and'], ['ash', 'dust', 'cinders'],
           ['pile', 'build', 'stack', 'gather'], ['up'], ['into', 'to', 'forming'], ['a'], ['mountain.', 'peak.', 'cone.']]
wm, prev = [], '<s>'
for opts in options:
    pick = next((w for w in opts if is_green(prev, w)), opts[0])
    wm.append(pick); prev = pick

def colors(words):
    prev, out = '<s>', []
    for w in words:
        out.append(is_green(prev, w)); prev = w
    return out

fig = plt.figure(figsize=(12, 4.3), dpi=200); fig.patch.set_facecolor('white')
ax = fig.add_axes([0.02, 0.02, 0.96, 0.96]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
ax.text(0.0, 0.96, 'Prompt: "How do volcanoes form?"', fontsize=13, color=MUTED, va='top')

def passage(y0, words, title):
    flags = colors(words); n_green = sum(flags)
    ax.text(0.0, y0, title, fontsize=13, weight='bold', va='top')
    x, y = 0.0, y0 - 0.09
    for w, g in zip(words, flags):
        width = 0.0098 * len(w) + 0.024
        if x + width > 1.0:
            x, y = 0.0, y - 0.085
        ax.add_patch(FancyBboxPatch((x, y - 0.032), width - 0.012, 0.064, boxstyle='round,pad=0.003,rounding_size=0.012',
                                    facecolor=LGREEN if g else LRED, edgecolor='none'))
        ax.text(x + (width - 0.012) / 2, y, w, ha='center', va='center', fontsize=11)
        x += width
    share = n_green / len(words)
    ax.text(0.0, y - 0.085, f'{n_green} / {len(words)} green  ({share:.0%})', fontsize=12, weight='bold',
            color=GREEN if share > 0.65 else INK, va='center')
    return share

s1 = passage(0.84, normal, 'Normal model')
s2 = passage(0.44, wm, 'Watermarked model')
ax.text(0.955, 0.96, 'green', fontsize=11, ha='right', va='top', color=INK,
        bbox=dict(boxstyle='round,pad=0.3', facecolor=LGREEN, edgecolor='none'))
ax.text(1.0, 0.96, 'red', fontsize=11, ha='right', va='top', color=INK,
        bbox=dict(boxstyle='round,pad=0.3', facecolor=LRED, edgecolor='none'))
ax.text(0.0, 0.06, 'Each word is green or red depending on the secret key and the previous word. '
        'The watermarked model prefers green words when it has a choice.', fontsize=10, color=MUTED, style='italic')
print(f'normal {s1:.0%}  watermarked {s2:.0%}')
for ext in ('png', 'svg'):
    fig.savefig(f'green_red_example.{ext}', bbox_inches='tight', facecolor='white')
