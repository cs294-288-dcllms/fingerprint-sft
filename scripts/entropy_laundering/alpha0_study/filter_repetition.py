"""Repetition filter from TextSeal (arXiv 2605.12456, App. E.4): sample ~200 evenly
spaced 100-char windows; reject a trace if any window occurs >= 3 times.
Responses of <= 200 characters pass automatically."""
import json, sys

WINDOW, SAMPLES, MAX_COUNT = 100, 200, 3


def is_repetitive(text: str) -> bool:
    if len(text) <= 2 * WINDOW:
        return False
    last = len(text) - WINDOW
    starts = sorted({round(i * last / (SAMPLES - 1)) for i in range(SAMPLES)})
    return any(text.count(text[s:s + WINDOW]) >= MAX_COUNT for s in starts)


def main(src: str, dst: str | None) -> None:
    kept = total = 0
    out = open(dst, 'w') if dst else None
    for line in open(src):
        row = json.loads(line)
        total += 1
        if is_repetitive(row['response']):
            continue
        kept += 1
        if out:
            out.write(line)
    print(json.dumps({'source': src, 'total': total, 'kept': kept, 'removed': total - kept,
                      'kept_rate': round(kept / total, 4)}))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
