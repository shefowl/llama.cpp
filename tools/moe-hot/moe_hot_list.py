"""Pick the routed experts to copy into VRAM for LLAMA_MOE_HOT.

Reads per-expert routing counts from llama-imatrix GGUF files (the *.counts tensors)
or from [n_layer, n_expert] .npy arrays saved from them, ranks experts by share of their
layer's calls per byte, and fills a byte budget.
Output: one line per layer, "<layer> <expert> <expert> ...".

usage: moe_hot_list.py OUT BUDGET_GB COUNTS.{gguf,npy} [...] --model GLOB [--layers 0-2]
"""
import argparse, glob, re
import numpy as np
from gguf import GGUFReader

ap = argparse.ArgumentParser()
ap.add_argument('out')
ap.add_argument('budget_gb', type=float)
ap.add_argument('imatrix', nargs='+')
ap.add_argument('--layers', default='', help='only these layers, e.g. 0-2 or 5,7')
ap.add_argument('--model', required=True, help='GGUF file or glob of the shards, e.g. "model-*-of-*.gguf"')
args = ap.parse_args()

# bytes per expert (gate+up+down) for each layer, from the model itself
pat = re.compile(r'blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight')
bpe, n_expert = {}, None
for f in glob.glob(args.model):
    for t in GGUFReader(f).tensors:
        m = pat.fullmatch(t.name)
        if m:
            n_expert = int(t.shape[-1])
            bpe[int(m.group(1))] = bpe.get(int(m.group(1)), 0) + int(t.n_bytes) / n_expert
L = max(bpe) + 1
bpe = np.array([bpe[l] for l in range(L)])

counts = np.zeros((L, n_expert))
for f in args.imatrix:
    if f.endswith('.npy'):
        counts += np.load(f)[:L, :n_expert]
        continue
    for t in GGUFReader(f).tensors:
        m = re.fullmatch(r'blk\.(\d+)\.ffn_gate_exps\.weight\.counts', t.name)
        if m:
            counts[int(m.group(1))] += np.asarray(t.data, dtype=np.float64).reshape(-1)[:n_expert]

allowed = np.zeros(L, bool)
if args.layers:
    for part in args.layers.split(','):
        a, _, b = part.partition('-')
        allowed[int(a):int(b or a) + 1] = True
else:
    allowed[:] = True

share = counts / np.maximum(counts.sum(1, keepdims=True), 1)
score = np.where(allowed[:, None], share / bpe[:, None], -1)
budget = args.budget_gb * 1e9
sel = np.zeros((L, n_expert), bool)
used = 0.0
for i in np.argsort(-score, axis=None, kind='stable'):
    l, e = divmod(int(i), n_expert)
    if score[l, e] <= 0 or sel[l].sum() >= n_expert - 1:
        continue
    if used + bpe[l] > budget:
        continue
    sel[l, e] = True
    used += bpe[l]

cover = (share * sel).sum(1)
with open(args.out, 'w') as fo:
    fo.write(f'# {used/1e9:.2f} GB, {int(sel.sum())} experts, calibration: {" ".join(args.imatrix)}\n')
    fo.write(f'# expected share of expert calls served from VRAM (calibration data): {cover.mean():.1%}\n')
    for l in range(L):
        if sel[l].any():
            fo.write(f'{l} ' + ' '.join(str(e) for e in np.flatnonzero(sel[l])) + '\n')
print(f'{args.out}: {used/1e9:.2f} GB, {int(sel.sum())} experts in {int(sel.any(1).sum())} layers, '
      f'calibration coverage {cover.mean():.1%}')
