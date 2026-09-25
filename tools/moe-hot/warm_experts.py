"""Pull the cold routed experts of a model into the page cache and drop the hot ones.

Run it after llama-server has loaded the model with LLAMA_MOE_HOT: the load reads the dense
weights and the hot experts, which go to VRAM. The cold experts are then read once (no
readahead, it would pull the neighboring hot experts in), the hot experts are dropped from
the page cache, and so are the further FILEs (e.g. an MTP head GGUF that is already in VRAM).

usage: warm_experts.py MODEL_GLOB HOT_LIST [FILE ...]
"""
import glob, os, re, sys, time
from gguf import GGUFReader

model_glob, hot_list = sys.argv[1], sys.argv[2]
hot = {}
for line in open(hot_list):
    if line.strip() and not line.startswith('#'):
        l, *es = map(int, line.split())
        hot[l] = set(es)
pat = re.compile(r'blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight')
PG = 4096
t0, total, dropped = time.time(), 0, 0
for f in sorted(glob.glob(model_glob)):
    ranges, hot_ranges = [], []
    for t in GGUFReader(f).tensors:
        m = pat.fullmatch(t.name)
        if not m:
            continue
        l, n_exp = int(m.group(1)), int(t.shape[-1])
        per = int(t.n_bytes) // n_exp
        for e in range(n_exp):
            (hot_ranges if e in hot.get(l, ()) else ranges).append((int(t.data_offset) + e * per, per))
    if not ranges:
        continue
    with open(f, 'rb', buffering=0) as fh:
        os.posix_fadvise(fh.fileno(), 0, 0, os.POSIX_FADV_RANDOM)
        for off, size in ranges:
            fh.seek(off)
            while size > 0:
                size -= len(fh.read(min(size, 64 << 20)))
        # edge pages are shared with cold neighbors and stay
        for off, size in hot_ranges:
            a0, a1 = -(-off // PG) * PG, (off + size) // PG * PG
            if a1 > a0:
                os.posix_fadvise(fh.fileno(), a0, a1 - a0, os.POSIX_FADV_DONTNEED)
                dropped += a1 - a0
    total += sum(s for _, s in ranges)
for f in sys.argv[3:]:
    with open(f, 'rb') as fh:
        os.posix_fadvise(fh.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
print(f'warmed {total/1e9:.1f} GB of cold experts in {time.time()-t0:.0f}s, dropped {dropped/1e9:.1f} GB of hot experts from the page cache')
