"""Build a GGUF whose routed experts come from another quant of the same model.

usage: gguf_swap_experts.py BASE.gguf DONOR.gguf OUT.gguf
OUT gets the metadata and all tensors of BASE, except the routed experts (*_exps), which are copied byte for byte
from DONOR, with DONOR's types. Nothing is re-quantized. Works on one shard at a time: name OUT like BASE's shard
and link the other shards next to it. Example: the GSQ-RCO hybrid, IQ3_XXS trunk + Q2_0 experts (README.md).
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'gguf-py'))   # knows Q2_0
import gguf

base, donor, out = gguf.GGUFReader(sys.argv[1]), gguf.GGUFReader(sys.argv[2]), sys.argv[3]
donor_t = {t.name: t for t in donor.tensors}
w = gguf.GGUFWriter(out, arch=base.fields[gguf.Keys.General.ARCHITECTURE].contents(), endianess=base.endianess)
align = base.fields.get(gguf.Keys.General.ALIGNMENT)
if align is not None:
    w.data_alignment = align.contents()
for f in base.fields.values():
    if f.name == gguf.Keys.General.ARCHITECTURE or f.name.startswith('GGUF.'):
        continue
    vt = f.types[0]
    w.add_key_value(f.name, f.contents(), vt, sub_type=f.types[-1] if vt == gguf.GGUFValueType.ARRAY else None)
tensors = []
for t in base.tensors:
    if '_exps' in t.name:
        d = donor_t[t.name]
        assert list(d.shape) == list(t.shape), f'{t.name}: another shape in the donor'
        t = d
    tensors.append(t)
    w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_ti_data_to_file()
for t in tensors:
    w.write_tensor_data(t.data, tensor_endianess=base.endianess)
w.close()
print(f'{out}: {len(tensors)} tensors, experts from {sys.argv[2]}')
