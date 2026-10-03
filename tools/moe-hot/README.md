# Hot MoE experts in VRAM for Vulkan

A personal fork of llama.cpp (base: ggml-org/llama.cpp `64709a0`, 2026-09-12, branch `vulkan-moe-hot`) for MoE models that do not fit in VRAM and run with the routed experts on the CPU (`-ot exps=CPU`), on a Vulkan GPU. Stock llama.cpp puts either whole layers of experts on the GPU (`--fit`) or none. This fork keeps copies of the often used experts of every layer in VRAM, computes only the rest on the CPU, and removes most of the synchronization cost of the per-layer CPU/GPU round trips.

Tested on one machine only: RX 7900 XTX 24 GB (RADV, Linux), Ryzen 7 7700X (8 cores), 64 GB DDR5, NVMe. Not affiliated with upstream, not meant as a PR.

## Results

Model: Qwen3.8-Flash-Next ([original GGUF quants by AtomicChat](https://huggingface.co/AtomicChat/Qwen3.8-Flash-Next-GGUF); `qwen4exp`: 48 layers, 512 routed experts, 10 active). The numbers are for the `mainline` files (33 shards, 94.5 GB) of [Navin-Models/Qwen3.8-Flash-Next-Uncensored-AD-4.27-GGUF](https://huggingface.co/Navin-Models/Qwen3.8-Flash-Next-Uncensored-AD-4.27-GGUF), an uncensored variant of the AD-4.27 quant: routed experts 50.6 GB (gate/up IQ2_S or IQ3_S, down IQ4_NL), a 38.4 GB n-gram embedding table (Q5_1, read from disk on demand), ~5 GB of dense Q8_0 weights. 12.5 GB of experts are hot copies in VRAM; together with the MTP head (`mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf` from [drluoto/Qwen3.8-Flash-Next-MTP-GGUF](https://huggingface.co/drluoto/Qwen3.8-Flash-Next-MTP-GGUF); drluoto now recommends the Q5_K frspec-65k file there, not measured here), 32k context (q4_0 KV), `-ub 1024` and the desktop this fills the 24 GB card.

Decode t/s: MTP self-speculation (3 draft tokens), greedy, 400 tokens, mean of 3 prompts per domain, second pass (the first one pages the model in). All rows measured the same day with the same prompts; run-to-run noise is 2-5%. The first 3 rows ran before the page cache changes (below), which mostly matter under memory pressure.

| | code | science | English prose | Russian |
|---|---|---|---|---|
| hot experts + MTP, stock scheduler, GPU power level `auto` | 29.3 | 29.6 | 19.8 | 20.8 |
| + scheduler: one GPU wait per split | 33.0 | 33.4 | 21.9 | 23.8 |
| + GPU power level `high` | 39.1 | 41.0 | 26.3 | 30.4 |
| + one submit per small graph, page cache fixes | 45.2 | 46.3 | 28.2 | 32.7 |
| + cold experts locked in RAM | **46.1** | **46.3** | **28.7** | **33.2** |

Prefill of a 2.5k token prompt: 239 t/s. Without MTP (GPU `high`, one submit per small graph): 27.2 / 28.3 / 21.8 / 25.6.

For reference, measured the day before (one prompt per domain, GPU `auto`, not re-measured with `high`): stock `--fit` 23.0 / 23.3 / 23.2 / 21.2 with a prefill of 50 t/s, and `--fit` + MTP 26.4 / 28.3 / 25.2 / 21.7.

Hot experts alone help little without speculative decoding: most of the gain comes from the combination. When most experts of a layer are in VRAM, verifying 4 draft tokens costs little more than decoding 1.

### GSQ-RCO hybrid: per-expert precision (2026-10-03)

The [GSQ-RCO quants by IST-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) of the same model have smaller experts: 1.75 MB per expert in IQ3_XXS and 1.44 MB in the Q2_0 tier, against 2.06 MB in AD-4.27, so more of them fit in VRAM. The Q2_0 tier also cheapens the dense part (shared experts down to Q2_0, attention to Q3_K), and that is where most of its quality loss is. The hybrid takes the dense weights and the hot experts from IQ3_XXS and the cold experts from Q2_0:

- `gguf_swap_experts.py` writes one GGUF with the IQ3_XXS trunk and the Q2_0 experts (tensors copied, nothing re-quantized);
- `LLAMA_MOE_HOT_SRC=<IQ3_XXS shard 1>` makes the hot copies come from IQ3_XXS.

The merged file: [shefowl/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-Hybrid-GGUF](https://huggingface.co/shefowl/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-Hybrid-GGUF), built from [SC117's abliterated GSQ-RCO GGUFs](https://huggingface.co/SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF).

Decode t/s, a different bench from the table above (1 prompt per language or topic, 400 tokens, second pass), GPU `high`, cold experts locked, MTP with 3 draft tokens, about 1.9 GB of VRAM left, all rows in one session. Run to run the numbers move by about 5%, between sessions by up to 10% (the desktop's VRAM use changes the hot list):

| | hot experts | code | science | English prose | Cyrillic (Russian) | Chinese | locked in RAM |
|---|---|---|---|---|---|---|---|
| hybrid | 12.6 GB, 71.6% of calls | **54.5** | **40.1** | **25.5** | **27.2** | **23.1** | 22.4 GiB |
| GSQ-RCO IQ3_XXS | 12.0 GB, 69.9% | 48.6 | 34.4 | 19.6 | 23.3 | 18.4 | 28.9 GiB |
| GSQ-RCO Q2_0 tier | 12.9 GB, 80.1% | 65.8 | 47.5 | 29.8 | 30.5 | 27.0 | 19.7 GiB |
| AD-4.27 | 11.5 GB, 63.4% | 43.8 | 28.6 | 17.7 | 20.2 | 17.1 | 36.5 GiB |

The Q2_0 tier is the fastest, but at +4.3% perplexity against IQ3_XXS (below). At equal free VRAM the hybrid holds 0.6-0.9 GB more hot experts than IQ3_XXS: the prefill compute buffer holds the cold tensors of one layer, and Q2_0 ones are smaller.

Quality against IQ3_XXS (the only difference is the cold experts): perplexity ratio 0.998 ± 0.008 (the Q2_0 tier 1.043 ± 0.011) on 20k tokens of mixed English, C++ and Russian text; GSM-Plus 78/100 on both, with the same answer on every task; CRUXEval-O 94 vs 97 out of 100 (five character-level slips by the hybrid, not significant at this size). The mean KLD is 0.22 with the same top-1 token in 84% of positions, so the cold Q2_0 experts do move the distribution; with the hot/cold split alone (both IQ3_XXS) the KLD is 0.000.

## What changed

- **`LLAMA_MOE_HOT=<list>`** (`src/llama-model.cpp`, `src/llama-graph.cpp`). The list names the hot experts per layer. At load they are copied into VRAM. `build_moe_ffn` splits each MoE layer into a hot part (GPU, the copies) and a cold part (CPU, the mmap), with four small CPU ops that remap the router ids and weights. Hot fillers are distinct unused slots. Cold ids are `-1` for small batches (the CPU writes a zero row). For batches of 32+ tokens the scheduler may run the cold part on the GPU; there cold ids are distinct zero-weight fillers.
- **`LLAMA_MOE_HOT_ADAPT=<tokens>`**: swaps experts between hot and cold at run time from decayed use counts. In a long one-topic session this gave +21%; when the topic changes every few prompts, about -5%.
- **Scheduler** (`ggml/src/ggml-backend.cpp`). Without events, the split backend is synchronized once per split: a sync before every input only flushed the async copies of the previous inputs. Device-to-host inputs are queued with `get_tensor_async` and waited for once. Before this, each layer had about 6 GPU waits, now 1.
- **`GGML_VK_SMALL_GRAPH_GFLOPS=20`** (`ggml-vulkan.cpp`). A per-layer decode graph is submitted once, instead of about 6 flops-based chunks plus the `almost_ready` fence. Prefill graphs stay above the threshold.
- **Page cache.** After the load, the pages of the weights that went to the GPU and of the hot copies are dropped (`MADV_PAGEOUT`). `warm_experts.py` reads the cold experts without readahead: readahead, and btrfs compressed extents, pulled ~7 GB of hot experts back in.
- **`LLAMA_MOE_HOT_MLOCK=1`** locks the cold experts (35.6 GiB here) in RAM, so other programs cannot evict them. Adaptation unlocks swapped-in experts and locks evicted ones in a background thread.
- **`ggml-cpu`**: `mul_mat_id` gives a zero row for a negative expert id.
- **`LLAMA_MOE_HOT_SRC=<file>`**: the hot copies are read from another single-file GGUF of the same model and keep its types, so hot and cold experts can have different precision. Adaptation copies from the same file.
- **`ggml-cpu`, Q2_0 on x86**: upstream maps the Q2_0 dot product to the scalar code on x86, 48 cycles per 32 weights on Zen 4, which made Q2_0 experts on the CPU slower than IQ3_XXS ones. Now AVX2 (5.0 cycles) and AVX-512 VBMI (3.4 cycles: `vpmultishiftqb` unpacks 32 two-bit weights at once).
- **`qwen4exp`**: the transposed mat-vec for the `hc_*_inject` weights only for F32/F16; BF16 weights (GSQ-RCO) hit a Vulkan assert.
- **`qwen4exp`**, ported from other branches (see Credits): the MTP head can come from a sidecar GGUF (`--spec-type draft-mtp -md <file>`), and `--lazy-mode` reads the big n-gram embedding table on demand.

## Usage

1. Build with `-DGGML_VULKAN=ON` as usual.
2. Collect routing counts: run `llama-imatrix` on text like your workload with GGUF output. It stores per-expert counts in `blk.N.ffn_gate_exps.weight.counts`. Several files, e.g. code and prose, can be mixed.
3. Make a list for a VRAM budget: what is left after the dense weights, KV cache, compute buffers and the MTP head.
   ```bash
   python3 tools/moe-hot/moe_hot_list.py hot.txt 12.5 imatrix-code.gguf imatrix-prose.gguf --model 'model-*-of-*.gguf'
   ```
   Expert sizes come from `--model`: with `LLAMA_MOE_HOT_SRC`, give the file the hot copies are read from. The scripts need this repo's `gguf-py` for newer types such as Q2_0 (`PYTHONPATH=gguf-py`).
4. Run:
   ```bash
   MODEL=model-00001-of-00033.gguf HOT_LIST=hot.txt MTP=mtp-head.gguf tools/moe-hot/run-hot.sh
   ```
5. System settings (Linux, AMD), both need root:
   - **GPU clock.** Pin it high:
     ```bash
     echo high | sudo tee /sys/class/drm/card1/device/power_dpm_force_performance_level
     ```
     With `auto`, the GPU idles while the CPU computes the cold experts, and DPM keeps it at ~1.5 GHz. The `COMPUTE` power profile did not help. The setting resets on reboot. drluoto gives the same advice in [flash-next-strix-halo](https://github.com/drluoto/flash-next-strix-halo).
   - **Memlock limit for `LLAMA_MOE_HOT_MLOCK`.** Set it to at least the size of the cold experts, e.g. with `/etc/systemd/system/user@.service.d/memlock.conf` containing `[Service]` and `LimitMEMLOCK=40G`, then log in again. Without it the server warns and runs unlocked.

| env var | default | |
|---|---|---|
| `LLAMA_MOE_HOT` | unset | hot expert list |
| `LLAMA_MOE_HOT_SRC` | unset | one GGUF file of another quant of the model; the hot copies are read from it |
| `LLAMA_MOE_HOT_ADAPT` | 0 (off) | adaptation period in tokens (the script sets 256) |
| `LLAMA_MOE_HOT_SWAPS` / `_DECAY` / `_HYST` | 64 / 0.8 / 2.0 | swaps per step, decay of use counts, how much more a cold expert must be used |
| `LLAMA_MOE_HOT_MLOCK` | 0 | lock the cold experts in RAM |
| `LLAMA_MOE_HOT_DEBUG` | 0 | 1: print the share of expert calls served hot; 2: also drop the cold experts (wrong output, timing only) |
| `GGML_VK_SMALL_GRAPH_GFLOPS` | 0 (off) | graphs below this many GFLOP are submitted once |
| `GGML_SCHED_LEGACY_COPY` | unset | old scheduler input copies |

## Pitfalls found on the way

- **Repeated expert ids.** Vulkan `mul_mm_id` (batches over 8 tokens) assumes each expert appears at most once per token. A repeated id leaves rows unwritten, and the output turns to garbage.
- **Batches of 32+ tokens.** Here (`GGML_OP_OFFLOAD_MIN_BATCH`) the scheduler moves the cold part to the GPU, where `-1` ids are not allowed.
- **CPU cost.** IQ2_S experts on the CPU are compute bound, not memory bound. More threads than cores (SMT) made it much slower, 6 threads instead of 8 slower too.
- **Measuring.** A cold page cache makes the first pass after a load useless, so compare second passes and check major faults in `/proc/PID/stat`. A full zram swap made the kernel evict the experts, and prefill fell from 234 to 57 t/s.
- **MTP draft length.** A different draft length changes the text (the batched verification is numerically different), so compare means over several prompts. Here 3 draft tokens was best; longer drafts and a `p_min` cutoff did not help.
- **Trimmed-vocabulary MTP heads (`frspec-65k`).** They can only draft the 65k most frequent tokens. The Q8_0-frspec-65k head instead of the full-vocabulary Q4_K_M (same hot list) cut Russian acceptance from 60% to 25%, and decode from 29.6 to 18.4 t/s. English prose acceptance fell from 69% to 62% (-10% speed); code was unchanged. For languages other than English, keep a full-vocabulary head.

## Limitations

- Only `qwen4exp` is wired. The hot/cold split lives in the generic `build_moe_ffn`, so another arch needs one change: pass `layer.moe_hot` to its `build_moe_ffn` call. Other archs are untested.
- The GGUF must keep gate/up/down experts as separate tensors, without expert biases or per-expert scales. Layers that do not are skipped with a warning; gpt-oss, for example, has expert biases.
- The scheduler and Vulkan changes are not tied to a model: they apply to any MoE run with the experts on the CPU.
- Configuration is by env vars. The page cache and mlock parts are Linux only. Only RDNA3 with RADV was tested.

## Credits

- [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT) by ggml-org and contributors.
- The first commit is ported, not written here: the qwen4exp NextN/MTP draft head by Ryan Monsurate; loading a detached MTP head GGUF by crusaderky (crusaderky/llama.cpp@a82a58a), as combined by drluoto in [drluoto/llama.cpp](https://github.com/drluoto/llama.cpp/tree/strix-halo-vulkan); lazy reads of the PLE table (`--lazy-mode`, `llama-lazy-reader.h`) by Josh Leverette, [ggml-org/llama.cpp#28136](https://github.com/ggml-org/llama.cpp/pull/28136).
- The MTP head GGUF: [drluoto/Qwen3.8-Flash-Next-MTP-GGUF](https://huggingface.co/drluoto/Qwen3.8-Flash-Next-MTP-GGUF). The model: AtomicChat and Navin-Models, linked above.
- The hybrid: GSQ + RCO and the quantized weights by IST-DASLab, the abliterated weights by orcarouter, the abliterated GGUFs by SC117; the base model by Qwen.
- The other commits were developed with AI assistance (Claude); they carry an `Assisted-by` trailer.
