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

## What changed

- **`LLAMA_MOE_HOT=<list>`** (`src/llama-model.cpp`, `src/llama-graph.cpp`). The list names the hot experts per layer. At load they are copied into VRAM. `build_moe_ffn` splits each MoE layer into a hot part (GPU, the copies) and a cold part (CPU, the mmap), with four small CPU ops that remap the router ids and weights. Hot fillers are distinct unused slots. Cold ids are `-1` for small batches (the CPU writes a zero row). For batches of 32+ tokens the scheduler may run the cold part on the GPU; there cold ids are distinct zero-weight fillers.
- **`LLAMA_MOE_HOT_ADAPT=<tokens>`**: swaps experts between hot and cold at run time from decayed use counts. In a long one-topic session this gave +21%; when the topic changes every few prompts, about -5%.
- **Scheduler** (`ggml/src/ggml-backend.cpp`). Without events, the split backend is synchronized once per split: a sync before every input only flushed the async copies of the previous inputs. Device-to-host inputs are queued with `get_tensor_async` and waited for once. Before this, each layer had about 6 GPU waits, now 1.
- **`GGML_VK_SMALL_GRAPH_GFLOPS=20`** (`ggml-vulkan.cpp`). A per-layer decode graph is submitted once, instead of about 6 flops-based chunks plus the `almost_ready` fence. Prefill graphs stay above the threshold.
- **Page cache.** After the load, the pages of the weights that went to the GPU and of the hot copies are dropped (`MADV_PAGEOUT`). `warm_experts.py` reads the cold experts without readahead: readahead, and btrfs compressed extents, pulled ~7 GB of hot experts back in.
- **`LLAMA_MOE_HOT_MLOCK=1`** locks the cold experts (35.6 GiB here) in RAM, so other programs cannot evict them. Adaptation unlocks swapped-in experts and locks evicted ones in a background thread.
- **`ggml-cpu`**: `mul_mat_id` gives a zero row for a negative expert id.
- **`qwen4exp`**, ported from other branches (see Credits): the MTP head can come from a sidecar GGUF (`--spec-type draft-mtp -md <file>`), and `--lazy-mode` reads the big n-gram embedding table on demand.

## Usage

1. Build with `-DGGML_VULKAN=ON` as usual.
2. Collect routing counts: run `llama-imatrix` on text like your workload with GGUF output. It stores per-expert counts in `blk.N.ffn_gate_exps.weight.counts`. Several files, e.g. code and prose, can be mixed.
3. Make a list for a VRAM budget: what is left after the dense weights, KV cache, compute buffers and the MTP head.
   ```bash
   python3 tools/moe-hot/moe_hot_list.py hot.txt 12.5 imatrix-code.gguf imatrix-prose.gguf --model 'model-*-of-*.gguf'
   ```
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

## Limitations

- Only `qwen4exp` is wired. For another arch, pass `layer.moe_hot` to `build_moe_ffn`. Fused `gate_up`, expert biases (gpt-oss) and expert scales are not supported.
- Configuration is by env vars. The page cache and mlock parts are Linux only. Only RDNA3 with RADV was tested.

## Credits

- [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT) by ggml-org and contributors.
- The first commit is ported, not written here: the qwen4exp NextN/MTP draft head by Ryan Monsurate; loading a detached MTP head GGUF by crusaderky (crusaderky/llama.cpp@a82a58a), as combined by drluoto in [drluoto/llama.cpp](https://github.com/drluoto/llama.cpp/tree/strix-halo-vulkan); lazy reads of the PLE table (`--lazy-mode`, `llama-lazy-reader.h`) by Josh Leverette, [ggml-org/llama.cpp#28136](https://github.com/ggml-org/llama.cpp/pull/28136).
- The MTP head GGUF: [drluoto/Qwen3.8-Flash-Next-MTP-GGUF](https://huggingface.co/drluoto/Qwen3.8-Flash-Next-MTP-GGUF). The model: AtomicChat and Navin-Models, linked above.
- The other commits were developed with AI assistance (Claude); they carry an `Assisted-by` trailer.
