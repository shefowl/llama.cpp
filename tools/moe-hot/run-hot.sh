#!/bin/bash
# Example launcher: routed experts on the CPU, hot copies in VRAM (LLAMA_MOE_HOT), optional MTP.
# usage: MODEL=<first .gguf shard> HOT_LIST=<list> [MTP=<mtp head .gguf>] [CTX=32768] [PORT=8080] [THREADS=8] \
#        tools/moe-hot/run-hot.sh [extra llama-server args]
# These are the settings the numbers in README.md were measured with.
DIR=$(cd "$(dirname "$0")" && pwd)
BIN="${BIN:-$DIR/../../build/bin}"
: "${MODEL:?set MODEL to the model GGUF (the first shard if split)}"
: "${HOT_LIST:?set HOT_LIST (see moe_hot_list.py)}"
PORT="${PORT:-8080}"
THREADS="${THREADS:-8}"

export LLAMA_MOE_HOT="$HOT_LIST"
export LLAMA_MOE_HOT_ADAPT="${LLAMA_MOE_HOT_ADAPT:-256}"
export LLAMA_MOE_HOT_MLOCK="${LLAMA_MOE_HOT_MLOCK:-1}"
export GGML_VK_SMALL_GRAPH_GFLOPS="${GGML_VK_SMALL_GRAPH_GFLOPS:-20}"
export GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1

SPEC=()
[ -n "$MTP" ] && SPEC=(--spec-type draft-mtp -md "$MTP" --spec-draft-n-max "${DRAFT_N:-3}")

# split models: warm all shards
case "$MODEL" in
    *-00001-of-*.gguf) GLOB="${MODEL%-00001-of-*}-*-of-*.gguf" ;;
    *)                 GLOB="$MODEL" ;;
esac

"$BIN/llama-server" -m "$MODEL" --port "$PORT" -np 1 \
    --device Vulkan0 -ngl 99 -ot exps=CPU -fit off \
    -c "${CTX:-32768}" -b 2048 -ub 1024 \
    -t "$THREADS" --cpu-range 0-$((THREADS - 1)) --cpu-strict 1 --poll 100 \
    -ctk q4_0 -ctv q4_0 --flash-attn on --load-mode mmap \
    "${SPEC[@]}" "$@" &
PID=$!
trap 'kill $PID 2>/dev/null' INT TERM
until curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; do
    kill -0 $PID 2>/dev/null || exit 1
    sleep 1
done
python3 "$DIR/warm_experts.py" "$GLOB" "$HOT_LIST" ${MTP:+"$MTP"}
echo "locked in RAM: $(awk '/VmLck/ {printf "%.1f GiB", $2/1048576}' /proc/$PID/status)"
wait $PID
