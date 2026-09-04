
#!/usr/bin/env bash
set -euo pipefail

# DFlash2 wrapper. We serve Qwen3.8-27B with the DFlash2 block-diffusion
# draft instead of start.sh's EAGLE/MTP, by injecting EXTRA_ARGS (appended
# last, argparse last-wins). The draft is pinned to DRAFT_MODEL@DRAFT_REVISION
# (z-lab's DFlash2 draft; incoai/... is a mirror of the same weights).
# Self-contained: if the derived image is missing, it is built automatically
# from patch/ (patch/build-dflash2-image.sh + overlay-dflash2/; needs git +
# network on first build).
# CRASH RULES (NVFP4): --mem-fraction-static 0.90 (0.95 hard-rebooted the
# GB10 once at draft-graph capture). Default DF_TARGET=nvfp4 is the
# BF16-lm_head export (dense head; DFLASH selector works without the
# quantized-head patch). Packed-FP4-head (DF_TARGET=nvfp4-fp4) still
# needs the image patch or the first request dies ("DFlash2 selector
# requires a dense ... target lm_head"). DFLASH requires --mamba-radix-cache-strategy extra_buffer
# (extra_buffer_lazy is rejected).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Ensure the draft (and only the draft) is cached where the container reads it.
HF_CACHE="${SCRIPT_DIR}/.cache/huggingface/hub"
mkdir -p "${HF_CACHE}"

DRAFT_MODEL="${DRAFT_MODEL:-z-lab/Qwen3.8-27B-DFlash2}"
DRAFT_REVISION="${DRAFT_REVISION:-50307d4c4cde6860d4eee73e2547cd786fe8e8a4}"

snapshot_present() {
  local base="${HF_CACHE}/models--${1//\//--}"
  local rev=""
  [[ -n "${DRAFT_REVISION}" ]] && rev="/snapshots/${DRAFT_REVISION}" || rev="/snapshots"
  [[ -n "$(find -L "${base}${rev}" -maxdepth 2 -type f -print -quit 2>/dev/null)" ]]
}

if [[ -z "${HF_TOKEN:-}" && -f "${HOME}/.bashrc" ]]; then
  HF_TOKEN="$(sed -n 's/^[[:space:]]*\(export[[:space:]]\+\)\?HF_TOKEN=["'"'"']\?\([A-Za-z0-9_-]\+\).*/\2/p' "${HOME}/.bashrc" | head -1)"
fi
export HF_TOKEN

# Same for HF_ENDPOINT (hf-mirror etc.): used by the draft-pull container.
if [[ -z "${HF_ENDPOINT:-}" && -f "${HOME}/.bashrc" ]]; then
  HF_ENDPOINT="$(sed -n 's/^[[:space:]]*\(export[[:space:]]\+\)\?HF_ENDPOINT=["'"'"']\?\([A-Za-z0-9_.\/:-]\+\).*/\2/p' "${HOME}/.bashrc" | head -1)"
fi
export HF_ENDPOINT

IMAGE="${IMAGE:-lmsysorg/sglang:latest-dflash2}"
case "${IMAGE}" in *-minoverlay) build_mode=(--minimal) ;; *) build_mode=() ;; esac
ensure_image() {
  if docker image inspect "${IMAGE}" >/dev/null 2>&1; then
    echo "Using ${IMAGE}"
  else
    echo "${IMAGE} not present locally — building DFlash2 image ..."
    "${SCRIPT_DIR}/patch/build-dflash2-image.sh" "${build_mode[@]}" || { echo "DFLASH2 image build failed"; exit 1; }
  fi
}
ensure_image
export IMAGE

ensure_cached() {
  local repo="$1"
  local label="${repo}${DRAFT_REVISION:+ @ ${DRAFT_REVISION}}"
  if snapshot_present "${repo}"; then
    echo "draft already cached (${label})"
  else
    echo "draft not cached — pulling ${label} ..."
    docker run --rm --network host \
      -e HF_HOME=/root/.cache/huggingface \
      -e HF_TOKEN="${HF_TOKEN:-}" \
      -e HF_ENDPOINT="${HF_ENDPOINT:-}" \
      -v "${SCRIPT_DIR}/.cache/huggingface:/root/.cache/huggingface" \
      "${IMAGE}" \
      python3 -c "from huggingface_hub import snapshot_download; snapshot_download('${repo}'${DRAFT_REVISION:+, revision='${DRAFT_REVISION}'})" \
      || { echo "pull failed for ${label}"; exit 1; }
    snapshot_present "${repo}" || { echo "pull failed for ${label}"; exit 1; }
  fi
}
ensure_cached "${DRAFT_MODEL}"

DF_TARGET="${DF_TARGET:-nvfp4}"
case "${DF_TARGET}" in
  bf16) TARGET_PATH="Qwen/Qwen3.8-27B" ;;
  nvfp4|nvfp4-bf16|nvfp4-bf16-head)
        TARGET_PATH="RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead" ;;
  nvfp4-fp4|nvfp4-fp4-head)
        TARGET_PATH="RadixArk/Qwen3.8-27B-NVFP4" ;;
  *) echo "DF_TARGET must be bf16, nvfp4, or nvfp4-fp4, got '${DF_TARGET}'"; exit 1 ;;
esac

EXTRA_ARGS="--model-path ${TARGET_PATH} \
--speculative-algorithm DFLASH \
--speculative-draft-model-path ${DRAFT_MODEL}${DRAFT_REVISION:+ --speculative-draft-model-revision ${DRAFT_REVISION}} \
--speculative-num-draft-tokens 8 \
--torch-compile-max-bs 8 \
--enable-torch-compile \
--kv-cache-dtype fp8_e4m3 \
--speculative-draft-kv-cache-dtype fp8_e4m3 \
--stream-interval 2 \
--tokenizer-worker-num 4 \
--detokenizer-worker-num 4 \
--fp8-gemm-backend flashinfer_cutlass \
--fp4-gemm-backend flashinfer_cutlass \
--attention-backend flashinfer \
--tokenizer-mode auto \
--mamba-backend flashinfer \
--linear-attn-decode-backend cutedsl \
--tokenizer-backend fastokens \
--max-mamba-cache-size 72 \
--chunked-prefill-size 4096 \
--max-running-requests 8 \
--mamba-ssm-dtype bfloat16 \
--enable-streaming-session \
--mamba-full-memory-ratio 4.21 \
--reasoning-parser qwen3 \
--tool-call-parser qwen3_coder \
--mamba-radix-cache-strategy extra_buffer"
case "${DF_TARGET}" in
  nvfp4|nvfp4-bf16|nvfp4-bf16-head|nvfp4-fp4|nvfp4-fp4-head)
    EXTRA_ARGS+=" --mem-fraction-static 0.75" ;;
esac
EXTRA_ARGS+=" --cuda-graph-max-bs-decode 2"
EXTRA_ARGS+=" --cuda-graph-bs-decode 1 2"
EXTRA_ARGS+=" --num-continuous-decode-steps 4"
EXTRA_ARGS+=" ${DF_EXTRA:-}"
export EXTRA_ARGS

echo "DFlash mode: EXTRA_ARGS=${EXTRA_ARGS}"
echo "Delegating to ${SCRIPT_DIR}/start.sh"
exec "${SCRIPT_DIR}/start.sh"
